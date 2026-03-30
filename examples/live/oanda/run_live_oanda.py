#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  OANDA Live Trading Script — Strategy Arena
#
#  Runs top-performing strategies from the backtesting arena on OANDA Practice.
#  Uses the OANDA v20 REST API directly (no NautilusTrader live node required).
#
#  Environment variables required:
#    OANDA_API_TOKEN   — your OANDA personal access token
#    OANDA_ACCOUNT_ID  — your OANDA account ID (e.g. 101-004-XXXXXXX-001)
#
#  Usage:
#    python3 run_live_oanda.py                      # default: EMA Cross on EUR/USD
#    python3 run_live_oanda.py --strategy aroon      # Aroon Trend (arena champion)
#    python3 run_live_oanda.py --strategy rsi_mean   # RSI Mean Reversion
#    python3 run_live_oanda.py --pair GBP_USD        # different pair
#    python3 run_live_oanda.py --list                # list all strategies
#    python3 run_live_oanda.py --stop                # close all trades & cancel orders
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal

import requests

# ---------------------------------------------------------------------------
# OANDA API client (synchronous, simple)
# ---------------------------------------------------------------------------

class OandaClient:
    """Simple synchronous OANDA v20 REST API client."""

    def __init__(self, token: str, account_id: str, practice: bool = True):
        self.token = token
        self.account_id = account_id
        base = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
        self.base_url = f"{base}/v3"
        self.stream_base = (
            "https://stream-fxpractice.oanda.com/v3"
            if practice
            else "https://stream-fxtrade.oanda.com/v3"
        )
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = requests.get(f"{self.base_url}{path}", headers=self.headers, params=params)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict) -> dict:
        r = requests.post(f"{self.base_url}{path}", headers=self.headers, json=data)
        r.raise_for_status()
        return r.json()

    def _put(self, path: str, data: dict | None = None) -> dict:
        r = requests.put(f"{self.base_url}{path}", headers=self.headers, json=data or {})
        r.raise_for_status()
        return r.json()

    # -- Account -----------------------------------------------------------------
    def account_summary(self) -> dict:
        return self._get(f"/accounts/{self.account_id}/summary")["account"]

    # -- Market Data -------------------------------------------------------------
    def candles(self, instrument: str, granularity: str = "M5",
                count: int = 200, price: str = "MBA") -> list[dict]:
        resp = self._get(f"/instruments/{instrument}/candles", {
            "granularity": granularity, "count": str(count), "price": price,
        })
        return [c for c in resp.get("candles", []) if c.get("complete")]

    def current_price(self, instrument: str) -> dict:
        resp = self._get(f"/accounts/{self.account_id}/pricing", {
            "instruments": instrument,
        })
        prices = resp.get("prices", [])
        return prices[0] if prices else {}

    # -- Orders ------------------------------------------------------------------
    def market_order(self, instrument: str, units: int) -> dict:
        """Place a market order. Positive units = BUY, negative = SELL."""
        return self._post(f"/accounts/{self.account_id}/orders", {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        })

    def limit_order(self, instrument: str, units: int, price: str,
                    time_in_force: str = "GTC") -> dict:
        return self._post(f"/accounts/{self.account_id}/orders", {
            "order": {
                "type": "LIMIT",
                "instrument": instrument,
                "units": str(units),
                "price": price,
                "timeInForce": time_in_force,
                "positionFill": "DEFAULT",
            }
        })

    def cancel_all_orders(self, instrument: str | None = None) -> int:
        params = {}
        if instrument:
            params["instrument"] = instrument
        resp = self._get(f"/accounts/{self.account_id}/orders", params or None)
        orders = resp.get("orders", [])
        count = 0
        for o in orders:
            try:
                self._put(f"/accounts/{self.account_id}/orders/{o['id']}/cancel")
                count += 1
            except Exception:
                pass
        return count

    # -- Trades ------------------------------------------------------------------
    def open_trades(self, instrument: str | None = None) -> list[dict]:
        params = {"state": "OPEN"}
        if instrument:
            params["instrument"] = instrument
        return self._get(f"/accounts/{self.account_id}/trades", params).get("trades", [])

    def close_trade(self, trade_id: str, units: str = "ALL") -> dict:
        return self._put(f"/accounts/{self.account_id}/trades/{trade_id}/close",
                         {"units": units})

    def close_all_trades(self, instrument: str | None = None) -> int:
        trades = self.open_trades(instrument)
        count = 0
        for t in trades:
            try:
                self.close_trade(t["id"])
                count += 1
            except Exception:
                pass
        return count

    # -- Positions ---------------------------------------------------------------
    def position(self, instrument: str) -> dict | None:
        try:
            return self._get(
                f"/accounts/{self.account_id}/positions/{instrument}"
            ).get("position")
        except Exception:
            return None

    def net_position_units(self, instrument: str) -> int:
        """Return net position units (positive=long, negative=short, 0=flat)."""
        pos = self.position(instrument)
        if not pos:
            return 0
        long_units = int(pos.get("long", {}).get("units", "0"))
        short_units = int(pos.get("short", {}).get("units", "0"))
        return long_units + short_units


# ---------------------------------------------------------------------------
# Indicators (pure Python, no NautilusTrader dependency)
# ---------------------------------------------------------------------------

def ema(values: list[float], period: int) -> list[float]:
    """Exponential Moving Average."""
    result = []
    k = 2.0 / (period + 1)
    prev = values[0]
    for v in values:
        prev = v * k + prev * (1 - k)
        result.append(prev)
    return result


def sma(values: list[float], period: int) -> list[float]:
    """Simple Moving Average."""
    result = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(sum(values[:i+1]) / (i + 1))
        else:
            result.append(sum(values[i-period+1:i+1]) / period)
    return result


def rsi(closes: list[float], period: int = 14) -> list[float]:
    """Relative Strength Index."""
    result = [50.0]  # seed
    gains, losses = 0.0, 0.0
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        if i <= period:
            if delta > 0:
                gains += delta
            else:
                losses -= delta
            if i == period:
                avg_gain = gains / period
                avg_loss = losses / period
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
                result.append(100 - 100 / (1 + rs))
            else:
                result.append(50.0)
        else:
            gain = max(delta, 0)
            loss = max(-delta, 0)
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            result.append(100 - 100 / (1 + rs))
    return result


def atr(highs: list[float], lows: list[float], closes: list[float],
        period: int = 14) -> list[float]:
    """Average True Range."""
    trs = [highs[0] - lows[0]]
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return ema(trs, period)


def aroon_osc(highs: list[float], lows: list[float], period: int = 25) -> list[float]:
    """Aroon Oscillator."""
    result = []
    for i in range(len(highs)):
        window = min(i + 1, period)
        h_slice = highs[max(0, i - period + 1):i + 1]
        l_slice = lows[max(0, i - period + 1):i + 1]
        days_since_high = window - 1 - h_slice.index(max(h_slice))
        days_since_low = window - 1 - l_slice.index(min(l_slice))
        aroon_up = ((period - days_since_high) / period) * 100
        aroon_down = ((period - days_since_low) / period) * 100
        result.append(aroon_up - aroon_down)
    return result


def bollinger_bands(closes: list[float], period: int = 20,
                    std_dev: float = 2.0) -> tuple[list[float], list[float], list[float]]:
    """Bollinger Bands: (upper, middle, lower)."""
    mid = sma(closes, period)
    upper, lower = [], []
    for i in range(len(closes)):
        window = closes[max(0, i - period + 1):i + 1]
        mean = mid[i]
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        sd = variance ** 0.5
        upper.append(mean + std_dev * sd)
        lower.append(mean - std_dev * sd)
    return upper, mid, lower


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class BaseStrategy:
    """Base class for live strategies."""

    name: str = "base"

    def __init__(self, client: OandaClient, instrument: str, units: int = 100):
        self.client = client
        self.instrument = instrument
        self.units = units

    def compute_signal(self, candles: list[dict]) -> str:
        """Return 'BUY', 'SELL', or 'HOLD'."""
        raise NotImplementedError

    def execute(self, signal: str) -> str | None:
        """Execute the signal, managing position."""
        current_pos = self.client.net_position_units(self.instrument)

        if signal == "BUY" and current_pos <= 0:
            # Close short if any, then go long
            if current_pos < 0:
                self.client.close_all_trades(self.instrument)
            resp = self.client.market_order(self.instrument, self.units)
            fill = resp.get("orderFillTransaction", {})
            return f"BUY {self.units} @ {fill.get('price', '?')}"

        elif signal == "SELL" and current_pos >= 0:
            # Close long if any, then go short
            if current_pos > 0:
                self.client.close_all_trades(self.instrument)
            resp = self.client.market_order(self.instrument, -self.units)
            fill = resp.get("orderFillTransaction", {})
            return f"SELL {self.units} @ {fill.get('price', '?')}"

        return None  # HOLD or already in position


class EMACrossStrategy(BaseStrategy):
    """EMA Cross — fast/slow crossover."""
    name = "ema_cross"

    def __init__(self, client, instrument, units=100, fast=10, slow=20):
        super().__init__(client, instrument, units)
        self.fast = fast
        self.slow = slow

    def compute_signal(self, candles):
        closes = [float(c["mid"]["c"]) for c in candles]
        fast_ema = ema(closes, self.fast)
        slow_ema = ema(closes, self.slow)
        if fast_ema[-1] > slow_ema[-1] and fast_ema[-2] <= slow_ema[-2]:
            return "BUY"
        elif fast_ema[-1] < slow_ema[-1] and fast_ema[-2] >= slow_ema[-2]:
            return "SELL"
        return "HOLD"


class AroonTrendStrategy(BaseStrategy):
    """Aroon Trend — the arena champion."""
    name = "aroon_trend"

    def __init__(self, client, instrument, units=100, period=25):
        super().__init__(client, instrument, units)
        self.period = period

    def compute_signal(self, candles):
        highs = [float(c["mid"]["h"]) for c in candles]
        lows = [float(c["mid"]["l"]) for c in candles]
        osc = aroon_osc(highs, lows, self.period)
        if osc[-1] > 50 and osc[-2] <= 50:
            return "BUY"
        elif osc[-1] < -50 and osc[-2] >= -50:
            return "SELL"
        return "HOLD"


class RSIMeanRevStrategy(BaseStrategy):
    """RSI Mean Reversion — buy oversold, sell overbought."""
    name = "rsi_mean"

    def __init__(self, client, instrument, units=100, period=14,
                 oversold=30, overbought=70):
        super().__init__(client, instrument, units)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def compute_signal(self, candles):
        closes = [float(c["mid"]["c"]) for c in candles]
        rsi_vals = rsi(closes, self.period)
        if rsi_vals[-1] < self.oversold:
            return "BUY"
        elif rsi_vals[-1] > self.overbought:
            return "SELL"
        return "HOLD"


class BollingerReversionStrategy(BaseStrategy):
    """Bollinger Band Mean Reversion."""
    name = "bollinger"

    def __init__(self, client, instrument, units=100, period=20, std_dev=2.0):
        super().__init__(client, instrument, units)
        self.period = period
        self.std_dev = std_dev

    def compute_signal(self, candles):
        closes = [float(c["mid"]["c"]) for c in candles]
        upper, mid, lower = bollinger_bands(closes, self.period, self.std_dev)
        if closes[-1] < lower[-1]:
            return "BUY"
        elif closes[-1] > upper[-1]:
            return "SELL"
        return "HOLD"


class SuperTrendStrategy(BaseStrategy):
    """SuperTrend — ATR-based trend following."""
    name = "supertrend"

    def __init__(self, client, instrument, units=100, atr_period=10,
                 multiplier=3.0):
        super().__init__(client, instrument, units)
        self.atr_period = atr_period
        self.multiplier = multiplier

    def compute_signal(self, candles):
        highs = [float(c["mid"]["h"]) for c in candles]
        lows = [float(c["mid"]["l"]) for c in candles]
        closes = [float(c["mid"]["c"]) for c in candles]
        atr_vals = atr(highs, lows, closes, self.atr_period)

        # Compute SuperTrend
        upper_band = [(h + l) / 2 + self.multiplier * a for h, l, a in zip(highs, lows, atr_vals)]
        lower_band = [(h + l) / 2 - self.multiplier * a for h, l, a in zip(highs, lows, atr_vals)]

        supertrend = [0.0] * len(closes)
        direction = [1] * len(closes)  # 1 = up, -1 = down

        for i in range(1, len(closes)):
            if closes[i] > upper_band[i - 1]:
                direction[i] = 1
            elif closes[i] < lower_band[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        if direction[-1] == 1 and direction[-2] == -1:
            return "BUY"
        elif direction[-1] == -1 and direction[-2] == 1:
            return "SELL"
        return "HOLD"


class MACDStrategy(BaseStrategy):
    """MACD crossover strategy."""
    name = "macd"

    def __init__(self, client, instrument, units=100,
                 fast=12, slow=26, signal_period=9):
        super().__init__(client, instrument, units)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def compute_signal(self, candles):
        closes = [float(c["mid"]["c"]) for c in candles]
        fast_ema = ema(closes, self.fast)
        slow_ema = ema(closes, self.slow)
        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
        signal_line = ema(macd_line, self.signal_period)

        if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
            return "BUY"
        elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
            return "SELL"
        return "HOLD"


class TripleEMAStrategy(BaseStrategy):
    """Triple EMA — 3 timeframe trend confirmation."""
    name = "triple_ema"

    def __init__(self, client, instrument, units=100,
                 fast=5, medium=13, slow=34):
        super().__init__(client, instrument, units)
        self.fast_p = fast
        self.medium_p = medium
        self.slow_p = slow

    def compute_signal(self, candles):
        closes = [float(c["mid"]["c"]) for c in candles]
        f = ema(closes, self.fast_p)
        m = ema(closes, self.medium_p)
        s = ema(closes, self.slow_p)
        if f[-1] > m[-1] > s[-1] and not (f[-2] > m[-2] > s[-2]):
            return "BUY"
        elif f[-1] < m[-1] < s[-1] and not (f[-2] < m[-2] < s[-2]):
            return "SELL"
        return "HOLD"


class RSISuperTrendCombo(BaseStrategy):
    """SuperTrend + RSI filter — best crypto strategy from arena."""
    name = "supertrend_rsi"

    def __init__(self, client, instrument, units=100,
                 atr_period=10, multiplier=3.0, rsi_period=14):
        super().__init__(client, instrument, units)
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.rsi_period = rsi_period

    def compute_signal(self, candles):
        highs = [float(c["mid"]["h"]) for c in candles]
        lows = [float(c["mid"]["l"]) for c in candles]
        closes = [float(c["mid"]["c"]) for c in candles]
        atr_vals = atr(highs, lows, closes, self.atr_period)
        rsi_vals = rsi(closes, self.rsi_period)

        # SuperTrend direction
        upper_band = [(h + l) / 2 + self.multiplier * a for h, l, a in zip(highs, lows, atr_vals)]
        lower_band = [(h + l) / 2 - self.multiplier * a for h, l, a in zip(highs, lows, atr_vals)]
        direction = [1] * len(closes)
        for i in range(1, len(closes)):
            if closes[i] > upper_band[i - 1]:
                direction[i] = 1
            elif closes[i] < lower_band[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        # Combine: SuperTrend + RSI filter
        if direction[-1] == 1 and rsi_vals[-1] < 70:
            if direction[-2] == -1:  # Fresh crossover
                return "BUY"
        elif direction[-1] == -1 and rsi_vals[-1] > 30:
            if direction[-2] == 1:
                return "SELL"
        return "HOLD"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    "ema_cross": EMACrossStrategy,
    "aroon": AroonTrendStrategy,
    "rsi_mean": RSIMeanRevStrategy,
    "bollinger": BollingerReversionStrategy,
    "supertrend": SuperTrendStrategy,
    "macd": MACDStrategy,
    "triple_ema": TripleEMAStrategy,
    "supertrend_rsi": RSISuperTrendCombo,
}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_live(strategy: BaseStrategy, interval_secs: int = 300,
             granularity: str = "M5", candle_count: int = 200):
    """
    Run a strategy live on OANDA.

    Polls candles every `interval_secs`, computes signal, and executes.
    """
    running = True

    def _handle_signal(signum, frame):
        nonlocal running
        print("\n[SHUTDOWN] Stopping strategy...")
        running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("=" * 70)
    print(f"  OANDA Live Trading — {strategy.name.upper()}")
    print(f"  Instrument: {strategy.instrument}")
    print(f"  Units: {strategy.units}")
    print(f"  Interval: {interval_secs}s | Granularity: {granularity}")
    print("=" * 70)

    # Show account info
    acct = strategy.client.account_summary()
    print(f"  Account: {acct['id']}")
    print(f"  Balance: {acct['balance']} {acct['currency']}")
    print(f"  NAV: {acct['NAV']} {acct['currency']}")
    print(f"  Open trades: {acct['openTradeCount']}")
    print("=" * 70)
    print()

    tick = 0
    while running:
        tick += 1
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        try:
            # Fetch candles
            candles = strategy.client.candles(
                strategy.instrument, granularity, candle_count,
            )
            if len(candles) < 50:
                print(f"[{now}] Tick {tick}: Not enough candle data ({len(candles)}), waiting...")
                time.sleep(interval_secs)
                continue

            # Compute signal
            sig = strategy.compute_signal(candles)

            # Current price
            last_close = float(candles[-1]["mid"]["c"])
            pos = strategy.client.net_position_units(strategy.instrument)

            # Execute
            action = strategy.execute(sig)

            status = "FLAT" if pos == 0 else (f"LONG {pos}" if pos > 0 else f"SHORT {pos}")
            msg = f"[{now}] Tick {tick}: signal={sig:4s} | price={last_close:.5f} | pos={status}"
            if action:
                msg += f" | >>> {action}"
            print(msg)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[{now}] Tick {tick}: ERROR — {e}")

        # Wait for next interval
        for _ in range(interval_secs):
            if not running:
                break
            time.sleep(1)

    # Final status
    print("\n[STOPPED] Final account state:")
    acct = strategy.client.account_summary()
    print(f"  Balance: {acct['balance']} {acct['currency']}")
    print(f"  Unrealized P&L: {acct['unrealizedPL']} {acct['currency']}")
    print(f"  Open trades: {acct['openTradeCount']}")


def main():
    parser = argparse.ArgumentParser(description="OANDA Live Trading — Strategy Arena")
    parser.add_argument("--strategy", "-s", default="ema_cross",
                        help=f"Strategy name: {', '.join(STRATEGIES.keys())}")
    parser.add_argument("--pair", "-p", default="EUR_USD",
                        help="OANDA instrument (e.g. EUR_USD, GBP_USD, XAU_USD)")
    parser.add_argument("--units", "-u", type=int, default=100,
                        help="Trade size in units (default: 100)")
    parser.add_argument("--interval", "-i", type=int, default=300,
                        help="Polling interval in seconds (default: 300 = 5min)")
    parser.add_argument("--granularity", "-g", default="M5",
                        help="Candle granularity (S5,M1,M5,M15,M30,H1,H4,D)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all available strategies")
    parser.add_argument("--stop", action="store_true",
                        help="Close all trades and cancel all orders")
    parser.add_argument("--status", action="store_true",
                        help="Show account status and open positions")
    args = parser.parse_args()

    # Get credentials
    token = os.environ.get("OANDA_API_TOKEN")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not token or not account_id:
        print("ERROR: Set OANDA_API_TOKEN and OANDA_ACCOUNT_ID environment variables")
        sys.exit(1)

    client = OandaClient(token, account_id, practice=True)

    # List strategies
    if args.list:
        print("\nAvailable strategies:")
        print("-" * 50)
        for key, cls in STRATEGIES.items():
            print(f"  {key:20s} — {cls.__doc__.strip()}")
        print()
        return

    # Stop all
    if args.stop:
        n_trades = client.close_all_trades()
        n_orders = client.cancel_all_orders()
        print(f"Closed {n_trades} trades, cancelled {n_orders} orders")
        return

    # Status
    if args.status:
        acct = client.account_summary()
        print(f"\nAccount: {acct['id']}")
        print(f"Balance: {acct['balance']} {acct['currency']}")
        print(f"NAV: {acct['NAV']} {acct['currency']}")
        print(f"Unrealized P&L: {acct['unrealizedPL']} {acct['currency']}")
        print(f"Margin used: {acct['marginUsed']} {acct['currency']}")
        print(f"Margin available: {acct['marginAvailable']} {acct['currency']}")
        print(f"Open trades: {acct['openTradeCount']}")
        trades = client.open_trades()
        if trades:
            print(f"\nOpen positions:")
            for t in trades:
                print(f"  {t['instrument']:12s} {t['currentUnits']:>8s} units  "
                      f"P&L: {t.get('unrealizedPL', '?'):>10s}  "
                      f"opened: {t['openTime'][:19]}")
        print()
        return

    # Run strategy
    if args.strategy not in STRATEGIES:
        print(f"Unknown strategy: {args.strategy}")
        print(f"Available: {', '.join(STRATEGIES.keys())}")
        sys.exit(1)

    strat_cls = STRATEGIES[args.strategy]
    strategy = strat_cls(client, args.pair, args.units)

    run_live(strategy, args.interval, args.granularity)


if __name__ == "__main__":
    main()
