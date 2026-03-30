#!/usr/bin/env python3
"""
Multi-Asset Strategy Arena - Backtest 25 strategies across 4 asset classes:

  1. FX      - AUD/USD (tick data -> 1-min bars, ~100K ticks, Jan 30 2020)
  2. FX      - USD/JPY (tick data -> 1-min bars, ~1K ticks)
  3. Futures - EURUSD 6E March 2024 (1-min bars, ~30K bars, Jan 2024)
  4. Crypto  - BTC-PERP (1-min bars, ~45K bars, Dec 31 2021 - Feb 1 2022)

25 strategies x 4 assets = 100 backtests.
Each strategy gets $1M starting capital per backtest.
Results ranked per asset and combined to find the best overall strategy.
"""

import os
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader import TEST_DATA_DIR
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler
from nautilus_trader.test_kit.providers import TestDataProvider
from nautilus_trader.test_kit.providers import TestInstrumentProvider

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies import ADXTrendArena
from strategies import ADXTrendArenaConfig
from strategies import AroonTrendArena
from strategies import AroonTrendArenaConfig
from strategies import BBMeanReversionArena
from strategies import BBMeanReversionArenaConfig
from strategies import CCIBreakoutArena
from strategies import CCIBreakoutArenaConfig
from strategies import DonchianBreakoutArena
from strategies import DonchianBreakoutArenaConfig
from strategies import DoubleBBArena
from strategies import DoubleBBArenaConfig
from strategies import DualThrustArena
from strategies import DualThrustArenaConfig
from strategies import EMACrossArena
from strategies import EMACrossArenaConfig
from strategies import EMARSIComboArena
from strategies import EMARSIComboArenaConfig
from strategies import HullMACrossArena
from strategies import HullMACrossArenaConfig
from strategies import IchimokuArena
from strategies import IchimokuArenaConfig
from strategies import IchimokuRSIArena
from strategies import IchimokuRSIArenaConfig
from strategies import KeltnerSqueezeArena
from strategies import KeltnerSqueezeArenaConfig
from strategies import KeltnerTrendArena
from strategies import KeltnerTrendArenaConfig
from strategies import LinRegChannelArena
from strategies import LinRegChannelArenaConfig
from strategies import MACDTrendArena
from strategies import MACDTrendArenaConfig
from strategies import MeanRevSMAArena
from strategies import MeanRevSMAArenaConfig
from strategies import MomentumBreakoutArena
from strategies import MomentumBreakoutArenaConfig
from strategies import PivotReversalArena
from strategies import PivotReversalArenaConfig
from strategies import RSIMomentumArena
from strategies import RSIMomentumArenaConfig
from strategies import StochasticReversalArena
from strategies import StochasticReversalArenaConfig
from strategies import SuperTrendArena
from strategies import SuperTrendArenaConfig
from strategies import SuperTrendRSIArena
from strategies import SuperTrendRSIArenaConfig
from strategies import TripleEMAArena
from strategies import TripleEMAArenaConfig
from strategies import VWAPReversionArena
from strategies import VWAPReversionArenaConfig


STARTING_BALANCE = 1_000_000


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_fx_audusd():
    """Load AUD/USD tick data -> produces 1-min internal bars."""
    SIM = Venue("SIM")
    instrument = TestInstrumentProvider.default_fx_ccy("AUD/USD", SIM)
    provider = TestDataProvider()
    wrangler = QuoteTickDataWrangler(instrument=instrument)
    ticks = wrangler.process(provider.read_csv_ticks("truefx/audusd-ticks.csv"))
    bar_type = BarType.from_str("AUD/USD.SIM-1-MINUTE-MID-INTERNAL")
    return {
        "name": "FX AUD/USD",
        "venue": SIM,
        "instrument": instrument,
        "data": ticks,
        "bar_type": bar_type,
        "currency": USD,
        "trade_size": Decimal(100_000),
        "oms_type": OmsType.HEDGING,
    }


def load_fx_usdjpy():
    """Load USD/JPY tick data -> produces 1-min internal bars."""
    SIM = Venue("SIM")
    instrument = TestInstrumentProvider.default_fx_ccy("USD/JPY", SIM)
    provider = TestDataProvider()
    wrangler = QuoteTickDataWrangler(instrument=instrument)
    ticks = wrangler.process(provider.read_csv_ticks("truefx/usdjpy-ticks.csv"))
    bar_type = BarType.from_str("USD/JPY.SIM-1-MINUTE-MID-INTERNAL")
    return {
        "name": "FX USD/JPY",
        "venue": SIM,
        "instrument": instrument,
        "data": ticks,
        "bar_type": bar_type,
        "currency": USD,
        "trade_size": Decimal(100_000),
        "oms_type": OmsType.HEDGING,
    }


def load_futures_eurusd():
    """Load EURUSD futures (6E) 1-min bar data from CSV."""
    XCME = Venue("XCME")
    instrument = TestInstrumentProvider.eurusd_future(
        expiry_year=2024,
        expiry_month=3,
        venue_name="XCME",
    )
    csv_path = Path(TEST_DATA_DIR) / "xcme" / "6EH4.XCME_1min_bars_20240101_20240131.csv.gz"
    df = pd.read_csv(csv_path, header=0, index_col=False)
    df = df.reindex(columns=["timestamp_utc", "open", "high", "low", "close", "volume"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="%Y-%m-%d %H:%M:%S")
    df = df.rename(columns={"timestamp_utc": "timestamp"})
    df = df.set_index("timestamp")

    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    wrangler = BarDataWrangler(bar_type, instrument)
    bars = wrangler.process(df)

    return {
        "name": "Futures EURUSD 6E",
        "venue": XCME,
        "instrument": instrument,
        "data": bars,
        "bar_type": bar_type,
        "currency": USD,
        "trade_size": Decimal(1),  # 1 contract = 125,000 EUR notional
        "oms_type": OmsType.NETTING,
    }


def load_crypto_btcperp():
    """Load BTC-PERP 1-min bar data from CSV."""
    BINANCE = Venue("BINANCE")
    instrument = TestInstrumentProvider.btcusdt_perp_binance()

    csv_path = Path(TEST_DATA_DIR) / "btc-perp-20211231-20220201_1m.csv"
    df = pd.read_csv(csv_path, header=0, index_col=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.reindex(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.set_index("timestamp")

    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    wrangler = BarDataWrangler(bar_type, instrument)
    bars = wrangler.process(df)

    return {
        "name": "Crypto BTC-PERP",
        "venue": BINANCE,
        "instrument": instrument,
        "data": bars,
        "bar_type": bar_type,
        "currency": USDT,
        "trade_size": Decimal("0.100"),  # 0.1 BTC per trade
        "oms_type": OmsType.NETTING,
    }


# ---------------------------------------------------------------------------
# Strategy definitions (same 15 as before, with asset-specific trade sizes)
# ---------------------------------------------------------------------------
def get_strategy_defs():
    return [
        # --- Original 15 ---
        ("EMA_Cross", EMACrossArena, EMACrossArenaConfig, {
            "fast_ema_period": 10, "slow_ema_period": 20,
        }),
        ("BB_MeanRev", BBMeanReversionArena, BBMeanReversionArenaConfig, {
            "bb_period": 20, "bb_std": 2.0, "rsi_period": 14,
            "rsi_buy_threshold": 0.30, "rsi_sell_threshold": 0.70,
        }),
        ("MACD_Trend", MACDTrendArena, MACDTrendArenaConfig, {
            "macd_fast": 12, "macd_slow": 26, "ema_filter_period": 50,
        }),
        ("Donchian_Break", DonchianBreakoutArena, DonchianBreakoutArenaConfig, {
            "donchian_period": 20, "atr_period": 14, "atr_stop_multiple": 2.0,
        }),
        ("RSI_Momentum", RSIMomentumArena, RSIMomentumArenaConfig, {
            "rsi_period": 14, "ema_period": 20,
            "rsi_overbought": 0.70, "rsi_oversold": 0.30,
            "rsi_exit_upper": 0.55, "rsi_exit_lower": 0.45,
        }),
        ("Triple_EMA", TripleEMAArena, TripleEMAArenaConfig, {
            "short_period": 8, "mid_period": 21, "long_period": 55,
        }),
        ("Keltner_Squeeze", KeltnerSqueezeArena, KeltnerSqueezeArenaConfig, {
            "bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_multiplier": 1.5,
        }),
        ("Stoch_Reversal", StochasticReversalArena, StochasticReversalArenaConfig, {
            "stoch_period": 14, "overbought": 80.0, "oversold": 20.0,
        }),
        ("CCI_Breakout", CCIBreakoutArena, CCIBreakoutArenaConfig, {
            "cci_period": 20, "upper_threshold": 100.0, "lower_threshold": -100.0,
        }),
        ("MeanRev_SMA", MeanRevSMAArena, MeanRevSMAArenaConfig, {
            "sma_period": 50, "entry_deviation": 0.015, "exit_deviation": 0.002,
        }),
        ("ADX_Trend", ADXTrendArena, ADXTrendArenaConfig, {
            "adx_period": 14, "adx_threshold": 25.0,
        }),
        ("Double_BB", DoubleBBArena, DoubleBBArenaConfig, {
            "bb_period": 20, "bb_inner_std": 1.0, "bb_outer_std": 2.0,
        }),
        ("EMA_RSI_Combo", EMARSIComboArena, EMARSIComboArenaConfig, {
            "ema_period": 50, "rsi_period": 14,
            "rsi_buy_zone": 0.40, "rsi_sell_zone": 0.60,
        }),
        ("Momentum_ROC", MomentumBreakoutArena, MomentumBreakoutArenaConfig, {
            "roc_period": 12, "roc_threshold": 0.5,
        }),
        ("Aroon_Trend", AroonTrendArena, AroonTrendArenaConfig, {
            "aroon_period": 25, "threshold": 70.0,
        }),
        # --- Advanced Quant / TradingView Strategies ---
        ("SuperTrend", SuperTrendArena, SuperTrendArenaConfig, {
            "atr_period": 10, "atr_multiplier": 3.0,
        }),
        ("Ichimoku", IchimokuArena, IchimokuArenaConfig, {
            "tenkan_period": 9, "kijun_period": 26, "senkou_period": 52,
        }),
        ("VWAP_Reversion", VWAPReversionArena, VWAPReversionArenaConfig, {
            "atr_period": 14, "entry_atr_multiple": 1.5, "exit_atr_multiple": 0.3,
        }),
        ("LinReg_Channel", LinRegChannelArena, LinRegChannelArenaConfig, {
            "period": 50, "entry_std": 2.0,
        }),
        ("Dual_Thrust", DualThrustArena, DualThrustArenaConfig, {
            "lookback": 4, "k_up": 0.5, "k_down": 0.5,
        }),
        ("ST_RSI_Combo", SuperTrendRSIArena, SuperTrendRSIArenaConfig, {
            "atr_period": 10, "atr_multiplier": 3.0, "rsi_period": 14,
            "rsi_overbought": 0.70, "rsi_oversold": 0.30,
        }),
        ("Keltner_Trend", KeltnerTrendArena, KeltnerTrendArenaConfig, {
            "kc_period": 20, "kc_multiplier": 2.0,
        }),
        ("Ichimoku_RSI", IchimokuRSIArena, IchimokuRSIArenaConfig, {
            "tenkan_period": 9, "kijun_period": 26, "senkou_period": 52,
            "rsi_period": 14, "rsi_overbought": 0.65, "rsi_oversold": 0.35,
        }),
        ("Pivot_Reversal", PivotReversalArena, PivotReversalArenaConfig, {
            "swing_period": 5, "ema_period": 20,
        }),
        ("Hull_MA_Cross", HullMACrossArena, HullMACrossArenaConfig, {
            "fast_period": 9, "slow_period": 21,
        }),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _parse_pnl(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    parts = s.split()
    return float(parts[0])


def run_one(strategy_name, strategy_cls, config_cls, config_kwargs, asset_info):
    """Run a single strategy on a single asset."""
    venue = asset_info["venue"]
    instrument = asset_info["instrument"]
    bar_type = asset_info["bar_type"]
    data = asset_info["data"]
    currency = asset_info["currency"]
    trade_size = asset_info["trade_size"]
    oms = asset_info["oms_type"]

    cfg = BacktestEngineConfig(
        trader_id=TraderId(f"ARENA-{strategy_name[:10]}-001"),
        logging=LoggingConfig(log_level="ERROR"),
    )
    engine = BacktestEngine(config=cfg)
    engine.add_venue(
        venue=venue,
        oms_type=oms,
        account_type=AccountType.MARGIN,
        base_currency=currency,
        starting_balances=[Money(STARTING_BALANCE, currency)],
        default_leverage=Decimal(10),
    )
    engine.add_instrument(instrument)
    engine.add_data(data)

    strat_config = config_cls(
        instrument_id=instrument.id,
        bar_type=bar_type,
        trade_size=trade_size,
        **config_kwargs,
    )
    strat = strategy_cls(config=strat_config)
    engine.add_strategy(strat)

    engine.run()
    result = extract_results(engine, strategy_name, venue)
    engine.reset()
    engine.dispose()
    return result


def extract_results(engine, name, venue):
    fills = engine.trader.generate_order_fills_report()
    pos = engine.trader.generate_positions_report()

    total_orders = len(fills) if fills is not None and not fills.empty else 0
    total_pnl = 0.0
    winning = losing = total_pos = 0
    max_win = 0.0
    max_loss = 0.0

    if pos is not None and not pos.empty and "realized_pnl" in pos.columns:
        total_pos = len(pos)
        for p in pos["realized_pnl"]:
            v = _parse_pnl(p)
            total_pnl += v
            if v > 0:
                winning += 1
                max_win = max(max_win, v)
            elif v < 0:
                losing += 1
                max_loss = min(max_loss, v)

    win_rate = (winning / total_pos * 100) if total_pos > 0 else 0.0

    max_dd = 0.0
    if total_pos > 0:
        cum = peak = 0.0
        for p in pos["realized_pnl"]:
            cum += _parse_pnl(p)
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

    gross_profit = sum(_parse_pnl(p) for p in pos["realized_pnl"] if _parse_pnl(p) > 0) if total_pos > 0 else 0.0
    gross_loss = abs(sum(_parse_pnl(p) for p in pos["realized_pnl"] if _parse_pnl(p) < 0)) if total_pos > 0 else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg = total_pnl / total_pos if total_pos > 0 else 0.0
    ret = (total_pnl / STARTING_BALANCE) * 100

    return {
        "Strategy": name,
        "PnL": round(total_pnl, 2),
        "Return%": round(ret, 3),
        "Trades": total_pos,
        "WinRate%": round(win_rate, 1),
        "AvgTrade": round(avg, 2),
        "MaxWin": round(max_win, 2),
        "MaxLoss": round(max_loss, 2),
        "MaxDD": round(max_dd, 2),
        "PF": round(pf, 2),
    }


def print_leaderboard(title, results):
    df = pd.DataFrame(results)
    if df.empty:
        print(f"  No results for {title}")
        return df
    df = df.sort_values("PnL", ascending=False).reset_index(drop=True)
    df.index = range(1, len(df) + 1)
    df.index.name = "#"

    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", lambda x: f"{x:,.2f}"):
        print(df.to_string())

    if len(df) >= 1:
        w = df.iloc[0]
        print(f"\n  Best: {w['Strategy']} | PnL: ${w['PnL']:>+12,.2f} | Win Rate: {w['WinRate%']:.1f}% | PF: {w['PF']:.2f}")
    return df


def main():
    print()
    print("#" * 100)
    print("#  MULTI-ASSET STRATEGY ARENA")
    print("#  25 Strategies x 4 Asset Classes = 100 Backtests")
    print("#  Starting Capital: $1,000,000 per backtest")
    print("#" * 100)

    # Load all datasets
    print("\n  Loading datasets...")
    datasets = []
    for loader, label in [
        (load_fx_audusd, "FX AUD/USD"),
        (load_fx_usdjpy, "FX USD/JPY"),
        (load_futures_eurusd, "Futures 6E"),
        (load_crypto_btcperp, "Crypto BTC-PERP"),
    ]:
        print(f"    Loading {label}...", end=" ", flush=True)
        ds = loader()
        data_len = len(ds["data"])
        print(f"OK ({data_len:,} data points)")
        datasets.append(ds)

    strategies = get_strategy_defs()
    all_results = {}  # asset_name -> list of results
    grand_scores = {}  # strategy_name -> total PnL across all assets

    for ds in datasets:
        asset_name = ds["name"]
        print(f"\n{'=' * 100}")
        print(f"  RUNNING: {asset_name} ({len(ds['data']):,} data points)")
        print(f"  Trade Size: {ds['trade_size']} | OMS: {ds['oms_type']}")
        print(f"{'=' * 100}")

        results = []
        for i, (name, scls, ccls, kwargs) in enumerate(strategies, 1):
            print(f"    [{i:2d}/25] {name:<20s}...", end=" ", flush=True)
            try:
                r = run_one(name, scls, ccls, kwargs, ds)
                results.append(r)
                print(f"PnL: ${r['PnL']:>+12,.2f}  WR: {r['WinRate%']:5.1f}%  Trades: {r['Trades']}")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "Strategy": name, "PnL": 0.0, "Return%": 0.0, "Trades": 0,
                    "WinRate%": 0.0, "AvgTrade": 0.0, "MaxWin": 0.0, "MaxLoss": 0.0,
                    "MaxDD": 0.0, "PF": 0.0,
                })

        all_results[asset_name] = results
        print_leaderboard(f"LEADERBOARD: {asset_name}", results)

        for r in results:
            name = r["Strategy"]
            grand_scores[name] = grand_scores.get(name, 0.0) + r["PnL"]

    # Grand summary
    print(f"\n\n{'#' * 100}")
    print("#  GRAND SUMMARY - Combined PnL Across All Asset Classes")
    print(f"{'#' * 100}\n")

    grand_df = pd.DataFrame([
        {"Strategy": k, "Combined PnL ($)": round(v, 2)}
        for k, v in grand_scores.items()
    ]).sort_values("Combined PnL ($)", ascending=False).reset_index(drop=True)
    grand_df.index = range(1, len(grand_df) + 1)
    grand_df.index.name = "#"

    # Add per-asset columns
    for asset_name, results in all_results.items():
        pnl_map = {r["Strategy"]: r["PnL"] for r in results}
        short_name = asset_name.split()[-1]  # "AUD/USD", "6E", "BTC-PERP"
        grand_df[f"PnL {short_name}"] = grand_df["Strategy"].map(pnl_map).fillna(0.0).round(2)

    # Add per-asset win rates
    for asset_name, results in all_results.items():
        wr_map = {r["Strategy"]: r["WinRate%"] for r in results}
        short_name = asset_name.split()[-1]
        grand_df[f"WR% {short_name}"] = grand_df["Strategy"].map(wr_map).fillna(0.0).round(1)

    with pd.option_context("display.max_columns", None, "display.width", 250, "display.float_format", lambda x: f"{x:,.2f}"):
        print(grand_df.to_string())

    # Winner
    w = grand_df.iloc[0]
    print(f"\n{'=' * 100}")
    print(f"  OVERALL CHAMPION: {w['Strategy']}")
    print(f"  Combined PnL across all assets: ${w['Combined PnL ($)']:>+12,.2f}")
    print(f"{'=' * 100}\n")

    # Per-asset winners
    for asset_name, results in all_results.items():
        best = max(results, key=lambda r: r["PnL"])
        print(f"  Best on {asset_name:<20s}: {best['Strategy']:<20s} PnL: ${best['PnL']:>+12,.2f}  WR: {best['WinRate%']:.1f}%")

    print()
    return grand_df


if __name__ == "__main__":
    main()
