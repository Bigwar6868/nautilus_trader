#!/usr/bin/env python3
"""
Strategy Arena - Backtest 15 trading strategies head-to-head on AUD/USD tick data.

Strategies tested:
  1.  EMA Cross           - Trend-following EMA crossover
  2.  BB Mean Reversion   - Bollinger Band + RSI mean reversion
  3.  MACD Trend          - MACD zero-line crossover with EMA filter
  4.  Donchian Breakout   - Channel breakout (turtle-style) with ATR stop
  5.  RSI Momentum        - RSI overbought/oversold with EMA confirmation
  6.  Triple EMA          - 3 EMAs alignment trend strategy
  7.  Keltner Squeeze     - BB inside KC squeeze breakout (TTM Squeeze)
  8.  Stochastic Reversal - Stochastic K/D crossover in extreme zones
  9.  CCI Breakout        - Commodity Channel Index trend breakout
  10. Mean Reversion SMA  - Price deviation from SMA mean reversion
  11. ADX Trend Strength  - Directional Movement with ADX filter
  12. Double BB           - Double Bollinger Bands walk-the-band
  13. EMA+RSI Combo       - EMA trend + RSI pullback entry
  14. Momentum Breakout   - Rate of Change breakout
  15. Aroon Trend         - Aroon oscillator trend identification

Each strategy gets its own backtest run with identical data and $1M starting capital.
"""

import os
import sys
from decimal import Decimal

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler
from nautilus_trader.test_kit.providers import TestDataProvider
from nautilus_trader.test_kit.providers import TestInstrumentProvider

# Add strategy directory to path
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
from strategies import EMACrossArena
from strategies import EMACrossArenaConfig
from strategies import EMARSIComboArena
from strategies import EMARSIComboArenaConfig
from strategies import KeltnerSqueezeArena
from strategies import KeltnerSqueezeArenaConfig
from strategies import MACDTrendArena
from strategies import MACDTrendArenaConfig
from strategies import MeanRevSMAArena
from strategies import MeanRevSMAArenaConfig
from strategies import MomentumBreakoutArena
from strategies import MomentumBreakoutArenaConfig
from strategies import RSIMomentumArena
from strategies import RSIMomentumArenaConfig
from strategies import StochasticReversalArena
from strategies import StochasticReversalArenaConfig
from strategies import TripleEMAArena
from strategies import TripleEMAArenaConfig


STARTING_BALANCE = 1_000_000
TRADE_SIZE = Decimal(100_000)


def load_audusd_tick_data():
    """Load AUD/USD tick data from bundled test data."""
    SIM = Venue("SIM")
    instrument = TestInstrumentProvider.default_fx_ccy("AUD/USD", SIM)
    provider = TestDataProvider()
    wrangler = QuoteTickDataWrangler(instrument=instrument)
    ticks = wrangler.process(provider.read_csv_ticks("truefx/audusd-ticks.csv"))
    return SIM, instrument, ticks


# Cache data globally so we only load once
_cached_data = None


def get_data():
    global _cached_data
    if _cached_data is None:
        _cached_data = load_audusd_tick_data()
    return _cached_data


def run_strategy(name, strategy_cls, config_cls, config_kwargs):
    """Run a single strategy backtest and return results."""
    SIM, instrument, ticks = get_data()
    bar_type = BarType.from_str("AUD/USD.SIM-1-MINUTE-MID-INTERNAL")

    config = BacktestEngineConfig(
        trader_id=TraderId(f"ARENA-{name[:12]}-001"),
        logging=LoggingConfig(log_level="ERROR"),
    )
    engine = BacktestEngine(config=config)

    engine.add_venue(
        venue=SIM,
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(STARTING_BALANCE, USD)],
    )
    engine.add_instrument(instrument)
    engine.add_data(ticks)

    strat_config = config_cls(
        instrument_id=instrument.id,
        bar_type=bar_type,
        trade_size=TRADE_SIZE,
        **config_kwargs,
    )
    strat = strategy_cls(config=strat_config)
    engine.add_strategy(strat)

    engine.run()
    result = extract_results(engine, name, SIM)
    engine.reset()
    engine.dispose()
    return result


def _parse_pnl(val) -> float:
    """Parse PnL value which may be a float or string like '-20.68 USD'."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # Remove currency suffix like ' USD'
    parts = s.split()
    return float(parts[0])


def extract_results(engine, name, venue):
    """Extract performance metrics from a completed backtest."""
    fills_report = engine.trader.generate_order_fills_report()
    positions_report = engine.trader.generate_positions_report()

    total_orders = len(fills_report) if fills_report is not None and not fills_report.empty else 0

    total_pnl = 0.0
    winning = 0
    losing = 0
    total_pos = 0
    max_win = 0.0
    max_loss = 0.0

    if positions_report is not None and not positions_report.empty and "realized_pnl" in positions_report.columns:
        total_pos = len(positions_report)
        for pnl in positions_report["realized_pnl"]:
            v = _parse_pnl(pnl)
            total_pnl += v
            if v > 0:
                winning += 1
                max_win = max(max_win, v)
            elif v < 0:
                losing += 1
                max_loss = min(max_loss, v)

    win_rate = (winning / total_pos * 100) if total_pos > 0 else 0.0

    # Max drawdown from cumulative PnL
    max_dd = 0.0
    if positions_report is not None and not positions_report.empty and "realized_pnl" in positions_report.columns:
        cum = 0.0
        peak = 0.0
        for pnl in positions_report["realized_pnl"]:
            cum += _parse_pnl(pnl)
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

    # Profit factor
    gross_profit = sum(_parse_pnl(p) for p in positions_report["realized_pnl"] if _parse_pnl(p) > 0) if total_pos > 0 else 0.0
    gross_loss = abs(sum(_parse_pnl(p) for p in positions_report["realized_pnl"] if _parse_pnl(p) < 0)) if total_pos > 0 else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    # Average trade
    avg_trade = total_pnl / total_pos if total_pos > 0 else 0.0

    # Return on capital
    roc = (total_pnl / STARTING_BALANCE) * 100

    return {
        "Strategy": name,
        "Total PnL ($)": round(total_pnl, 2),
        "Return (%)": round(roc, 3),
        "Trades": total_pos,
        "Orders": total_orders,
        "Win Rate (%)": round(win_rate, 1),
        "Avg Trade ($)": round(avg_trade, 2),
        "Max Win ($)": round(max_win, 2),
        "Max Loss ($)": round(max_loss, 2),
        "Max DD ($)": round(max_dd, 2),
        "Profit Factor": round(pf, 2),
    }


# All 15 strategies with their parameters
STRATEGIES = [
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
]


def main():
    print()
    print("=" * 90)
    print("  STRATEGY ARENA - 15 Strategies Head-to-Head Backtest")
    print("  Instrument: AUD/USD | Timeframe: 1-Min bars from ticks | Capital: $1,000,000")
    print("  Trade Size: 100,000 units | Data: ~100K ticks (Jan 30, 2020)")
    print("=" * 90)
    print()

    results = []
    for i, (name, strat_cls, cfg_cls, kwargs) in enumerate(STRATEGIES, 1):
        print(f"  [{i:2d}/15] Running {name:<20s}...", end=" ", flush=True)
        result = run_strategy(name, strat_cls, cfg_cls, kwargs)
        results.append(result)
        pnl = result["Total PnL ($)"]
        wr = result["Win Rate (%)"]
        tr = result["Trades"]
        print(f"PnL: ${pnl:>+10,.2f}  |  Win Rate: {wr:5.1f}%  |  Trades: {tr}")

    # Build and sort results
    df = pd.DataFrame(results)
    df = df.sort_values("Total PnL ($)", ascending=False).reset_index(drop=True)
    df.index = range(1, len(df) + 1)
    df.index.name = "Rank"

    print()
    print("=" * 90)
    print("  FINAL LEADERBOARD")
    print("=" * 90)
    print()

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 20,
        "display.float_format", lambda x: f"{x:,.2f}",
    ):
        print(df.to_string())

    # Top 3
    print()
    print("=" * 90)
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        medal = ["GOLD", "SILVER", "BRONZE"][i]
        print(f"  {medal:>6s}:  {row['Strategy']:<20s}  "
              f"PnL: ${row['Total PnL ($)']:>+12,.2f}  |  "
              f"Win Rate: {row['Win Rate (%)']:5.1f}%  |  "
              f"PF: {row['Profit Factor']:6.2f}  |  "
              f"Max DD: ${row['Max DD ($)']:>10,.2f}")
    print("=" * 90)

    # Winner details
    w = df.iloc[0]
    print()
    print(f"  >>> BEST STRATEGY: {w['Strategy']} <<<")
    print(f"      Total PnL:     ${w['Total PnL ($)']:>+12,.2f}")
    print(f"      Return:         {w['Return (%)']:>+8.3f}%")
    print(f"      Win Rate:       {w['Win Rate (%)']:>8.1f}%")
    print(f"      Profit Factor:  {w['Profit Factor']:>8.2f}")
    print(f"      Total Trades:   {int(w['Trades']):>8d}")
    print(f"      Avg Trade:     ${w['Avg Trade ($)']:>+12,.2f}")
    print(f"      Max Drawdown:  ${w['Max DD ($)']:>12,.2f}")
    print()

    return df


if __name__ == "__main__":
    results_df = main()
