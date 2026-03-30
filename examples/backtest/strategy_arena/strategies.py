"""
25 competing trading strategies for head-to-head backtesting.
Inspired by popular TradingView strategies, classic quant approaches,
and research from quantified strategies / institutional quant desks.

Strategies:
 1.  EMA Cross             - Trend following with fast/slow EMA crossover
 2.  BB Mean Reversion     - Bollinger Band + RSI mean reversion
 3.  MACD Trend            - MACD zero-line crossover with EMA filter
 4.  Donchian Breakout     - Channel breakout (turtle-style) with ATR stop
 5.  RSI Momentum          - RSI overbought/oversold with EMA confirmation
 6.  Triple EMA            - 3 EMAs (short/mid/long) alignment trend strategy
 7.  Keltner Squeeze       - Bollinger inside Keltner channel squeeze breakout
 8.  Stochastic Reversal   - Stochastic K/D crossover in extreme zones
 9.  CCI Breakout          - Commodity Channel Index trend breakout
10.  Mean Reversion SMA    - Price deviation from SMA with mean reversion
11.  ADX Trend Strength    - Directional Movement with ADX trend filter
12.  Double BB             - Double Bollinger Bands walk-the-band trend strategy
13.  EMA+RSI Combo         - EMA trend direction + RSI pullback entry
14.  Momentum Breakout     - Rate of Change breakout
15.  Aroon Trend           - Aroon oscillator trend identification
--- Advanced Quant / TradingView Strategies ---
16.  SuperTrend            - ATR-based trailing stop trend follower (TV #1)
17.  Ichimoku Cloud        - Multi-component Japanese trend system
18.  VWAP Reversion        - Volume-weighted average price mean reversion
19.  Linear Reg Channel    - Linear regression channel breakout
20.  Dual Thrust           - Opening range breakout (famous Chinese quant)
21.  RSI+SuperTrend Combo  - Multi-indicator confirmation (TV popular)
22.  Keltner Trend         - Keltner Channel trend riding
23.  Ichimoku+RSI          - Ichimoku cloud with RSI momentum filter
24.  Pivot Reversal        - Swing high/low pivot point reversal
25.  Hull MA Cross         - Hull Moving Average fast crossover
"""

from collections import deque
from datetime import datetime
from decimal import Decimal

from nautilus_trader.config import PositiveFloat
from nautilus_trader.config import PositiveInt
from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators import AroonOscillator
from nautilus_trader.indicators import AverageTrueRange
from nautilus_trader.indicators import BollingerBands
from nautilus_trader.indicators import CommodityChannelIndex
from nautilus_trader.indicators import DirectionalMovement
from nautilus_trader.indicators import DonchianChannel
from nautilus_trader.indicators import ExponentialMovingAverage
from nautilus_trader.indicators import HullMovingAverage
from nautilus_trader.indicators import KeltnerChannel
from nautilus_trader.indicators import LinearRegression
from nautilus_trader.indicators import MovingAverageConvergenceDivergence
from nautilus_trader.indicators import OnBalanceVolume
from nautilus_trader.indicators import RateOfChange
from nautilus_trader.indicators import RelativeStrengthIndex
from nautilus_trader.indicators import SimpleMovingAverage
from nautilus_trader.indicators import Stochastics
from nautilus_trader.indicators import Swings
from nautilus_trader.indicators import VolumeWeightedAveragePrice
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy


# ===========================================================================
# Simple Ichimoku Cloud calculator (not in installed nautilus_trader 1.221)
# ===========================================================================
class _IchimokuCalc:
    """Minimal Ichimoku Cloud calculation."""

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
        self.tenkan_period = tenkan
        self.kijun_period = kijun
        self.senkou_period = senkou
        self._highs = deque(maxlen=senkou)
        self._lows = deque(maxlen=senkou)
        self.tenkan_sen = 0.0
        self.kijun_sen = 0.0
        self.senkou_span_a = 0.0
        self.senkou_span_b = 0.0
        self.initialized = False

    def update(self, high: float, low: float) -> None:
        self._highs.append(high)
        self._lows.append(low)
        n = len(self._highs)
        if n >= self.senkou_period:
            self.initialized = True
        if n >= self.tenkan_period:
            t_highs = list(self._highs)[-self.tenkan_period:]
            t_lows = list(self._lows)[-self.tenkan_period:]
            self.tenkan_sen = (max(t_highs) + min(t_lows)) / 2.0
        if n >= self.kijun_period:
            k_highs = list(self._highs)[-self.kijun_period:]
            k_lows = list(self._lows)[-self.kijun_period:]
            self.kijun_sen = (max(k_highs) + min(k_lows)) / 2.0
        if n >= self.senkou_period:
            self.senkou_span_a = (self.tenkan_sen + self.kijun_sen) / 2.0
            s_highs = list(self._highs)[-self.senkou_period:]
            s_lows = list(self._lows)[-self.senkou_period:]
            self.senkou_span_b = (max(s_highs) + min(s_lows)) / 2.0

    def reset(self) -> None:
        self._highs.clear()
        self._lows.clear()
        self.tenkan_sen = self.kijun_sen = self.senkou_span_a = self.senkou_span_b = 0.0
        self.initialized = False


# ===========================================================================
# Helper base class for common order submission pattern
# ===========================================================================
class _ArenaBase(Strategy):
    """Base class with shared order submission logic."""

    def _submit_order(self, side: OrderSide) -> None:
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=self.instrument.make_qty(self.config.trade_size),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def _flip_or_enter(self, side: OrderSide) -> None:
        iid = self.config.instrument_id
        if side == OrderSide.BUY:
            if self.portfolio.is_net_short(iid):
                self.close_all_positions(iid)
            if self.portfolio.is_flat(iid):
                self._submit_order(side)
        else:
            if self.portfolio.is_net_long(iid):
                self.close_all_positions(iid)
            if self.portfolio.is_flat(iid):
                self._submit_order(side)

    def _default_on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        if self.config.close_positions_on_stop:
            self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)

    def _load_instrument(self) -> bool:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return False
        return True


# ===========================================================================
# 1. EMA Cross Strategy
# ===========================================================================
class EMACrossArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    fast_ema_period: PositiveInt = 10
    slow_ema_period: PositiveInt = 20
    close_positions_on_stop: bool = True


class EMACrossArena(_ArenaBase):
    """Classic fast/slow EMA crossover trend-following strategy."""

    def __init__(self, config: EMACrossArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.fast_ema)
        self.register_indicator_for_bars(self.config.bar_type, self.slow_ema)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return
        if self.fast_ema.value >= self.slow_ema.value:
            self._flip_or_enter(OrderSide.BUY)
        else:
            self._flip_or_enter(OrderSide.SELL)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.fast_ema.reset()
        self.slow_ema.reset()


# ===========================================================================
# 2. Bollinger Band Mean Reversion Strategy
# ===========================================================================
class BBMeanReversionArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    bb_period: PositiveInt = 20
    bb_std: PositiveFloat = 2.0
    rsi_period: PositiveInt = 14
    rsi_buy_threshold: float = 0.30
    rsi_sell_threshold: float = 0.70
    close_positions_on_stop: bool = True


class BBMeanReversionArena(_ArenaBase):
    """Buy at lower band + oversold RSI, sell at upper band + overbought RSI.
    Exit when price reverts to middle band."""

    def __init__(self, config: BBMeanReversionArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.bb = BollingerBands(config.bb_period, config.bb_std)
        self.rsi = RelativeStrengthIndex(config.rsi_period)

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.bb)
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return
        close = bar.close.as_double()
        iid = self.config.instrument_id

        if self.portfolio.is_net_long(iid) and close >= self.bb.middle:
            self.close_all_positions(iid)
            return
        if self.portfolio.is_net_short(iid) and close <= self.bb.middle:
            self.close_all_positions(iid)
            return

        if close <= self.bb.lower and self.rsi.value < self.config.rsi_buy_threshold:
            self._flip_or_enter(OrderSide.BUY)
        elif close >= self.bb.upper and self.rsi.value > self.config.rsi_sell_threshold:
            self._flip_or_enter(OrderSide.SELL)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.bb.reset()
        self.rsi.reset()


# ===========================================================================
# 3. MACD Trend Strategy
# ===========================================================================
class MACDTrendArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    macd_fast: PositiveInt = 12
    macd_slow: PositiveInt = 26
    ema_filter_period: PositiveInt = 50
    close_positions_on_stop: bool = True


class MACDTrendArena(_ArenaBase):
    """MACD zero-line crossover filtered by a longer-term EMA trend."""

    def __init__(self, config: MACDTrendArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.macd = MovingAverageConvergenceDivergence(config.macd_fast, config.macd_slow)
        self.ema_filter = ExponentialMovingAverage(config.ema_filter_period)
        self._prev_macd: float = 0.0

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.macd)
        self.register_indicator_for_bars(self.config.bar_type, self.ema_filter)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return
        close = bar.close.as_double()
        iid = self.config.instrument_id
        m = self.macd.value

        if m > 0 and self._prev_macd <= 0 and close > self.ema_filter.value:
            self._flip_or_enter(OrderSide.BUY)
        elif m < 0 and self._prev_macd >= 0 and close < self.ema_filter.value:
            self._flip_or_enter(OrderSide.SELL)

        if self.portfolio.is_net_long(iid) and close < self.ema_filter.value:
            self.close_all_positions(iid)
        elif self.portfolio.is_net_short(iid) and close > self.ema_filter.value:
            self.close_all_positions(iid)

        self._prev_macd = m

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.macd.reset()
        self.ema_filter.reset()
        self._prev_macd = 0.0


# ===========================================================================
# 4. Donchian Channel Breakout Strategy
# ===========================================================================
class DonchianBreakoutArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    donchian_period: PositiveInt = 20
    atr_period: PositiveInt = 14
    atr_stop_multiple: PositiveFloat = 2.0
    close_positions_on_stop: bool = True


class DonchianBreakoutArena(_ArenaBase):
    """Turtle-style breakout: enter on channel high/low break, ATR stop."""

    def __init__(self, config: DonchianBreakoutArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.donchian = DonchianChannel(config.donchian_period)
        self.atr = AverageTrueRange(config.atr_period)
        self._entry_price: float = 0.0
        self._prev_upper: float = 0.0
        self._prev_lower: float = 0.0

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.atr)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self.donchian.update_raw(bar.high.as_double(), bar.low.as_double())
        if not self.donchian.initialized or not self.atr.initialized or bar.is_single_price():
            return
        close = bar.close.as_double()
        iid = self.config.instrument_id
        atr_stop = self.atr.value * self.config.atr_stop_multiple

        if self.portfolio.is_net_long(iid):
            if close < self._entry_price - atr_stop or close <= self.donchian.middle:
                self.close_all_positions(iid)
                self._entry_price = 0.0
        elif self.portfolio.is_net_short(iid):
            if close > self._entry_price + atr_stop or close >= self.donchian.middle:
                self.close_all_positions(iid)
                self._entry_price = 0.0

        if self.portfolio.is_flat(iid):
            if self._prev_upper > 0 and close > self._prev_upper:
                self._submit_order(OrderSide.BUY)
                self._entry_price = close
            elif self._prev_lower > 0 and close < self._prev_lower:
                self._submit_order(OrderSide.SELL)
                self._entry_price = close

        self._prev_upper = self.donchian.upper
        self._prev_lower = self.donchian.lower

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.donchian.reset()
        self.atr.reset()
        self._entry_price = 0.0
        self._prev_upper = 0.0
        self._prev_lower = 0.0


# ===========================================================================
# 5. RSI Momentum Strategy
# ===========================================================================
class RSIMomentumArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    rsi_period: PositiveInt = 14
    ema_period: PositiveInt = 20
    rsi_overbought: float = 0.70
    rsi_oversold: float = 0.30
    rsi_exit_upper: float = 0.55
    rsi_exit_lower: float = 0.45
    close_positions_on_stop: bool = True


class RSIMomentumArena(_ArenaBase):
    """RSI exits oversold/overbought with EMA confirmation."""

    def __init__(self, config: RSIMomentumArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.rsi = RelativeStrengthIndex(config.rsi_period)
        self.ema = ExponentialMovingAverage(config.ema_period)
        self._prev_rsi: float = 0.5

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)
        self.register_indicator_for_bars(self.config.bar_type, self.ema)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return
        close = bar.close.as_double()
        iid = self.config.instrument_id
        r = self.rsi.value

        if self.portfolio.is_net_long(iid) and r >= self.config.rsi_exit_upper:
            self.close_all_positions(iid)
        elif self.portfolio.is_net_short(iid) and r <= self.config.rsi_exit_lower:
            self.close_all_positions(iid)

        if self.portfolio.is_flat(iid):
            if self._prev_rsi < self.config.rsi_oversold and r >= self.config.rsi_oversold and close > self.ema.value:
                self._submit_order(OrderSide.BUY)
            elif self._prev_rsi > self.config.rsi_overbought and r <= self.config.rsi_overbought and close < self.ema.value:
                self._submit_order(OrderSide.SELL)
        self._prev_rsi = r

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.rsi.reset()
        self.ema.reset()
        self._prev_rsi = 0.5


# ===========================================================================
# 6. Triple EMA Strategy (TradingView popular)
# ===========================================================================
class TripleEMAArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    short_period: PositiveInt = 8
    mid_period: PositiveInt = 21
    long_period: PositiveInt = 55
    close_positions_on_stop: bool = True


class TripleEMAArena(_ArenaBase):
    """Enter when all 3 EMAs align (short > mid > long = buy).
    Exit when short crosses below mid."""

    def __init__(self, config: TripleEMAArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.ema_short = ExponentialMovingAverage(config.short_period)
        self.ema_mid = ExponentialMovingAverage(config.mid_period)
        self.ema_long = ExponentialMovingAverage(config.long_period)

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.ema_short)
        self.register_indicator_for_bars(self.config.bar_type, self.ema_mid)
        self.register_indicator_for_bars(self.config.bar_type, self.ema_long)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return
        iid = self.config.instrument_id
        s, m, l = self.ema_short.value, self.ema_mid.value, self.ema_long.value

        if s > m > l:
            self._flip_or_enter(OrderSide.BUY)
        elif s < m < l:
            self._flip_or_enter(OrderSide.SELL)
        elif self.portfolio.is_net_long(iid) and s < m:
            self.close_all_positions(iid)
        elif self.portfolio.is_net_short(iid) and s > m:
            self.close_all_positions(iid)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.ema_short.reset()
        self.ema_mid.reset()
        self.ema_long.reset()


# ===========================================================================
# 7. Keltner Squeeze Strategy (TradingView: TTM Squeeze)
# ===========================================================================
class KeltnerSqueezeArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    bb_period: PositiveInt = 20
    bb_std: PositiveFloat = 2.0
    kc_period: PositiveInt = 20
    kc_multiplier: PositiveFloat = 1.5
    close_positions_on_stop: bool = True


class KeltnerSqueezeArena(_ArenaBase):
    """Squeeze detection: when BB is inside KC, volatility is compressed.
    When squeeze fires (BB expands outside KC), enter in direction of momentum."""

    def __init__(self, config: KeltnerSqueezeArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.bb = BollingerBands(config.bb_period, config.bb_std)
        self.kc = KeltnerChannel(config.kc_period, config.kc_multiplier)
        self._was_squeezed: bool = False
        self._momentum_bars: list[float] = []

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.bb)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        h, l, c = bar.high.as_double(), bar.low.as_double(), bar.close.as_double()
        self.kc.update_raw(h, l, c)
        if not self.bb.initialized or not self.kc.initialized or bar.is_single_price():
            return

        is_squeezed = self.bb.lower > self.kc.lower and self.bb.upper < self.kc.upper
        squeeze_fired = self._was_squeezed and not is_squeezed

        # Track momentum as close - midline
        self._momentum_bars.append(c - self.bb.middle)
        if len(self._momentum_bars) > 5:
            self._momentum_bars.pop(0)

        if squeeze_fired and len(self._momentum_bars) >= 2:
            momentum = self._momentum_bars[-1]
            if momentum > 0:
                self._flip_or_enter(OrderSide.BUY)
            elif momentum < 0:
                self._flip_or_enter(OrderSide.SELL)

        # Exit when squeeze re-engages
        iid = self.config.instrument_id
        if is_squeezed and not self.portfolio.is_flat(iid):
            self.close_all_positions(iid)

        self._was_squeezed = is_squeezed

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.bb.reset()
        self.kc.reset()
        self._was_squeezed = False
        self._momentum_bars.clear()


# ===========================================================================
# 8. Stochastic Reversal Strategy
# ===========================================================================
class StochasticReversalArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    stoch_period: PositiveInt = 14
    overbought: float = 80.0
    oversold: float = 20.0
    close_positions_on_stop: bool = True


class StochasticReversalArena(_ArenaBase):
    """Stochastic K/D crossover in extreme zones.
    Buy when K crosses above D in oversold, sell when K crosses below D in overbought."""

    def __init__(self, config: StochasticReversalArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.stoch = Stochastics(config.stoch_period, config.stoch_period)
        self._prev_k: float = 50.0
        self._prev_d: float = 50.0

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        h, l, c = bar.high.as_double(), bar.low.as_double(), bar.close.as_double()
        self.stoch.update_raw(h, l, c)
        if not self.stoch.initialized or bar.is_single_price():
            return

        k, d = self.stoch.value_k, self.stoch.value_d
        iid = self.config.instrument_id

        # Buy signal: K crosses above D in oversold
        if self._prev_k <= self._prev_d and k > d and k < self.config.oversold:
            self._flip_or_enter(OrderSide.BUY)
        # Sell signal: K crosses below D in overbought
        elif self._prev_k >= self._prev_d and k < d and k > self.config.overbought:
            self._flip_or_enter(OrderSide.SELL)

        # Exit at opposite extreme
        if self.portfolio.is_net_long(iid) and k > self.config.overbought:
            self.close_all_positions(iid)
        elif self.portfolio.is_net_short(iid) and k < self.config.oversold:
            self.close_all_positions(iid)

        self._prev_k = k
        self._prev_d = d

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.stoch.reset()
        self._prev_k = 50.0
        self._prev_d = 50.0


# ===========================================================================
# 9. CCI Breakout Strategy
# ===========================================================================
class CCIBreakoutArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    cci_period: PositiveInt = 20
    upper_threshold: float = 100.0
    lower_threshold: float = -100.0
    close_positions_on_stop: bool = True


class CCIBreakoutArena(_ArenaBase):
    """CCI above +100 = strong uptrend entry, below -100 = strong downtrend.
    Exit when CCI reverts toward zero."""

    def __init__(self, config: CCIBreakoutArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.cci = CommodityChannelIndex(config.cci_period)
        self._prev_cci: float = 0.0

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        h, l, c = bar.high.as_double(), bar.low.as_double(), bar.close.as_double()
        self.cci.update_raw(h, l, c)
        if not self.cci.initialized or bar.is_single_price():
            return

        v = self.cci.value
        iid = self.config.instrument_id

        # Entry: CCI crosses threshold
        if self._prev_cci <= self.config.upper_threshold and v > self.config.upper_threshold:
            self._flip_or_enter(OrderSide.BUY)
        elif self._prev_cci >= self.config.lower_threshold and v < self.config.lower_threshold:
            self._flip_or_enter(OrderSide.SELL)

        # Exit: CCI reverts toward zero
        if self.portfolio.is_net_long(iid) and v < 0:
            self.close_all_positions(iid)
        elif self.portfolio.is_net_short(iid) and v > 0:
            self.close_all_positions(iid)

        self._prev_cci = v

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.cci.reset()
        self._prev_cci = 0.0


# ===========================================================================
# 10. Mean Reversion SMA Strategy
# ===========================================================================
class MeanRevSMAArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    sma_period: PositiveInt = 50
    entry_deviation: PositiveFloat = 0.015  # 1.5% deviation from SMA
    exit_deviation: PositiveFloat = 0.002   # 0.2% from SMA = near mean
    close_positions_on_stop: bool = True


class MeanRevSMAArena(_ArenaBase):
    """Buy when price deviates >1.5% below SMA, sell when >1.5% above.
    Exit when price reverts near the SMA."""

    def __init__(self, config: MeanRevSMAArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.sma = SimpleMovingAverage(config.sma_period)

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.sma)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return
        close = bar.close.as_double()
        iid = self.config.instrument_id
        sma_val = self.sma.value
        if sma_val == 0:
            return
        deviation = (close - sma_val) / sma_val

        # Exit near mean
        if not self.portfolio.is_flat(iid) and abs(deviation) < self.config.exit_deviation:
            self.close_all_positions(iid)
            return

        # Entry on extreme deviation
        if deviation < -self.config.entry_deviation:
            self._flip_or_enter(OrderSide.BUY)
        elif deviation > self.config.entry_deviation:
            self._flip_or_enter(OrderSide.SELL)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.sma.reset()


# ===========================================================================
# 11. ADX Trend Strength Strategy (TradingView: ADX + DI)
# ===========================================================================
class ADXTrendArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    adx_period: PositiveInt = 14
    adx_threshold: PositiveFloat = 25.0  # Only trade when ADX > 25
    close_positions_on_stop: bool = True


class ADXTrendArena(_ArenaBase):
    """Trade in direction of DI+ vs DI- when ADX indicates strong trend (>25).
    Exit when ADX drops below threshold."""

    def __init__(self, config: ADXTrendArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.dm = DirectionalMovement(config.adx_period)

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        h, l = bar.high.as_double(), bar.low.as_double()
        self.dm.update_raw(h, l)
        if not self.dm.initialized or bar.is_single_price():
            return

        iid = self.config.instrument_id
        adx = self.dm.value  # ADX is the main value
        di_plus = self.dm.pos
        di_minus = self.dm.neg

        if adx > self.config.adx_threshold:
            if di_plus > di_minus:
                self._flip_or_enter(OrderSide.BUY)
            elif di_minus > di_plus:
                self._flip_or_enter(OrderSide.SELL)
        elif not self.portfolio.is_flat(iid):
            # ADX weak -> exit
            self.close_all_positions(iid)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.dm.reset()


# ===========================================================================
# 12. Double Bollinger Bands Strategy (TradingView: Kathy Lien)
# ===========================================================================
class DoubleBBArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    bb_period: PositiveInt = 20
    bb_inner_std: PositiveFloat = 1.0
    bb_outer_std: PositiveFloat = 2.0
    close_positions_on_stop: bool = True


class DoubleBBArena(_ArenaBase):
    """Double Bollinger Bands: price between outer and inner upper = uptrend.
    Price between outer and inner lower = downtrend. Inside inner = neutral."""

    def __init__(self, config: DoubleBBArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.bb_inner = BollingerBands(config.bb_period, config.bb_inner_std)
        self.bb_outer = BollingerBands(config.bb_period, config.bb_outer_std)

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.bb_inner)
        self.register_indicator_for_bars(self.config.bar_type, self.bb_outer)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return
        close = bar.close.as_double()
        iid = self.config.instrument_id

        # Uptrend zone: between inner upper and outer upper
        if close > self.bb_inner.upper:
            self._flip_or_enter(OrderSide.BUY)
        # Downtrend zone: between inner lower and outer lower
        elif close < self.bb_inner.lower:
            self._flip_or_enter(OrderSide.SELL)
        # Neutral zone: inside inner bands -> flatten
        elif not self.portfolio.is_flat(iid):
            self.close_all_positions(iid)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.bb_inner.reset()
        self.bb_outer.reset()


# ===========================================================================
# 13. EMA + RSI Combo Strategy (TradingView popular)
# ===========================================================================
class EMARSIComboArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    ema_period: PositiveInt = 50
    rsi_period: PositiveInt = 14
    rsi_buy_zone: float = 0.40
    rsi_sell_zone: float = 0.60
    close_positions_on_stop: bool = True


class EMARSIComboArena(_ArenaBase):
    """EMA defines trend direction. RSI pullback provides entry timing.
    Buy when price above EMA and RSI pulls back below 40 then recovers.
    Sell when price below EMA and RSI pulls up above 60 then drops."""

    def __init__(self, config: EMARSIComboArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.ema = ExponentialMovingAverage(config.ema_period)
        self.rsi = RelativeStrengthIndex(config.rsi_period)
        self._prev_rsi: float = 0.5

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.ema)
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return
        close = bar.close.as_double()
        iid = self.config.instrument_id
        r = self.rsi.value

        # Uptrend: price above EMA, RSI pulled back and recovering
        if close > self.ema.value and self._prev_rsi < self.config.rsi_buy_zone and r >= self.config.rsi_buy_zone:
            self._flip_or_enter(OrderSide.BUY)
        # Downtrend: price below EMA, RSI bounced up and dropping
        elif close < self.ema.value and self._prev_rsi > self.config.rsi_sell_zone and r <= self.config.rsi_sell_zone:
            self._flip_or_enter(OrderSide.SELL)

        # Exit if EMA flips against position
        if self.portfolio.is_net_long(iid) and close < self.ema.value:
            self.close_all_positions(iid)
        elif self.portfolio.is_net_short(iid) and close > self.ema.value:
            self.close_all_positions(iid)

        self._prev_rsi = r

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.ema.reset()
        self.rsi.reset()
        self._prev_rsi = 0.5


# ===========================================================================
# 14. Momentum Breakout (Rate of Change + OBV)
# ===========================================================================
class MomentumBreakoutArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    roc_period: PositiveInt = 12
    roc_threshold: PositiveFloat = 0.5  # % threshold
    close_positions_on_stop: bool = True


class MomentumBreakoutArena(_ArenaBase):
    """Rate of Change breakout: enter when ROC exceeds threshold.
    Exit when ROC crosses zero in opposite direction."""

    def __init__(self, config: MomentumBreakoutArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.roc = RateOfChange(config.roc_period)
        self._prev_roc: float = 0.0

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.roc)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return
        v = self.roc.value
        iid = self.config.instrument_id

        # Breakout entry
        if v > self.config.roc_threshold:
            self._flip_or_enter(OrderSide.BUY)
        elif v < -self.config.roc_threshold:
            self._flip_or_enter(OrderSide.SELL)

        # Exit when momentum fades
        if self.portfolio.is_net_long(iid) and v < 0:
            self.close_all_positions(iid)
        elif self.portfolio.is_net_short(iid) and v > 0:
            self.close_all_positions(iid)

        self._prev_roc = v

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.roc.reset()
        self._prev_roc = 0.0


# ===========================================================================
# 15. Aroon Trend Strategy
# ===========================================================================
class AroonTrendArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    aroon_period: PositiveInt = 25
    threshold: PositiveFloat = 70.0
    close_positions_on_stop: bool = True


class AroonTrendArena(_ArenaBase):
    """Aroon Up > 70 and Aroon Down < 30 = uptrend -> buy.
    Aroon Down > 70 and Aroon Up < 30 = downtrend -> sell.
    Exit when both oscillate near middle."""

    def __init__(self, config: AroonTrendArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.aroon = AroonOscillator(config.aroon_period)

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        h, l = bar.high.as_double(), bar.low.as_double()
        self.aroon.update_raw(h, l)
        if not self.aroon.initialized or bar.is_single_price():
            return

        iid = self.config.instrument_id
        up = self.aroon.aroon_up
        down = self.aroon.aroon_down
        threshold = self.config.threshold
        weak = 100 - threshold  # 30

        if up > threshold and down < weak:
            self._flip_or_enter(OrderSide.BUY)
        elif down > threshold and up < weak:
            self._flip_or_enter(OrderSide.SELL)
        elif not self.portfolio.is_flat(iid) and abs(up - down) < 20:
            self.close_all_positions(iid)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.aroon.reset()


# ===========================================================================
# 16. SuperTrend Strategy (TradingView #1 indicator)
# ===========================================================================
class SuperTrendArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    atr_period: PositiveInt = 10
    atr_multiplier: PositiveFloat = 3.0
    close_positions_on_stop: bool = True


class SuperTrendArena(_ArenaBase):
    """ATR-based SuperTrend: calculates trailing stop bands using ATR.
    Flips from long to short when price crosses below lower band, and vice versa.
    This is the most popular TradingView indicator."""

    def __init__(self, config: SuperTrendArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.atr = AverageTrueRange(config.atr_period)
        self._upper_band: float = 0.0
        self._lower_band: float = 0.0
        self._prev_upper: float = 0.0
        self._prev_lower: float = 0.0
        self._trend: int = 1  # 1 = up, -1 = down
        self._prev_close: float = 0.0

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.atr)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.atr.initialized or bar.is_single_price():
            self._prev_close = bar.close.as_double()
            return

        h, l, c = bar.high.as_double(), bar.low.as_double(), bar.close.as_double()
        hl2 = (h + l) / 2.0
        atr_val = self.atr.value * self.config.atr_multiplier

        # Calculate basic bands
        basic_upper = hl2 + atr_val
        basic_lower = hl2 - atr_val

        # Final bands (carry forward if price hasn't crossed)
        if basic_lower > self._prev_lower or self._prev_close < self._prev_lower:
            self._lower_band = basic_lower
        else:
            self._lower_band = self._prev_lower

        if basic_upper < self._prev_upper or self._prev_close > self._prev_upper:
            self._upper_band = basic_upper
        else:
            self._upper_band = self._prev_upper

        # Determine trend direction
        prev_trend = self._trend
        if prev_trend == -1 and c > self._prev_upper:
            self._trend = 1
        elif prev_trend == 1 and c < self._prev_lower:
            self._trend = -1

        # Trade on trend flip
        if self._trend == 1 and prev_trend == -1:
            self._flip_or_enter(OrderSide.BUY)
        elif self._trend == -1 and prev_trend == 1:
            self._flip_or_enter(OrderSide.SELL)

        self._prev_upper = self._upper_band
        self._prev_lower = self._lower_band
        self._prev_close = c

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.atr.reset()
        self._upper_band = self._lower_band = 0.0
        self._prev_upper = self._prev_lower = 0.0
        self._trend = 1
        self._prev_close = 0.0


# ===========================================================================
# 17. Ichimoku Cloud Strategy
# ===========================================================================
class IchimokuArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    tenkan_period: PositiveInt = 9
    kijun_period: PositiveInt = 26
    senkou_period: PositiveInt = 52
    close_positions_on_stop: bool = True


class IchimokuArena(_ArenaBase):
    """Ichimoku Cloud strategy:
    Buy when price above cloud AND tenkan > kijun (TK cross).
    Sell when price below cloud AND tenkan < kijun.
    Exit when price enters the cloud."""

    def __init__(self, config: IchimokuArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.ichimoku = _IchimokuCalc(config.tenkan_period, config.kijun_period, config.senkou_period)

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        h, l, c = bar.high.as_double(), bar.low.as_double(), bar.close.as_double()
        self.ichimoku.update(h, l)
        if not self.ichimoku.initialized or bar.is_single_price():
            return

        iid = self.config.instrument_id
        tenkan = self.ichimoku.tenkan_sen
        kijun = self.ichimoku.kijun_sen
        span_a = self.ichimoku.senkou_span_a
        span_b = self.ichimoku.senkou_span_b

        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)

        # Price above cloud + TK cross bullish
        if c > cloud_top and tenkan > kijun:
            self._flip_or_enter(OrderSide.BUY)
        # Price below cloud + TK cross bearish
        elif c < cloud_bottom and tenkan < kijun:
            self._flip_or_enter(OrderSide.SELL)
        # Price inside cloud -> exit
        elif not self.portfolio.is_flat(iid) and cloud_bottom <= c <= cloud_top:
            self.close_all_positions(iid)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.ichimoku.reset()


# ===========================================================================
# 18. VWAP Reversion Strategy (Institutional favorite)
# ===========================================================================
class VWAPReversionArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    atr_period: PositiveInt = 14
    entry_atr_multiple: PositiveFloat = 1.5
    exit_atr_multiple: PositiveFloat = 0.3
    close_positions_on_stop: bool = True


class VWAPReversionArena(_ArenaBase):
    """VWAP mean reversion: enter when price deviates >1.5 ATR from VWAP,
    exit when price reverts within 0.3 ATR of VWAP. Institutional strategy."""

    def __init__(self, config: VWAPReversionArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.vwap = VolumeWeightedAveragePrice()
        self.atr = AverageTrueRange(config.atr_period)

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.atr)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        c = bar.close.as_double()
        vol = bar.volume.as_double()
        ts = datetime.utcfromtimestamp(bar.ts_event / 1e9)
        self.vwap.update_raw(c, vol, ts)
        if not self.vwap.initialized or not self.atr.initialized or bar.is_single_price():
            return
        if self.atr.value == 0:
            return

        iid = self.config.instrument_id
        deviation = (c - self.vwap.value) / self.atr.value

        # Exit near VWAP
        if not self.portfolio.is_flat(iid) and abs(deviation) < self.config.exit_atr_multiple:
            self.close_all_positions(iid)
            return

        # Entry on extreme deviation
        if deviation < -self.config.entry_atr_multiple:
            self._flip_or_enter(OrderSide.BUY)
        elif deviation > self.config.entry_atr_multiple:
            self._flip_or_enter(OrderSide.SELL)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.vwap.reset()
        self.atr.reset()


# ===========================================================================
# 19. Linear Regression Channel Strategy
# ===========================================================================
class LinRegChannelArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    period: PositiveInt = 50
    entry_std: PositiveFloat = 2.0
    close_positions_on_stop: bool = True


class LinRegChannelArena(_ArenaBase):
    """Linear regression channel: enter when price breaks beyond 2 std devs
    from regression line. Exit when price reverts to regression value.
    Uses slope for trend confirmation."""

    def __init__(self, config: LinRegChannelArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.linreg = LinearRegression(config.period)
        self._prices: list[float] = []

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.linreg)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.linreg.initialized or bar.is_single_price():
            return

        c = bar.close.as_double()
        self._prices.append(c)
        if len(self._prices) > self.config.period:
            self._prices.pop(0)

        if len(self._prices) < self.config.period:
            return

        reg_val = self.linreg.value
        slope = self.linreg.slope

        # Calculate std dev of residuals
        mean_price = sum(self._prices) / len(self._prices)
        variance = sum((p - mean_price) ** 2 for p in self._prices) / len(self._prices)
        std = variance ** 0.5
        if std == 0:
            return

        deviation = (c - reg_val) / std
        iid = self.config.instrument_id

        # Exit when price reverts near regression
        if not self.portfolio.is_flat(iid) and abs(deviation) < 0.5:
            self.close_all_positions(iid)
            return

        # Trend-confirmed channel breakout
        if deviation < -self.config.entry_std and slope > 0:
            self._flip_or_enter(OrderSide.BUY)  # Oversold in uptrend
        elif deviation > self.config.entry_std and slope < 0:
            self._flip_or_enter(OrderSide.SELL)  # Overbought in downtrend

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.linreg.reset()
        self._prices.clear()


# ===========================================================================
# 20. Dual Thrust Strategy (Famous Chinese quant strategy)
# ===========================================================================
class DualThrustArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    lookback: PositiveInt = 4
    k_up: PositiveFloat = 0.5
    k_down: PositiveFloat = 0.5
    close_positions_on_stop: bool = True


class DualThrustArena(_ArenaBase):
    """Dual Thrust: calculates a range from N-period high/low/close,
    then sets upper/lower trigger from the open.
    Buy when price > open + k*range, sell when price < open - k*range.
    Popular strategy in Chinese futures markets."""

    def __init__(self, config: DualThrustArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []
        self._bar_count: int = 0
        self._session_open: float = 0.0

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if bar.is_single_price():
            return

        h, l, c, o = bar.high.as_double(), bar.low.as_double(), bar.close.as_double(), bar.open.as_double()
        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._bar_count += 1

        n = self.config.lookback
        if len(self._highs) > n:
            self._highs.pop(0)
            self._lows.pop(0)
            self._closes.pop(0)

        if len(self._highs) < n:
            return

        # Use open of current bar as session reference
        self._session_open = o

        # Calculate range
        hh = max(self._highs)
        hc = max(self._closes)
        ll = min(self._lows)
        lc = min(self._closes)
        range_val = max(hh - lc, hc - ll)

        upper_trigger = self._session_open + self.config.k_up * range_val
        lower_trigger = self._session_open - self.config.k_down * range_val

        iid = self.config.instrument_id

        if c > upper_trigger:
            self._flip_or_enter(OrderSide.BUY)
        elif c < lower_trigger:
            self._flip_or_enter(OrderSide.SELL)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._bar_count = 0
        self._session_open = 0.0


# ===========================================================================
# 21. SuperTrend + RSI Combo (TradingView popular multi-indicator)
# ===========================================================================
class SuperTrendRSIArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    atr_period: PositiveInt = 10
    atr_multiplier: PositiveFloat = 3.0
    rsi_period: PositiveInt = 14
    rsi_overbought: float = 0.70
    rsi_oversold: float = 0.30
    close_positions_on_stop: bool = True


class SuperTrendRSIArena(_ArenaBase):
    """SuperTrend for direction + RSI for confirmation.
    Only buy on ST flip up when RSI was recently oversold.
    Only sell on ST flip down when RSI was recently overbought.
    Higher quality signals from multi-indicator confirmation."""

    def __init__(self, config: SuperTrendRSIArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.atr = AverageTrueRange(config.atr_period)
        self.rsi = RelativeStrengthIndex(config.rsi_period)
        self._upper_band: float = 0.0
        self._lower_band: float = 0.0
        self._prev_upper: float = 0.0
        self._prev_lower: float = 0.0
        self._trend: int = 1
        self._prev_close: float = 0.0

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.atr)
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.atr.initialized or not self.rsi.initialized or bar.is_single_price():
            self._prev_close = bar.close.as_double()
            return

        h, l, c = bar.high.as_double(), bar.low.as_double(), bar.close.as_double()
        hl2 = (h + l) / 2.0
        atr_val = self.atr.value * self.config.atr_multiplier

        basic_upper = hl2 + atr_val
        basic_lower = hl2 - atr_val

        if basic_lower > self._prev_lower or self._prev_close < self._prev_lower:
            self._lower_band = basic_lower
        else:
            self._lower_band = self._prev_lower
        if basic_upper < self._prev_upper or self._prev_close > self._prev_upper:
            self._upper_band = basic_upper
        else:
            self._upper_band = self._prev_upper

        prev_trend = self._trend
        if prev_trend == -1 and c > self._prev_upper:
            self._trend = 1
        elif prev_trend == 1 and c < self._prev_lower:
            self._trend = -1

        rsi_val = self.rsi.value

        # Buy: ST flips up + RSI not overbought (was recently oversold)
        if self._trend == 1 and prev_trend == -1 and rsi_val < self.config.rsi_overbought:
            self._flip_or_enter(OrderSide.BUY)
        # Sell: ST flips down + RSI not oversold (was recently overbought)
        elif self._trend == -1 and prev_trend == 1 and rsi_val > self.config.rsi_oversold:
            self._flip_or_enter(OrderSide.SELL)

        self._prev_upper = self._upper_band
        self._prev_lower = self._lower_band
        self._prev_close = c

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.atr.reset()
        self.rsi.reset()
        self._upper_band = self._lower_band = 0.0
        self._prev_upper = self._prev_lower = 0.0
        self._trend = 1
        self._prev_close = 0.0


# ===========================================================================
# 22. Keltner Trend Riding Strategy
# ===========================================================================
class KeltnerTrendArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    kc_period: PositiveInt = 20
    kc_multiplier: PositiveFloat = 2.0
    close_positions_on_stop: bool = True


class KeltnerTrendArena(_ArenaBase):
    """Keltner Channel trend riding: price above upper KC = strong uptrend,
    price below lower KC = strong downtrend. Exit when price returns to middle."""

    def __init__(self, config: KeltnerTrendArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.kc = KeltnerChannel(config.kc_period, config.kc_multiplier)

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        h, l, c = bar.high.as_double(), bar.low.as_double(), bar.close.as_double()
        self.kc.update_raw(h, l, c)
        if not self.kc.initialized or bar.is_single_price():
            return

        iid = self.config.instrument_id

        if c > self.kc.upper:
            self._flip_or_enter(OrderSide.BUY)
        elif c < self.kc.lower:
            self._flip_or_enter(OrderSide.SELL)
        elif not self.portfolio.is_flat(iid):
            # Price returned to middle -> exit
            if self.portfolio.is_net_long(iid) and c < self.kc.middle:
                self.close_all_positions(iid)
            elif self.portfolio.is_net_short(iid) and c > self.kc.middle:
                self.close_all_positions(iid)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.kc.reset()


# ===========================================================================
# 23. Ichimoku + RSI Combo Strategy
# ===========================================================================
class IchimokuRSIArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    tenkan_period: PositiveInt = 9
    kijun_period: PositiveInt = 26
    senkou_period: PositiveInt = 52
    rsi_period: PositiveInt = 14
    rsi_overbought: float = 0.65
    rsi_oversold: float = 0.35
    close_positions_on_stop: bool = True


class IchimokuRSIArena(_ArenaBase):
    """Ichimoku cloud for trend + RSI for momentum confirmation.
    Buy when above cloud, TK bullish, RSI > 50.
    Sell when below cloud, TK bearish, RSI < 50.
    Exit inside cloud or RSI extreme."""

    def __init__(self, config: IchimokuRSIArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.ichimoku = _IchimokuCalc(config.tenkan_period, config.kijun_period, config.senkou_period)
        self.rsi = RelativeStrengthIndex(config.rsi_period)

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.rsi)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        h, l, c = bar.high.as_double(), bar.low.as_double(), bar.close.as_double()
        self.ichimoku.update(h, l)
        if not self.ichimoku.initialized or not self.rsi.initialized or bar.is_single_price():
            return

        iid = self.config.instrument_id
        tenkan = self.ichimoku.tenkan_sen
        kijun = self.ichimoku.kijun_sen
        cloud_top = max(self.ichimoku.senkou_span_a, self.ichimoku.senkou_span_b)
        cloud_bottom = min(self.ichimoku.senkou_span_a, self.ichimoku.senkou_span_b)
        rsi_val = self.rsi.value

        # Exit on RSI extreme or inside cloud
        if self.portfolio.is_net_long(iid):
            if rsi_val > self.config.rsi_overbought or c < cloud_bottom:
                self.close_all_positions(iid)
                return
        elif self.portfolio.is_net_short(iid):
            if rsi_val < self.config.rsi_oversold or c > cloud_top:
                self.close_all_positions(iid)
                return

        # Entry with cloud + TK + RSI confirmation
        if c > cloud_top and tenkan > kijun and rsi_val > 0.50 and rsi_val < self.config.rsi_overbought:
            self._flip_or_enter(OrderSide.BUY)
        elif c < cloud_bottom and tenkan < kijun and rsi_val < 0.50 and rsi_val > self.config.rsi_oversold:
            self._flip_or_enter(OrderSide.SELL)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.ichimoku.reset()
        self.rsi.reset()


# ===========================================================================
# 24. Pivot Reversal Strategy (Swing High/Low)
# ===========================================================================
class PivotReversalArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    swing_period: PositiveInt = 5
    ema_period: PositiveInt = 20
    close_positions_on_stop: bool = True


class PivotReversalArena(_ArenaBase):
    """Trade swing high/low pivot reversals with EMA trend filter.
    Buy at swing low when above EMA (pullback in uptrend).
    Sell at swing high when below EMA (bounce in downtrend)."""

    def __init__(self, config: PivotReversalArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.swings = Swings(config.swing_period)
        self.ema = ExponentialMovingAverage(config.ema_period)
        self._prev_direction: int = 0

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.ema)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        h, l = bar.high.as_double(), bar.low.as_double()
        c = bar.close.as_double()
        ts = datetime.utcfromtimestamp(bar.ts_event / 1e9)
        self.swings.update_raw(h, l, ts)
        if not self.swings.initialized or not self.ema.initialized or bar.is_single_price():
            return

        iid = self.config.instrument_id
        direction = self.swings.direction

        # Swing changed direction
        if direction != self._prev_direction and self._prev_direction != 0:
            # Swing turned up (new swing low formed) + above EMA
            if direction == 1 and c > self.ema.value:
                self._flip_or_enter(OrderSide.BUY)
            # Swing turned down (new swing high formed) + below EMA
            elif direction == -1 and c < self.ema.value:
                self._flip_or_enter(OrderSide.SELL)

        # Exit if EMA flips
        if self.portfolio.is_net_long(iid) and c < self.ema.value:
            self.close_all_positions(iid)
        elif self.portfolio.is_net_short(iid) and c > self.ema.value:
            self.close_all_positions(iid)

        self._prev_direction = direction

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.swings.reset()
        self.ema.reset()
        self._prev_direction = 0


# ===========================================================================
# 25. Hull Moving Average Cross Strategy
# ===========================================================================
class HullMACrossArenaConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    fast_period: PositiveInt = 9
    slow_period: PositiveInt = 21
    close_positions_on_stop: bool = True


class HullMACrossArena(_ArenaBase):
    """Hull MA is faster and smoother than EMA, reducing lag.
    Fast Hull crosses above Slow Hull = buy, below = sell.
    Hull MA eliminates most lag from traditional MAs."""

    def __init__(self, config: HullMACrossArenaConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument = None
        self.fast_hma = HullMovingAverage(config.fast_period)
        self.slow_hma = HullMovingAverage(config.slow_period)

    def on_start(self) -> None:
        if not self._load_instrument():
            return
        self.register_indicator_for_bars(self.config.bar_type, self.fast_hma)
        self.register_indicator_for_bars(self.config.bar_type, self.slow_hma)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        if not self.indicators_initialized() or bar.is_single_price():
            return

        if self.fast_hma.value >= self.slow_hma.value:
            self._flip_or_enter(OrderSide.BUY)
        else:
            self._flip_or_enter(OrderSide.SELL)

    def on_stop(self) -> None:
        self._default_on_stop()

    def on_reset(self) -> None:
        self.fast_hma.reset()
        self.slow_hma.reset()
