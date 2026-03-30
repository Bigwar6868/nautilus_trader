"""
15 competing trading strategies for head-to-head backtesting.
Inspired by popular TradingView strategies and classic quant approaches.

Strategies:
 1. EMA Cross             - Trend following with fast/slow EMA crossover
 2. BB Mean Reversion     - Bollinger Band + RSI mean reversion
 3. MACD Trend            - MACD zero-line crossover with EMA filter
 4. Donchian Breakout     - Channel breakout (turtle-style) with ATR stop
 5. RSI Momentum          - RSI overbought/oversold with EMA confirmation
 6. Triple EMA            - 3 EMAs (short/mid/long) alignment trend strategy
 7. Keltner Squeeze       - Bollinger inside Keltner channel squeeze breakout
 8. Stochastic Reversal   - Stochastic K/D crossover in extreme zones
 9. CCI Breakout          - Commodity Channel Index trend breakout
10. Mean Reversion SMA    - Price deviation from SMA with mean reversion
11. ADX Trend Strength    - Directional Movement with ADX trend filter
12. Double BB             - Double Bollinger Bands walk-the-band trend strategy
13. EMA+RSI Combo         - EMA trend direction + RSI pullback entry
14. Momentum Breakout     - Rate of Change breakout with volume filter (OBV)
15. Aroon Trend           - Aroon oscillator trend identification
"""

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
from nautilus_trader.indicators import KeltnerChannel
from nautilus_trader.indicators import MovingAverageConvergenceDivergence
from nautilus_trader.indicators import OnBalanceVolume
from nautilus_trader.indicators import RateOfChange
from nautilus_trader.indicators import RelativeStrengthIndex
from nautilus_trader.indicators import SimpleMovingAverage
from nautilus_trader.indicators import Stochastics
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy


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
