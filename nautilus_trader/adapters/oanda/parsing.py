# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Parsing and conversion utilities for the OANDA adapter.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from nautilus_trader.adapters.oanda.constants import OANDA_COMMODITIES
from nautilus_trader.adapters.oanda.constants import OANDA_FX_PAIRS
from nautilus_trader.adapters.oanda.constants import OANDA_INDICES
from nautilus_trader.adapters.oanda.constants import OANDA_METALS
from nautilus_trader.adapters.oanda.constants import OANDA_VENUE
from nautilus_trader.core.rust.model import AssetClass
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.instruments.cfd import Cfd
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


def oanda_symbol_to_instrument_id(oanda_symbol: str) -> InstrumentId:
    """Convert OANDA symbol (e.g. 'EUR_USD') to NautilusTrader InstrumentId."""
    nautilus_symbol = oanda_symbol.replace("_", "/")
    return InstrumentId(symbol=Symbol(nautilus_symbol), venue=OANDA_VENUE)


def instrument_id_to_oanda_symbol(instrument_id: InstrumentId) -> str:
    """Convert NautilusTrader InstrumentId to OANDA symbol (e.g. 'EUR_USD')."""
    return str(instrument_id.symbol).replace("/", "_")


def determine_asset_class(oanda_symbol: str) -> AssetClass:
    """Determine the asset class for an OANDA instrument symbol."""
    parts = oanda_symbol.split("_")
    base = parts[0] if parts else oanda_symbol

    if base in OANDA_METALS:
        return AssetClass.METAL
    if base in OANDA_INDICES:
        return AssetClass.INDEX
    if base in OANDA_COMMODITIES:
        return AssetClass.COMMODITY

    # Check if both parts are FX currencies
    if len(parts) == 2 and parts[0] in OANDA_FX_PAIRS and parts[1] in OANDA_FX_PAIRS:
        return AssetClass.FX

    return AssetClass.CFD


def parse_instrument(inst_data: dict, ts_init: int) -> Cfd:
    """
    Parse an OANDA instrument response into a NautilusTrader Cfd instrument.

    Parameters
    ----------
    inst_data : dict
        The instrument data from the OANDA API.
    ts_init : int
        The initialization timestamp (nanoseconds).

    """
    oanda_symbol = inst_data["name"]
    instrument_id = oanda_symbol_to_instrument_id(oanda_symbol)
    asset_class = determine_asset_class(oanda_symbol)

    display_precision = inst_data.get("displayPrecision", 5)
    pip_location = inst_data.get("pipLocation", -4)

    # Price precision from displayPrecision
    price_precision = display_precision
    price_increment = Price(Decimal(10) ** Decimal(-price_precision), precision=price_precision)

    # Trade units precision
    trade_units_precision = inst_data.get("tradeUnitsPrecision", 0)
    size_increment = Quantity(
        Decimal(10) ** Decimal(-trade_units_precision),
        precision=trade_units_precision,
    )

    # Determine base and quote currencies
    parts = oanda_symbol.split("_")
    quote_currency_str = parts[1] if len(parts) == 2 else "USD"
    base_currency_str = parts[0] if len(parts) == 2 else None

    quote_currency = Currency.from_str(quote_currency_str)
    base_currency = Currency.from_str(base_currency_str) if base_currency_str and base_currency_str in OANDA_FX_PAIRS else None

    # Margin requirements
    margin_rate = inst_data.get("marginRate", "0.05")
    margin_init = Decimal(str(margin_rate))
    margin_maint = Decimal(str(margin_rate))

    # Min/max trade size
    min_trade_size = inst_data.get("minimumTradeSize", "1")
    max_trade_size = inst_data.get("maximumOrderUnits", "100000000")

    return Cfd(
        instrument_id=instrument_id,
        raw_symbol=Symbol(oanda_symbol),
        asset_class=asset_class,
        quote_currency=quote_currency,
        price_precision=price_precision,
        size_precision=trade_units_precision,
        price_increment=price_increment,
        size_increment=size_increment,
        base_currency=base_currency,
        lot_size=Quantity.from_int(1),
        min_quantity=Quantity(Decimal(min_trade_size), precision=trade_units_precision),
        max_quantity=Quantity(Decimal(max_trade_size), precision=trade_units_precision),
        margin_init=margin_init,
        margin_maint=margin_maint,
        maker_fee=Decimal("0"),
        taker_fee=Decimal("0"),
        ts_event=ts_init,
        ts_init=ts_init,
        info=inst_data,
    )


def parse_quote_tick(price_data: dict, instrument_id: InstrumentId, ts_init: int) -> QuoteTick | None:
    """
    Parse an OANDA streaming PRICE object into a NautilusTrader QuoteTick.

    Parameters
    ----------
    price_data : dict
        The PRICE object from the OANDA streaming API.
    instrument_id : InstrumentId
        The instrument ID.
    ts_init : int
        The initialization timestamp (nanoseconds).

    """
    if price_data.get("type") != "PRICE":
        return None

    bids = price_data.get("bids", [])
    asks = price_data.get("asks", [])
    if not bids or not asks:
        return None

    bid_price = bids[0]["price"]
    ask_price = asks[0]["price"]
    bid_size = bids[0].get("liquidity", 1000000)
    ask_size = asks[0].get("liquidity", 1000000)

    # Parse timestamp
    time_str = price_data.get("time", "")
    ts_event = _parse_rfc3339_to_nanos(time_str) if time_str else ts_init

    return QuoteTick(
        instrument_id=instrument_id,
        bid_price=Price.from_str(bid_price),
        ask_price=Price.from_str(ask_price),
        bid_size=Quantity.from_int(int(bid_size)),
        ask_size=Quantity.from_int(int(ask_size)),
        ts_event=ts_event,
        ts_init=ts_init,
    )


def parse_candles_to_bars(
    candles_data: dict,
    bar_type: BarType,
    ts_init: int,
) -> list[Bar]:
    """
    Parse OANDA candles response into a list of NautilusTrader Bar objects.

    Parameters
    ----------
    candles_data : dict
        The candles response from the OANDA API.
    bar_type : BarType
        The bar type for the bars.
    ts_init : int
        The initialization timestamp (nanoseconds).

    """
    bars = []
    for candle in candles_data.get("candles", []):
        if not candle.get("complete", True):
            continue

        mid = candle.get("mid", {})
        if not mid:
            continue

        ts_event = _parse_rfc3339_to_nanos(candle["time"])

        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(mid["o"]),
            high=Price.from_str(mid["h"]),
            low=Price.from_str(mid["l"]),
            close=Price.from_str(mid["c"]),
            volume=Quantity.from_int(int(candle.get("volume", 0))),
            ts_event=ts_event,
            ts_init=ts_init,
        )
        bars.append(bar)

    return bars


def _parse_rfc3339_to_nanos(time_str: str) -> int:
    """Convert RFC3339 timestamp string to nanoseconds since epoch."""
    dt = pd.Timestamp(time_str)
    return int(dt.value)  # pandas Timestamp.value is nanoseconds
