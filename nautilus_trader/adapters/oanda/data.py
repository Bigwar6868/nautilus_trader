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
OANDA live market data client.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from nautilus_trader.adapters.oanda.constants import OANDA_GRANULARITIES
from nautilus_trader.adapters.oanda.constants import OANDA_MAX_CANDLES
from nautilus_trader.adapters.oanda.constants import OANDA_VENUE
from nautilus_trader.adapters.oanda.http_client import OandaHttpClient
from nautilus_trader.adapters.oanda.parsing import instrument_id_to_oanda_symbol
from nautilus_trader.adapters.oanda.parsing import oanda_symbol_to_instrument_id
from nautilus_trader.adapters.oanda.parsing import parse_candles_to_bars
from nautilus_trader.adapters.oanda.parsing import parse_quote_tick
from nautilus_trader.adapters.oanda.providers import OandaInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.enums import LogColor
from nautilus_trader.live.data_client import LiveMarketDataClient
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId

if TYPE_CHECKING:
    from nautilus_trader.data.messages import RequestBars
    from nautilus_trader.data.messages import RequestInstrument
    from nautilus_trader.data.messages import RequestInstruments
    from nautilus_trader.data.messages import RequestQuoteTicks
    from nautilus_trader.data.messages import SubscribeBars
    from nautilus_trader.data.messages import SubscribeInstrument
    from nautilus_trader.data.messages import SubscribeInstruments
    from nautilus_trader.data.messages import SubscribeQuoteTicks
    from nautilus_trader.data.messages import UnsubscribeBars
    from nautilus_trader.data.messages import UnsubscribeQuoteTicks


class OandaDataClient(LiveMarketDataClient):
    """
    Provides a live market data client for the OANDA venue.

    Streams real-time quote ticks via OANDA's HTTP chunked streaming API
    and fetches historical bar data via the REST candles endpoint.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        The event loop for the client.
    client : OandaHttpClient
        The OANDA HTTP client.
    msgbus : MessageBus
        The message bus for the client.
    cache : Cache
        The cache for the client.
    clock : LiveClock
        The clock for the client.
    instrument_provider : OandaInstrumentProvider
        The OANDA instrument provider.
    name : str, optional
        The custom client ID.
    config : NautilusConfig, optional
        The configuration for the client.

    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        client: OandaHttpClient,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
        instrument_provider: OandaInstrumentProvider,
        name: str | None = None,
        config: NautilusConfig | None = None,
    ) -> None:
        super().__init__(
            loop=loop,
            client_id=ClientId(name or OANDA_VENUE.value),
            venue=OANDA_VENUE,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=instrument_provider,
            config=config,
        )
        self._client = client
        self._instrument_provider = instrument_provider

        # Streaming state
        self._subscribed_quote_instruments: set[str] = set()  # OANDA symbols
        self._stream_task: asyncio.Task | None = None
        self._stream_running = False

    async def _connect(self) -> None:
        """Connect to OANDA and load instruments."""
        await self._client.connect()
        await self._instrument_provider.initialize()
        self._log.info("OANDA data client connected", color=LogColor.GREEN)

    async def _disconnect(self) -> None:
        """Disconnect from OANDA."""
        self._stream_running = False
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        await self._client.disconnect()
        self._log.info("OANDA data client disconnected")

    # -- Subscriptions -----------------------------------------------------------

    async def _subscribe_quote_ticks(self, command: SubscribeQuoteTicks) -> None:
        oanda_symbol = instrument_id_to_oanda_symbol(command.instrument_id)
        self._subscribed_quote_instruments.add(oanda_symbol)
        self._log.info(f"Subscribed to quote ticks for {command.instrument_id}")
        await self._restart_price_stream()

    async def _unsubscribe_quote_ticks(self, command: UnsubscribeQuoteTicks) -> None:
        oanda_symbol = instrument_id_to_oanda_symbol(command.instrument_id)
        self._subscribed_quote_instruments.discard(oanda_symbol)
        self._log.info(f"Unsubscribed from quote ticks for {command.instrument_id}")
        if self._subscribed_quote_instruments:
            await self._restart_price_stream()
        else:
            self._stream_running = False
            if self._stream_task and not self._stream_task.done():
                self._stream_task.cancel()

    async def _subscribe_instrument(self, command: SubscribeInstrument) -> None:
        await self._instrument_provider.load_async(command.instrument_id)
        instrument = self._instrument_provider.find(command.instrument_id)
        if instrument:
            self._handle_data(instrument)

    async def _subscribe_instruments(self, command: SubscribeInstruments) -> None:
        await self._instrument_provider.load_all_async()
        for instrument in self._instrument_provider.get_all().values():
            self._handle_data(instrument)

    async def _subscribe_bars(self, command: SubscribeBars) -> None:
        self._log.info(f"Subscribed to bars {command.bar_type} (polling not yet implemented)")

    async def _unsubscribe_bars(self, command: UnsubscribeBars) -> None:
        self._log.info(f"Unsubscribed from bars {command.bar_type}")

    # -- Requests ----------------------------------------------------------------

    async def _request_instrument(self, request: RequestInstrument) -> None:
        await self._instrument_provider.load_async(request.instrument_id)
        instrument = self._instrument_provider.find(request.instrument_id)
        if instrument:
            self._handle_instrument(instrument, request.id, request.ts_init)

    async def _request_instruments(self, request: RequestInstruments) -> None:
        await self._instrument_provider.load_all_async()
        instruments = list(self._instrument_provider.get_all().values())
        self._handle_instruments(instruments, request.venue, request.id, request.ts_init)

    async def _request_quote_ticks(self, request: RequestQuoteTicks) -> None:
        self._log.warning("Historical quote tick requests not supported by OANDA")

    async def _request_bars(self, request: RequestBars) -> None:
        """Request historical bars from OANDA."""
        bar_type = request.bar_type
        instrument_id = bar_type.instrument_id
        oanda_symbol = instrument_id_to_oanda_symbol(instrument_id)

        # Map bar spec to OANDA granularity
        bar_spec = bar_type.spec
        step_secs = bar_spec.step * self._bar_aggregation_to_seconds(bar_spec.aggregation)
        granularity = OANDA_GRANULARITIES.get(step_secs)

        if granularity is None:
            self._log.error(f"Unsupported bar aggregation for OANDA: {bar_spec}")
            return

        # Build request params
        count = min(request.limit, OANDA_MAX_CANDLES) if request.limit else 500
        from_time = request.start.isoformat() if request.start else None
        to_time = request.end.isoformat() if request.end else None

        try:
            response = await self._client.get_candles(
                instrument=oanda_symbol,
                granularity=granularity,
                count=count if not from_time else None,
                from_time=from_time,
                to_time=to_time,
            )

            ts_init = self._clock.timestamp_ns()
            bars = parse_candles_to_bars(response, bar_type, ts_init)
            self._handle_bars(bar_type, bars, None, request.id, request.ts_init)

            self._log.info(f"Received {len(bars)} bars for {bar_type}")

        except Exception as e:
            self._log.error(f"Failed to request bars for {bar_type}: {e}")

    # -- Streaming ---------------------------------------------------------------

    async def _restart_price_stream(self) -> None:
        """Restart the pricing stream with current subscriptions."""
        self._stream_running = False
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._subscribed_quote_instruments:
            self._stream_running = True
            self._stream_task = self.create_task(
                self._run_price_stream(),
                log_msg="oanda_price_stream",
            )

    async def _run_price_stream(self) -> None:
        """Run the OANDA price stream, reconnecting on failure."""
        instruments = list(self._subscribed_quote_instruments)
        backoff = 1.0

        while self._stream_running:
            try:
                self._log.info(
                    f"Starting price stream for {len(instruments)} instruments",
                    color=LogColor.BLUE,
                )
                async for msg in self._client.stream_pricing(instruments):
                    if not self._stream_running:
                        break

                    if msg.get("type") == "HEARTBEAT":
                        continue

                    if msg.get("type") == "PRICE":
                        oanda_symbol = msg.get("instrument", "")
                        instrument_id = oanda_symbol_to_instrument_id(oanda_symbol)
                        ts_init = self._clock.timestamp_ns()
                        tick = parse_quote_tick(msg, instrument_id, ts_init)
                        if tick:
                            self._handle_data(tick)

                backoff = 1.0  # Reset on clean exit

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._stream_running:
                    break
                self._log.warning(f"Price stream error: {e}, reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                instruments = list(self._subscribed_quote_instruments)

    @staticmethod
    def _bar_aggregation_to_seconds(aggregation) -> int:
        """Convert a BarAggregation enum to seconds multiplier."""
        from nautilus_trader.model.enums import BarAggregation

        mapping = {
            BarAggregation.SECOND: 1,
            BarAggregation.MINUTE: 60,
            BarAggregation.HOUR: 3600,
            BarAggregation.DAY: 86400,
            BarAggregation.WEEK: 604800,
        }
        return mapping.get(aggregation, 60)
