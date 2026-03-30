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
OANDA live execution client.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd

from nautilus_trader.adapters.oanda.constants import OANDA_VENUE
from nautilus_trader.adapters.oanda.http_client import OandaApiError
from nautilus_trader.adapters.oanda.http_client import OandaHttpClient
from nautilus_trader.adapters.oanda.parsing import instrument_id_to_oanda_symbol
from nautilus_trader.adapters.oanda.parsing import oanda_symbol_to_instrument_id
from nautilus_trader.adapters.oanda.providers import OandaInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.enums import LogColor
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.execution.reports import FillReport
from nautilus_trader.execution.reports import OrderStatusReport
from nautilus_trader.execution.reports import PositionStatusReport
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import LiquiditySide
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import OrderStatus
from nautilus_trader.model.enums import OrderType
from nautilus_trader.model.enums import PositionSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import AccountId
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.identifiers import VenueOrderId
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

if TYPE_CHECKING:
    from nautilus_trader.execution.messages import CancelAllOrders
    from nautilus_trader.execution.messages import CancelOrder
    from nautilus_trader.execution.messages import GenerateFillReports
    from nautilus_trader.execution.messages import GenerateOrderStatusReport
    from nautilus_trader.execution.messages import GenerateOrderStatusReports
    from nautilus_trader.execution.messages import GeneratePositionStatusReports
    from nautilus_trader.execution.messages import ModifyOrder
    from nautilus_trader.execution.messages import SubmitOrder
    from nautilus_trader.execution.reports import ExecutionMassStatus


class OandaExecutionClient(LiveExecutionClient):
    """
    Provides a live execution client for the OANDA venue.

    Supports market, limit, stop, and market-if-touched orders via the
    OANDA v20 REST API. Monitors fills via the transaction stream.

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
    base_currency : Currency
        The account base currency.
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
        base_currency: Currency,
        name: str | None = None,
        config: NautilusConfig | None = None,
    ) -> None:
        super().__init__(
            loop=loop,
            client_id=ClientId(name or OANDA_VENUE.value),
            venue=OANDA_VENUE,
            oms_type=OmsType.HEDGING,
            account_type=AccountType.MARGIN,
            base_currency=base_currency,
            instrument_provider=instrument_provider,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            config=config,
        )
        self._client = client
        self._instrument_provider = instrument_provider

        # Transaction stream state
        self._transaction_stream_task: asyncio.Task | None = None
        self._stream_running = False

        # Order ID mappings (client_order_id <-> venue_order_id)
        self._order_id_map: dict[str, ClientOrderId] = {}

    @property
    def account_id(self) -> AccountId:
        return AccountId(f"OANDA-{self._client.account_id}")

    async def _connect(self) -> None:
        """Connect to OANDA for execution."""
        await self._client.connect()
        await self._instrument_provider.initialize()

        # Start the transaction stream for fill monitoring
        self._stream_running = True
        self._transaction_stream_task = self.create_task(
            self._run_transaction_stream(),
            log_msg="oanda_transaction_stream",
        )

        self._log.info("OANDA execution client connected", color=LogColor.GREEN)

    async def _disconnect(self) -> None:
        """Disconnect from OANDA."""
        self._stream_running = False
        if self._transaction_stream_task and not self._transaction_stream_task.done():
            self._transaction_stream_task.cancel()
            try:
                await self._transaction_stream_task
            except asyncio.CancelledError:
                pass
        await self._client.disconnect()
        self._log.info("OANDA execution client disconnected")

    # -- Order Commands ----------------------------------------------------------

    async def _submit_order(self, command: SubmitOrder) -> None:
        """Submit an order to OANDA."""
        order = command.order
        instrument_id = order.instrument_id
        oanda_symbol = instrument_id_to_oanda_symbol(instrument_id)

        # Build OANDA order body
        order_body = self._build_order_body(order, oanda_symbol)

        try:
            self.generate_order_submitted(
                strategy_id=order.strategy_id,
                instrument_id=instrument_id,
                client_order_id=order.client_order_id,
                ts_event=self._clock.timestamp_ns(),
            )

            response = await self._client.create_order(order_body)

            # Handle the response
            self._handle_order_response(response, order)

        except OandaApiError as e:
            self.generate_order_rejected(
                strategy_id=order.strategy_id,
                instrument_id=instrument_id,
                client_order_id=order.client_order_id,
                reason=str(e),
                ts_event=self._clock.timestamp_ns(),
            )
        except Exception as e:
            self._log.error(f"Error submitting order: {e}")
            self.generate_order_rejected(
                strategy_id=order.strategy_id,
                instrument_id=instrument_id,
                client_order_id=order.client_order_id,
                reason=str(e),
                ts_event=self._clock.timestamp_ns(),
            )

    async def _modify_order(self, command: ModifyOrder) -> None:
        """Modify a pending order on OANDA."""
        venue_order_id = command.venue_order_id
        if venue_order_id is None:
            self._log.error(f"Cannot modify order: no venue order ID for {command.client_order_id}")
            return

        order = self._cache.order(command.client_order_id)
        if order is None:
            self._log.error(f"Cannot modify order: {command.client_order_id} not found in cache")
            return

        oanda_symbol = instrument_id_to_oanda_symbol(order.instrument_id)
        order_body = self._build_order_body(order, oanda_symbol)

        # Apply modifications
        if command.price is not None:
            order_body["price"] = str(command.price)
        if command.quantity is not None:
            units = int(command.quantity)
            if order.side == OrderSide.SELL:
                units = -units
            order_body["units"] = str(units)

        try:
            await self._client.replace_order(str(venue_order_id), order_body)
            self._log.info(f"Modified order {venue_order_id}")
        except OandaApiError as e:
            self._log.error(f"Failed to modify order {venue_order_id}: {e}")

    async def _cancel_order(self, command: CancelOrder) -> None:
        """Cancel a pending order on OANDA."""
        venue_order_id = command.venue_order_id
        if venue_order_id is None:
            self._log.error(f"Cannot cancel order: no venue order ID for {command.client_order_id}")
            return

        try:
            await self._client.cancel_order(str(venue_order_id))

            self.generate_order_canceled(
                strategy_id=command.strategy_id,
                instrument_id=command.instrument_id,
                client_order_id=command.client_order_id,
                venue_order_id=venue_order_id,
                ts_event=self._clock.timestamp_ns(),
            )

        except OandaApiError as e:
            self._log.error(f"Failed to cancel order {venue_order_id}: {e}")

    async def _cancel_all_orders(self, command: CancelAllOrders) -> None:
        """Cancel all pending orders for an instrument on OANDA."""
        oanda_symbol = instrument_id_to_oanda_symbol(command.instrument_id)

        try:
            response = await self._client.get_orders(instrument=oanda_symbol)
            orders = response.get("orders", [])

            for oanda_order in orders:
                order_id = oanda_order.get("id")
                if order_id:
                    try:
                        await self._client.cancel_order(order_id)
                    except OandaApiError as e:
                        self._log.warning(f"Failed to cancel order {order_id}: {e}")

            self._log.info(f"Cancelled {len(orders)} orders for {command.instrument_id}")

        except OandaApiError as e:
            self._log.error(f"Failed to cancel all orders for {command.instrument_id}: {e}")

    # -- Execution Reports -------------------------------------------------------

    async def generate_order_status_report(
        self,
        command: GenerateOrderStatusReport,
    ) -> OrderStatusReport | None:
        """Generate an order status report for a specific order."""
        venue_order_id = command.venue_order_id
        if venue_order_id is None:
            return None

        try:
            response = await self._client.get_order(str(venue_order_id))
            oanda_order = response.get("order", {})
            return self._parse_order_status_report(oanda_order)
        except OandaApiError as e:
            self._log.error(f"Failed to get order status for {venue_order_id}: {e}")
            return None

    async def generate_order_status_reports(
        self,
        command: GenerateOrderStatusReports,
    ) -> list[OrderStatusReport]:
        """Generate order status reports for all pending orders."""
        reports = []
        try:
            instrument_filter = None
            if command.instrument_id:
                instrument_filter = instrument_id_to_oanda_symbol(command.instrument_id)

            response = await self._client.get_orders(instrument=instrument_filter)
            for oanda_order in response.get("orders", []):
                report = self._parse_order_status_report(oanda_order)
                if report:
                    reports.append(report)
        except OandaApiError as e:
            self._log.error(f"Failed to generate order status reports: {e}")

        return reports

    async def generate_fill_reports(
        self,
        command: GenerateFillReports,
    ) -> list[FillReport]:
        """Generate fill reports from recent transactions."""
        reports = []
        try:
            response = await self._client.get_transactions(tx_type="ORDER_FILL")
            for tx in response.get("transactions", []):
                report = self._parse_fill_report(tx)
                if report:
                    reports.append(report)
        except OandaApiError as e:
            self._log.error(f"Failed to generate fill reports: {e}")

        return reports

    async def generate_position_status_reports(
        self,
        command: GeneratePositionStatusReports,
    ) -> list[PositionStatusReport]:
        """Generate position status reports."""
        reports = []
        try:
            response = await self._client.get_positions()
            for pos in response.get("positions", []):
                long_units = float(pos.get("long", {}).get("units", "0"))
                short_units = float(pos.get("short", {}).get("units", "0"))

                if long_units == 0 and short_units == 0:
                    continue

                oanda_symbol = pos.get("instrument", "")
                instrument_id = oanda_symbol_to_instrument_id(oanda_symbol)

                if long_units != 0:
                    reports.append(PositionStatusReport(
                        account_id=self.account_id,
                        instrument_id=instrument_id,
                        position_side=PositionSide.LONG,
                        quantity=Quantity.from_str(str(abs(long_units))),
                        report_id=UUID4(),
                        ts_last=self._clock.timestamp_ns(),
                        ts_init=self._clock.timestamp_ns(),
                    ))

                if short_units != 0:
                    reports.append(PositionStatusReport(
                        account_id=self.account_id,
                        instrument_id=instrument_id,
                        position_side=PositionSide.SHORT,
                        quantity=Quantity.from_str(str(abs(short_units))),
                        report_id=UUID4(),
                        ts_last=self._clock.timestamp_ns(),
                        ts_init=self._clock.timestamp_ns(),
                    ))

        except OandaApiError as e:
            self._log.error(f"Failed to generate position status reports: {e}")

        return reports

    async def generate_mass_status(
        self,
        lookback_mins: int | None = None,
    ) -> ExecutionMassStatus | None:
        """Generate a mass status report."""
        from nautilus_trader.execution.reports import ExecutionMassStatus

        ts_now = self._clock.timestamp_ns()
        mass_status = ExecutionMassStatus(
            client_id=self.client_id,
            account_id=self.account_id,
            venue=OANDA_VENUE,
            report_id=UUID4(),
            ts_init=ts_now,
        )

        # Add order reports
        try:
            response = await self._client.get_orders()
            for oanda_order in response.get("orders", []):
                report = self._parse_order_status_report(oanda_order)
                if report:
                    mass_status.add_order_reports([report])
        except Exception as e:
            self._log.warning(f"Error fetching orders for mass status: {e}")

        # Add position reports
        try:
            response = await self._client.get_positions()
            for pos in response.get("positions", []):
                long_units = float(pos.get("long", {}).get("units", "0"))
                short_units = float(pos.get("short", {}).get("units", "0"))

                if long_units == 0 and short_units == 0:
                    continue

                oanda_symbol = pos.get("instrument", "")
                instrument_id = oanda_symbol_to_instrument_id(oanda_symbol)

                if long_units != 0:
                    mass_status.add_position_reports([PositionStatusReport(
                        account_id=self.account_id,
                        instrument_id=instrument_id,
                        position_side=PositionSide.LONG,
                        quantity=Quantity.from_str(str(abs(long_units))),
                        report_id=UUID4(),
                        ts_last=ts_now,
                        ts_init=ts_now,
                    )])
                if short_units != 0:
                    mass_status.add_position_reports([PositionStatusReport(
                        account_id=self.account_id,
                        instrument_id=instrument_id,
                        position_side=PositionSide.SHORT,
                        quantity=Quantity.from_str(str(abs(short_units))),
                        report_id=UUID4(),
                        ts_last=ts_now,
                        ts_init=ts_now,
                    )])
        except Exception as e:
            self._log.warning(f"Error fetching positions for mass status: {e}")

        return mass_status

    # -- Internal Helpers --------------------------------------------------------

    def _build_order_body(self, order, oanda_symbol: str) -> dict:
        """Build an OANDA order body from a NautilusTrader order."""
        units = int(order.quantity)
        if order.side == OrderSide.SELL:
            units = -units

        body: dict = {
            "instrument": oanda_symbol,
            "units": str(units),
            "positionFill": "DEFAULT",
        }

        # Map NautilusTrader order type to OANDA
        if order.order_type == OrderType.MARKET:
            body["type"] = "MARKET"
            body["timeInForce"] = "FOK"
        elif order.order_type == OrderType.LIMIT:
            body["type"] = "LIMIT"
            body["price"] = str(order.price)
            body["timeInForce"] = self._map_time_in_force(order.time_in_force)
        elif order.order_type == OrderType.STOP_MARKET:
            body["type"] = "STOP"
            body["price"] = str(order.trigger_price)
            body["timeInForce"] = self._map_time_in_force(order.time_in_force)
        elif order.order_type == OrderType.STOP_LIMIT:
            body["type"] = "MARKET_IF_TOUCHED"
            body["price"] = str(order.price)
            body["timeInForce"] = self._map_time_in_force(order.time_in_force)
        else:
            body["type"] = "MARKET"
            body["timeInForce"] = "FOK"

        # Add client extensions for tracking
        body["clientExtensions"] = {
            "id": str(order.client_order_id),
        }

        return body

    @staticmethod
    def _map_time_in_force(tif: TimeInForce) -> str:
        """Map NautilusTrader TimeInForce to OANDA string."""
        mapping = {
            TimeInForce.GTC: "GTC",
            TimeInForce.GTD: "GTD",
            TimeInForce.FOK: "FOK",
            TimeInForce.IOC: "IOC",
            TimeInForce.DAY: "GFD",
        }
        return mapping.get(tif, "GTC")

    def _handle_order_response(self, response: dict, order) -> None:
        """Process an OANDA order creation response."""
        # Market order fill
        order_fill = response.get("orderFillTransaction")
        if order_fill:
            venue_order_id = VenueOrderId(str(order_fill.get("orderID", "")))
            trade_id = TradeId(str(order_fill.get("id", "")))

            self._order_id_map[str(venue_order_id)] = order.client_order_id

            self.generate_order_accepted(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=order.client_order_id,
                venue_order_id=venue_order_id,
                ts_event=self._clock.timestamp_ns(),
            )

            fill_price = order_fill.get("price", "0")
            fill_units = abs(float(order_fill.get("units", "0")))
            commission = order_fill.get("commission", "0")

            instrument = self._instrument_provider.find(order.instrument_id)
            quote_currency = instrument.quote_currency if instrument else Currency.from_str("USD")

            self.generate_order_filled(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=order.client_order_id,
                venue_order_id=venue_order_id,
                trade_id=trade_id,
                order_side=order.side,
                order_type=order.order_type,
                last_qty=Quantity.from_str(str(fill_units)),
                last_px=Price.from_str(fill_price),
                quote_currency=quote_currency,
                commission=Money(Decimal(str(commission)), quote_currency),
                liquidity_side=LiquiditySide.TAKER,
                ts_event=self._clock.timestamp_ns(),
            )
            return

        # Pending order accepted
        order_create = response.get("orderCreateTransaction")
        if order_create:
            venue_order_id = VenueOrderId(str(order_create.get("id", "")))
            self._order_id_map[str(venue_order_id)] = order.client_order_id

            self.generate_order_accepted(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=order.client_order_id,
                venue_order_id=venue_order_id,
                ts_event=self._clock.timestamp_ns(),
            )
            return

        # Order rejected
        reject_tx = response.get("orderRejectTransaction")
        if reject_tx:
            reason = reject_tx.get("rejectReason", "Unknown rejection")
            self.generate_order_rejected(
                strategy_id=order.strategy_id,
                instrument_id=order.instrument_id,
                client_order_id=order.client_order_id,
                reason=reason,
                ts_event=self._clock.timestamp_ns(),
            )

    def _parse_order_status_report(self, oanda_order: dict) -> OrderStatusReport | None:
        """Parse an OANDA order object into an OrderStatusReport."""
        try:
            oanda_symbol = oanda_order.get("instrument", "")
            instrument_id = oanda_symbol_to_instrument_id(oanda_symbol)

            venue_order_id = VenueOrderId(str(oanda_order.get("id", "")))

            # Determine client order ID from extensions
            client_ext = oanda_order.get("clientExtensions", {})
            client_order_id_str = client_ext.get("id", str(venue_order_id))
            client_order_id = ClientOrderId(client_order_id_str)

            # Parse order type
            oanda_type = oanda_order.get("type", "MARKET")
            order_type = self._parse_order_type(oanda_type)

            # Parse side from units
            units = float(oanda_order.get("units", "0"))
            side = OrderSide.BUY if units > 0 else OrderSide.SELL

            # Parse state
            state = oanda_order.get("state", "PENDING")
            order_status = self._parse_order_state(state)

            price_str = oanda_order.get("price")
            price = Price.from_str(price_str) if price_str else None

            return OrderStatusReport(
                account_id=self.account_id,
                instrument_id=instrument_id,
                client_order_id=client_order_id,
                venue_order_id=venue_order_id,
                order_side=side,
                order_type=order_type,
                time_in_force=TimeInForce.GTC,
                order_status=order_status,
                quantity=Quantity.from_str(str(abs(units))),
                filled_qty=Quantity.from_int(0),
                price=price,
                report_id=UUID4(),
                ts_accepted=self._clock.timestamp_ns(),
                ts_last=self._clock.timestamp_ns(),
                ts_init=self._clock.timestamp_ns(),
            )
        except Exception as e:
            self._log.warning(f"Failed to parse order status report: {e}")
            return None

    def _parse_fill_report(self, tx: dict) -> FillReport | None:
        """Parse an OANDA ORDER_FILL transaction into a FillReport."""
        try:
            oanda_symbol = tx.get("instrument", "")
            if not oanda_symbol:
                return None

            instrument_id = oanda_symbol_to_instrument_id(oanda_symbol)
            venue_order_id = VenueOrderId(str(tx.get("orderID", "")))
            trade_id = TradeId(str(tx.get("id", "")))

            units = float(tx.get("units", "0"))
            side = OrderSide.BUY if units > 0 else OrderSide.SELL
            fill_price = tx.get("price", "0")
            commission = tx.get("commission", "0")

            # Determine client order ID
            client_order_id_str = str(venue_order_id)
            if str(venue_order_id) in self._order_id_map:
                client_order_id = self._order_id_map[str(venue_order_id)]
            else:
                client_order_id = ClientOrderId(client_order_id_str)

            instrument = self._instrument_provider.find(instrument_id)
            quote_currency = instrument.quote_currency if instrument else Currency.from_str("USD")

            ts_event = self._clock.timestamp_ns()
            time_str = tx.get("time")
            if time_str:
                ts_event = int(pd.Timestamp(time_str).value)

            return FillReport(
                account_id=self.account_id,
                instrument_id=instrument_id,
                client_order_id=client_order_id,
                venue_order_id=venue_order_id,
                trade_id=trade_id,
                order_side=side,
                last_qty=Quantity.from_str(str(abs(units))),
                last_px=Price.from_str(fill_price),
                commission=Money(Decimal(str(commission)), quote_currency),
                liquidity_side=LiquiditySide.TAKER,
                report_id=UUID4(),
                ts_event=ts_event,
                ts_init=self._clock.timestamp_ns(),
            )
        except Exception as e:
            self._log.warning(f"Failed to parse fill report: {e}")
            return None

    @staticmethod
    def _parse_order_type(oanda_type: str) -> OrderType:
        mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP": OrderType.STOP_MARKET,
            "MARKET_IF_TOUCHED": OrderType.STOP_LIMIT,
        }
        return mapping.get(oanda_type, OrderType.MARKET)

    @staticmethod
    def _parse_order_state(state: str) -> OrderStatus:
        mapping = {
            "PENDING": OrderStatus.ACCEPTED,
            "FILLED": OrderStatus.FILLED,
            "TRIGGERED": OrderStatus.TRIGGERED,
            "CANCELLED": OrderStatus.CANCELED,
        }
        return mapping.get(state, OrderStatus.ACCEPTED)

    # -- Transaction Stream -------------------------------------------------------

    async def _run_transaction_stream(self) -> None:
        """Run the OANDA transaction stream for real-time fill monitoring."""
        backoff = 1.0

        while self._stream_running:
            try:
                self._log.info("Starting OANDA transaction stream", color=LogColor.BLUE)
                async for msg in self._client.stream_transactions():
                    if not self._stream_running:
                        break

                    if msg.get("type") == "HEARTBEAT":
                        continue

                    tx_type = msg.get("type", "")

                    if tx_type == "ORDER_FILL":
                        self._handle_fill_transaction(msg)
                    elif tx_type in ("ORDER_CANCEL", "ORDER_CANCELLED"):
                        self._handle_cancel_transaction(msg)

                backoff = 1.0

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._stream_running:
                    break
                self._log.warning(f"Transaction stream error: {e}, reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle_fill_transaction(self, tx: dict) -> None:
        """Handle a fill transaction from the stream."""
        venue_order_id_str = tx.get("orderID", "")
        client_order_id = self._order_id_map.get(venue_order_id_str)

        if client_order_id is None:
            # Order not submitted by us — could be manual or another client
            return

        order = self._cache.order(client_order_id)
        if order is None:
            return

        oanda_symbol = tx.get("instrument", "")
        instrument_id = oanda_symbol_to_instrument_id(oanda_symbol)
        instrument = self._instrument_provider.find(instrument_id)
        quote_currency = instrument.quote_currency if instrument else Currency.from_str("USD")

        units = float(tx.get("units", "0"))
        fill_price = tx.get("price", "0")
        commission = tx.get("commission", "0")

        self.generate_order_filled(
            strategy_id=order.strategy_id,
            instrument_id=instrument_id,
            client_order_id=client_order_id,
            venue_order_id=VenueOrderId(venue_order_id_str),
            trade_id=TradeId(str(tx.get("id", ""))),
            order_side=order.side,
            order_type=order.order_type,
            last_qty=Quantity.from_str(str(abs(units))),
            last_px=Price.from_str(fill_price),
            quote_currency=quote_currency,
            commission=Money(Decimal(str(commission)), quote_currency),
            liquidity_side=LiquiditySide.TAKER,
            ts_event=self._clock.timestamp_ns(),
        )

    def _handle_cancel_transaction(self, tx: dict) -> None:
        """Handle a cancel transaction from the stream."""
        venue_order_id_str = tx.get("orderID", "")
        client_order_id = self._order_id_map.get(venue_order_id_str)

        if client_order_id is None:
            return

        order = self._cache.order(client_order_id)
        if order is None:
            return

        self.generate_order_canceled(
            strategy_id=order.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=client_order_id,
            venue_order_id=VenueOrderId(venue_order_id_str),
            ts_event=self._clock.timestamp_ns(),
        )
