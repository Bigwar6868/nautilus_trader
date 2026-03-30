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
Async HTTP client for the OANDA v20 REST and Streaming APIs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

import aiohttp

from nautilus_trader.adapters.oanda.constants import OANDA_API_VERSION
from nautilus_trader.adapters.oanda.constants import OANDA_REST_LIVE
from nautilus_trader.adapters.oanda.constants import OANDA_REST_PRACTICE
from nautilus_trader.adapters.oanda.constants import OANDA_STREAM_LIVE
from nautilus_trader.adapters.oanda.constants import OANDA_STREAM_PRACTICE
from nautilus_trader.adapters.oanda.enums import OandaEnvironment


log = logging.getLogger(__name__)


class OandaHttpClient:
    """
    Async HTTP client for the OANDA v20 API.

    Handles both REST requests and streaming connections (chunked HTTP).

    Parameters
    ----------
    api_token : str
        The OANDA API personal access token.
    account_id : str
        The OANDA account ID.
    environment : OandaEnvironment
        The OANDA environment (PRACTICE or LIVE).
    base_url_rest : str, optional
        Override REST base URL.
    base_url_stream : str, optional
        Override streaming base URL.

    """

    def __init__(
        self,
        api_token: str,
        account_id: str,
        environment: OandaEnvironment = OandaEnvironment.PRACTICE,
        base_url_rest: str | None = None,
        base_url_stream: str | None = None,
    ) -> None:
        self._api_token = api_token
        self._account_id = account_id
        self._environment = environment

        if environment == OandaEnvironment.LIVE:
            self._base_url_rest = base_url_rest or OANDA_REST_LIVE
            self._base_url_stream = base_url_stream or OANDA_STREAM_LIVE
        else:
            self._base_url_rest = base_url_rest or OANDA_REST_PRACTICE
            self._base_url_stream = base_url_stream or OANDA_STREAM_PRACTICE

        self._session: aiohttp.ClientSession | None = None
        self._stream_session: aiohttp.ClientSession | None = None
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def connected(self) -> bool:
        return self._session is not None and not self._session.closed

    async def connect(self) -> None:
        """Open HTTP sessions."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._headers)
        if self._stream_session is None or self._stream_session.closed:
            self._stream_session = aiohttp.ClientSession(headers=self._headers)

    async def disconnect(self) -> None:
        """Close HTTP sessions."""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._stream_session and not self._stream_session.closed:
            await self._stream_session.close()

    # -- REST API Methods -------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        data: dict | None = None,
        base_url: str | None = None,
    ) -> dict:
        """Make an HTTP request to the OANDA v20 REST API."""
        if self._session is None or self._session.closed:
            raise RuntimeError("OandaHttpClient is not connected")

        url = f"{base_url or self._base_url_rest}{OANDA_API_VERSION}{path}"
        body = json.dumps(data) if data else None

        async with self._session.request(
            method,
            url,
            params=params,
            data=body,
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise OandaApiError(resp.status, text, url)
            return json.loads(text)

    async def get(self, path: str, params: dict | None = None) -> dict:
        return await self._request("GET", path, params=params)

    async def post(self, path: str, data: dict | None = None) -> dict:
        return await self._request("POST", path, data=data)

    async def put(self, path: str, data: dict | None = None) -> dict:
        return await self._request("PUT", path, data=data)

    async def patch(self, path: str, data: dict | None = None) -> dict:
        return await self._request("PATCH", path, data=data)

    # -- Account Endpoints -------------------------------------------------------

    async def get_account(self) -> dict:
        """GET /v3/accounts/{accountID}"""
        return await self.get(f"/accounts/{self._account_id}")

    async def get_account_summary(self) -> dict:
        """GET /v3/accounts/{accountID}/summary"""
        return await self.get(f"/accounts/{self._account_id}/summary")

    async def get_account_instruments(self, instruments: list[str] | None = None) -> dict:
        """GET /v3/accounts/{accountID}/instruments"""
        params = {}
        if instruments:
            params["instruments"] = ",".join(instruments)
        return await self.get(f"/accounts/{self._account_id}/instruments", params=params or None)

    # -- Instrument / Market Data Endpoints --------------------------------------

    async def get_candles(
        self,
        instrument: str,
        granularity: str = "M1",
        count: int | None = None,
        from_time: str | None = None,
        to_time: str | None = None,
        price: str = "MBA",
    ) -> dict:
        """GET /v3/instruments/{instrument}/candles"""
        params: dict = {
            "granularity": granularity,
            "price": price,
        }
        if count is not None:
            params["count"] = str(count)
        if from_time is not None:
            params["from"] = from_time
        if to_time is not None:
            params["to"] = to_time
        return await self.get(f"/instruments/{instrument}/candles", params=params)

    async def get_pricing(self, instruments: list[str]) -> dict:
        """GET /v3/accounts/{accountID}/pricing (snapshot)"""
        params = {"instruments": ",".join(instruments)}
        return await self.get(f"/accounts/{self._account_id}/pricing", params=params)

    # -- Streaming Endpoints -----------------------------------------------------

    async def stream_pricing(
        self,
        instruments: list[str],
    ) -> AsyncGenerator[dict, None]:
        """
        Stream live prices from OANDA.

        Yields PRICE and HEARTBEAT JSON objects via HTTP chunked streaming.
        """
        if self._stream_session is None or self._stream_session.closed:
            raise RuntimeError("OandaHttpClient stream session is not connected")

        url = (
            f"{self._base_url_stream}{OANDA_API_VERSION}"
            f"/accounts/{self._account_id}/pricing/stream"
        )
        params = {
            "instruments": ",".join(instruments),
            "snapshot": "true",
        }

        async with self._stream_session.get(url, params=params) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise OandaApiError(resp.status, text, url)
            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    log.warning(f"Failed to decode streaming line: {line!r}")

    async def stream_transactions(self) -> AsyncGenerator[dict, None]:
        """
        Stream live transactions from OANDA.

        Yields transaction JSON objects and heartbeats via HTTP chunked streaming.
        """
        if self._stream_session is None or self._stream_session.closed:
            raise RuntimeError("OandaHttpClient stream session is not connected")

        url = (
            f"{self._base_url_stream}{OANDA_API_VERSION}"
            f"/accounts/{self._account_id}/transactions/stream"
        )

        async with self._stream_session.get(url) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise OandaApiError(resp.status, text, url)
            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    log.warning(f"Failed to decode transaction line: {line!r}")

    # -- Order Endpoints ---------------------------------------------------------

    async def create_order(self, order_body: dict) -> dict:
        """POST /v3/accounts/{accountID}/orders"""
        return await self.post(
            f"/accounts/{self._account_id}/orders",
            data={"order": order_body},
        )

    async def get_orders(self, instrument: str | None = None) -> dict:
        """GET /v3/accounts/{accountID}/orders (pending orders)"""
        params = {}
        if instrument:
            params["instrument"] = instrument
        return await self.get(
            f"/accounts/{self._account_id}/orders",
            params=params or None,
        )

    async def get_order(self, order_id: str) -> dict:
        """GET /v3/accounts/{accountID}/orders/{orderID}"""
        return await self.get(f"/accounts/{self._account_id}/orders/{order_id}")

    async def cancel_order(self, order_id: str) -> dict:
        """PUT /v3/accounts/{accountID}/orders/{orderID}/cancel"""
        return await self.put(f"/accounts/{self._account_id}/orders/{order_id}/cancel")

    async def replace_order(self, order_id: str, order_body: dict) -> dict:
        """PUT /v3/accounts/{accountID}/orders/{orderID}"""
        return await self.put(
            f"/accounts/{self._account_id}/orders/{order_id}",
            data={"order": order_body},
        )

    # -- Trade Endpoints ---------------------------------------------------------

    async def get_trades(self, instrument: str | None = None, state: str = "OPEN") -> dict:
        """GET /v3/accounts/{accountID}/trades"""
        params: dict = {"state": state}
        if instrument:
            params["instrument"] = instrument
        return await self.get(f"/accounts/{self._account_id}/trades", params=params)

    async def get_trade(self, trade_id: str) -> dict:
        """GET /v3/accounts/{accountID}/trades/{tradeID}"""
        return await self.get(f"/accounts/{self._account_id}/trades/{trade_id}")

    async def close_trade(self, trade_id: str, units: str = "ALL") -> dict:
        """PUT /v3/accounts/{accountID}/trades/{tradeID}/close"""
        return await self.put(
            f"/accounts/{self._account_id}/trades/{trade_id}/close",
            data={"units": units},
        )

    # -- Position Endpoints ------------------------------------------------------

    async def get_positions(self) -> dict:
        """GET /v3/accounts/{accountID}/positions"""
        return await self.get(f"/accounts/{self._account_id}/positions")

    async def get_position(self, instrument: str) -> dict:
        """GET /v3/accounts/{accountID}/positions/{instrument}"""
        return await self.get(f"/accounts/{self._account_id}/positions/{instrument}")

    async def close_position(
        self,
        instrument: str,
        long_units: str | None = None,
        short_units: str | None = None,
    ) -> dict:
        """PUT /v3/accounts/{accountID}/positions/{instrument}/close"""
        body: dict = {}
        if long_units:
            body["longUnits"] = long_units
        if short_units:
            body["shortUnits"] = short_units
        return await self.put(
            f"/accounts/{self._account_id}/positions/{instrument}/close",
            data=body,
        )

    # -- Transaction Endpoints ---------------------------------------------------

    async def get_transactions(
        self,
        from_time: str | None = None,
        to_time: str | None = None,
        page_size: int = 100,
        tx_type: str | None = None,
    ) -> dict:
        """GET /v3/accounts/{accountID}/transactions"""
        params: dict = {"pageSize": str(page_size)}
        if from_time:
            params["from"] = from_time
        if to_time:
            params["to"] = to_time
        if tx_type:
            params["type"] = tx_type
        return await self.get(f"/accounts/{self._account_id}/transactions", params=params)

    async def get_transactions_since_id(self, transaction_id: str) -> dict:
        """GET /v3/accounts/{accountID}/transactions/sinceid"""
        return await self.get(
            f"/accounts/{self._account_id}/transactions/sinceid",
            params={"id": transaction_id},
        )


class OandaApiError(Exception):
    """Raised when the OANDA API returns an error response."""

    def __init__(self, status: int, message: str, url: str) -> None:
        self.status = status
        self.message = message
        self.url = url
        super().__init__(f"OANDA API error {status} for {url}: {message}")
