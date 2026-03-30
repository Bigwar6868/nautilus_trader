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
Factory classes for creating OANDA live data and execution clients.
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache

from nautilus_trader.adapters.oanda.config import OandaDataClientConfig
from nautilus_trader.adapters.oanda.config import OandaExecClientConfig
from nautilus_trader.adapters.oanda.data import OandaDataClient
from nautilus_trader.adapters.oanda.enums import OandaEnvironment
from nautilus_trader.adapters.oanda.execution import OandaExecutionClient
from nautilus_trader.adapters.oanda.http_client import OandaHttpClient
from nautilus_trader.adapters.oanda.providers import OandaInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.live.factories import LiveDataClientFactory
from nautilus_trader.live.factories import LiveExecClientFactory
from nautilus_trader.model.objects import Currency


def _get_api_token(config_token: str | None) -> str:
    """Resolve the OANDA API token from config or environment."""
    token = config_token or os.environ.get("OANDA_API_TOKEN")
    if not token:
        raise ValueError(
            "OANDA API token not provided. "
            "Set `api_token` in config or the `OANDA_API_TOKEN` environment variable.",
        )
    return token


def _get_account_id(config_account_id: str | None) -> str:
    """Resolve the OANDA account ID from config or environment."""
    account_id = config_account_id or os.environ.get("OANDA_ACCOUNT_ID")
    if not account_id:
        raise ValueError(
            "OANDA account ID not provided. "
            "Set `account_id` in config or the `OANDA_ACCOUNT_ID` environment variable.",
        )
    return account_id


@lru_cache(1)
def get_cached_oanda_http_client(
    api_token: str,
    account_id: str,
    environment: OandaEnvironment = OandaEnvironment.PRACTICE,
    base_url_rest: str | None = None,
    base_url_stream: str | None = None,
) -> OandaHttpClient:
    """
    Cache and return an OANDA HTTP client.

    If a cached client with matching parameters already exists,
    the cached client will be returned.

    Parameters
    ----------
    api_token : str
        The OANDA API personal access token.
    account_id : str
        The OANDA account ID.
    environment : OandaEnvironment, default PRACTICE
        The OANDA environment.
    base_url_rest : str, optional
        Override REST base URL.
    base_url_stream : str, optional
        Override streaming base URL.

    Returns
    -------
    OandaHttpClient

    """
    return OandaHttpClient(
        api_token=api_token,
        account_id=account_id,
        environment=environment,
        base_url_rest=base_url_rest,
        base_url_stream=base_url_stream,
    )


@lru_cache(1)
def get_cached_oanda_instrument_provider(
    client: OandaHttpClient,
    config: InstrumentProviderConfig,
) -> OandaInstrumentProvider:
    """
    Cache and return an OANDA instrument provider.

    Parameters
    ----------
    client : OandaHttpClient
        The OANDA HTTP client.
    config : InstrumentProviderConfig
        The instrument provider configuration.

    Returns
    -------
    OandaInstrumentProvider

    """
    return OandaInstrumentProvider(
        client=client,
        config=config,
    )


class OandaLiveDataClientFactory(LiveDataClientFactory):
    """
    Provides a factory for creating OANDA live data clients.
    """

    @staticmethod
    def create(  # type: ignore
        loop: asyncio.AbstractEventLoop,
        name: str,
        config: OandaDataClientConfig,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> OandaDataClient:
        """
        Create a new OANDA data client.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The event loop for the client.
        name : str
            The custom client ID.
        config : OandaDataClientConfig
            The client configuration.
        msgbus : MessageBus
            The message bus for the client.
        cache : Cache
            The cache for the client.
        clock : LiveClock
            The clock for the client.

        Returns
        -------
        OandaDataClient

        """
        api_token = _get_api_token(config.api_token)
        account_id = _get_account_id(config.account_id)

        client = get_cached_oanda_http_client(
            api_token=api_token,
            account_id=account_id,
            environment=config.environment,
            base_url_rest=config.base_url_rest,
            base_url_stream=config.base_url_stream,
        )

        provider = get_cached_oanda_instrument_provider(
            client=client,
            config=config.instrument_provider,
        )

        return OandaDataClient(
            loop=loop,
            client=client,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=provider,
            name=name,
            config=config,
        )


class OandaLiveExecClientFactory(LiveExecClientFactory):
    """
    Provides a factory for creating OANDA live execution clients.
    """

    @staticmethod
    def create(  # type: ignore
        loop: asyncio.AbstractEventLoop,
        name: str,
        config: OandaExecClientConfig,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
    ) -> OandaExecutionClient:
        """
        Create a new OANDA execution client.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The event loop for the client.
        name : str
            The custom client ID.
        config : OandaExecClientConfig
            The configuration for the client.
        msgbus : MessageBus
            The message bus for the client.
        cache : Cache
            The cache for the client.
        clock : LiveClock
            The clock for the client.

        Returns
        -------
        OandaExecutionClient

        """
        api_token = _get_api_token(config.api_token)
        account_id = _get_account_id(config.account_id)

        client = get_cached_oanda_http_client(
            api_token=api_token,
            account_id=account_id,
            environment=config.environment,
            base_url_rest=config.base_url_rest,
            base_url_stream=config.base_url_stream,
        )

        provider = get_cached_oanda_instrument_provider(
            client=client,
            config=config.instrument_provider,
        )

        return OandaExecutionClient(
            loop=loop,
            client=client,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=provider,
            base_currency=Currency.from_str("USD"),
            name=name,
            config=config,
        )
