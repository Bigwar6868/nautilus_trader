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

from nautilus_trader.adapters.oanda.constants import OANDA_VENUE
from nautilus_trader.adapters.oanda.enums import OandaEnvironment
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveDataClientConfig
from nautilus_trader.config import LiveExecClientConfig
from nautilus_trader.config import PositiveInt
from nautilus_trader.model.identifiers import Venue


class OandaInstrumentProviderConfig(InstrumentProviderConfig, frozen=True):
    """
    Configuration for ``OandaInstrumentProvider`` instances.

    Parameters
    ----------
    load_all : bool, default False
        If all venue instruments should be loaded on start.
    load_ids : frozenset[InstrumentId], optional
        The list of instrument IDs to be loaded on start (if `load_all` is False).
    filters : frozendict or dict[str, Any], optional
        The venue specific instrument loading filters to apply.

    """

    pass


class OandaDataClientConfig(LiveDataClientConfig, frozen=True):
    """
    Configuration for ``OandaDataClient`` instances.

    Parameters
    ----------
    venue : Venue, default OANDA_VENUE
        The venue for the client.
    api_token : str, optional
        The OANDA API personal access token.
        If ``None`` then will source from the ``OANDA_API_TOKEN`` env var.
    account_id : str, optional
        The OANDA account ID (e.g. "101-004-XXXXXXX-001").
        If ``None`` then will source from the ``OANDA_ACCOUNT_ID`` env var.
    environment : OandaEnvironment, default OandaEnvironment.PRACTICE
        The OANDA environment (PRACTICE or LIVE).
    base_url_rest : str, optional
        Override for the REST API base URL.
    base_url_stream : str, optional
        Override for the streaming API base URL.
    update_instruments_interval_mins : PositiveInt or None, default 60
        The interval (minutes) between reloading instruments from the venue.

    """

    venue: Venue = OANDA_VENUE
    api_token: str | None = None
    account_id: str | None = None
    environment: OandaEnvironment = OandaEnvironment.PRACTICE
    base_url_rest: str | None = None
    base_url_stream: str | None = None
    update_instruments_interval_mins: PositiveInt | None = 60


class OandaExecClientConfig(LiveExecClientConfig, frozen=True):
    """
    Configuration for ``OandaExecutionClient`` instances.

    Parameters
    ----------
    venue : Venue, default OANDA_VENUE
        The venue for the client.
    api_token : str, optional
        The OANDA API personal access token.
        If ``None`` then will source from the ``OANDA_API_TOKEN`` env var.
    account_id : str, optional
        The OANDA account ID (e.g. "101-004-XXXXXXX-001").
        If ``None`` then will source from the ``OANDA_ACCOUNT_ID`` env var.
    environment : OandaEnvironment, default OandaEnvironment.PRACTICE
        The OANDA environment (PRACTICE or LIVE).
    base_url_rest : str, optional
        Override for the REST API base URL.
    base_url_stream : str, optional
        Override for the streaming API base URL.
    max_retries : PositiveInt or None, optional
        The maximum number of times a submit/cancel/modify request will be retried.
    retry_delay_secs : float, default 1.0
        The initial delay (seconds) between retries.

    """

    venue: Venue = OANDA_VENUE
    api_token: str | None = None
    account_id: str | None = None
    environment: OandaEnvironment = OandaEnvironment.PRACTICE
    base_url_rest: str | None = None
    base_url_stream: str | None = None
    max_retries: PositiveInt | None = None
    retry_delay_secs: float = 1.0
