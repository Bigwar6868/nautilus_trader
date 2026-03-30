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
OANDA adapter for NautilusTrader.

Provides live market data streaming and execution via the OANDA v20 REST API.

Supported features:
- Real-time quote tick streaming (bid/ask) via HTTP chunked streaming
- Historical candlestick (bar) data requests
- Market, Limit, Stop, and Market-If-Touched order types
- Order modification and cancellation
- Position and trade management
- Account state synchronization
- CFD instruments (FX, Metals, Indices, Commodities, Bonds)
"""

from nautilus_trader.adapters.oanda.config import OandaDataClientConfig
from nautilus_trader.adapters.oanda.config import OandaExecClientConfig
from nautilus_trader.adapters.oanda.config import OandaInstrumentProviderConfig
from nautilus_trader.adapters.oanda.constants import OANDA_VENUE
from nautilus_trader.adapters.oanda.data import OandaDataClient
from nautilus_trader.adapters.oanda.execution import OandaExecutionClient
from nautilus_trader.adapters.oanda.factories import OandaLiveDataClientFactory
from nautilus_trader.adapters.oanda.factories import OandaLiveExecClientFactory
from nautilus_trader.adapters.oanda.factories import get_cached_oanda_http_client
from nautilus_trader.adapters.oanda.factories import get_cached_oanda_instrument_provider
from nautilus_trader.adapters.oanda.http_client import OandaHttpClient
from nautilus_trader.adapters.oanda.providers import OandaInstrumentProvider


__all__ = [
    "OANDA_VENUE",
    "OandaDataClient",
    "OandaDataClientConfig",
    "OandaExecClientConfig",
    "OandaExecutionClient",
    "OandaHttpClient",
    "OandaInstrumentProvider",
    "OandaInstrumentProviderConfig",
    "OandaLiveDataClientFactory",
    "OandaLiveExecClientFactory",
    "get_cached_oanda_http_client",
    "get_cached_oanda_instrument_provider",
]
