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

from enum import Enum


class OandaEnvironment(Enum):
    """OANDA trading environment."""
    PRACTICE = "practice"
    LIVE = "live"


class OandaOrderType(Enum):
    """OANDA v20 order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    MARKET_IF_TOUCHED = "MARKET_IF_TOUCHED"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    GUARANTEED_STOP_LOSS = "GUARANTEED_STOP_LOSS"
    TRAILING_STOP_LOSS = "TRAILING_STOP_LOSS"


class OandaTimeInForce(Enum):
    """OANDA v20 time-in-force options."""
    GTC = "GTC"
    GTD = "GTD"
    GFD = "GFD"
    FOK = "FOK"
    IOC = "IOC"


class OandaOrderState(Enum):
    """OANDA v20 order states."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    TRIGGERED = "TRIGGERED"
    CANCELLED = "CANCELLED"


class OandaTradeState(Enum):
    """OANDA v20 trade states."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CLOSE_WHEN_TRADEABLE = "CLOSE_WHEN_TRADEABLE"
