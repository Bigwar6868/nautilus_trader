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

from nautilus_trader.model.identifiers import Venue


OANDA_VENUE = Venue("OANDA")

# OANDA v20 REST API base URLs
OANDA_REST_PRACTICE = "https://api-fxpractice.oanda.com"
OANDA_REST_LIVE = "https://api-fxtrade.oanda.com"

# OANDA v20 Streaming API base URLs
OANDA_STREAM_PRACTICE = "https://stream-fxpractice.oanda.com"
OANDA_STREAM_LIVE = "https://stream-fxtrade.oanda.com"

# API version prefix
OANDA_API_VERSION = "/v3"

# OANDA granularity mappings (bar aggregation -> OANDA string)
OANDA_GRANULARITIES = {
    5: "S5",
    10: "S10",
    15: "S15",
    30: "S30",
    60: "M1",
    120: "M2",
    240: "M4",
    300: "M5",
    600: "M10",
    900: "M15",
    1800: "M30",
    3600: "H1",
    7200: "H2",
    10800: "H3",
    14400: "H4",
    21600: "H6",
    28800: "H8",
    43200: "H12",
    86400: "D",
    604800: "W",
}

# Maximum candles per request
OANDA_MAX_CANDLES = 5000

# Asset class mappings for OANDA instruments
OANDA_FX_PAIRS = {
    "AUD", "CAD", "CHF", "CNH", "CZK", "DKK", "EUR", "GBP", "HKD", "HUF",
    "INR", "JPY", "MXN", "NOK", "NZD", "PLN", "SAR", "SEK", "SGD", "THB",
    "TRY", "TWD", "USD", "ZAR",
}

OANDA_METALS = {"XAU", "XAG", "XPT", "XPD", "XCU"}

OANDA_INDICES = {
    "AU200", "CN50", "DE10YB", "DE30", "EU50", "FR40", "HK33",
    "IN50", "JP225", "NAS100", "NL25", "SG30", "TWIX",
    "UK100", "UK10YB", "US2000", "US30", "USB02Y", "USB05Y",
    "USB10Y", "USB30Y", "SPX500",
}

OANDA_COMMODITIES = {
    "BCO", "CORN", "NATGAS", "SOYBN", "SUGAR", "WHEAT", "WTICO",
}
