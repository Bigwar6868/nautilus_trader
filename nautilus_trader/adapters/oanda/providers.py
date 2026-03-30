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
OANDA instrument provider.
"""

from __future__ import annotations

import time

from nautilus_trader.adapters.oanda.http_client import OandaHttpClient
from nautilus_trader.adapters.oanda.parsing import parse_instrument
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.model.identifiers import InstrumentId


class OandaInstrumentProvider(InstrumentProvider):
    """
    Provides instruments from the OANDA venue.

    Fetches the full instrument list from the OANDA v20 API and converts
    them into NautilusTrader ``Cfd`` instrument objects.

    Parameters
    ----------
    client : OandaHttpClient
        The OANDA HTTP client.
    config : InstrumentProviderConfig, optional
        The instrument provider configuration.

    """

    def __init__(
        self,
        client: OandaHttpClient,
        config: InstrumentProviderConfig | None = None,
    ) -> None:
        super().__init__(config=config)
        self._client = client

    async def load_all_async(
        self,
        filters: dict | None = None,
    ) -> None:
        """
        Load all OANDA instruments into the provider.

        Parameters
        ----------
        filters : dict, optional
            Not used for OANDA (reserved for future use).

        """
        ts_init = int(time.time() * 1e9)

        response = await self._client.get_account_instruments()
        instruments_data = response.get("instruments", [])

        for inst_data in instruments_data:
            try:
                instrument = parse_instrument(inst_data, ts_init)
                self._instruments[instrument.id] = instrument
                if instrument.base_currency:
                    self._currencies[instrument.base_currency.code] = instrument.base_currency
                self._currencies[instrument.quote_currency.code] = instrument.quote_currency
            except Exception as e:
                self._log.warning(
                    f"Failed to parse instrument {inst_data.get('name', '?')}: {e}",
                )

        self._log.info(f"Loaded {len(self._instruments)} instruments from OANDA")

    async def load_ids_async(
        self,
        instrument_ids: list[InstrumentId],
        filters: dict | None = None,
    ) -> None:
        """
        Load specific instruments by ID.

        OANDA supports fetching specific instruments, so we use that
        rather than loading all and filtering.
        """
        from nautilus_trader.adapters.oanda.parsing import instrument_id_to_oanda_symbol

        if not instrument_ids:
            return

        oanda_symbols = [
            instrument_id_to_oanda_symbol(iid) for iid in instrument_ids
        ]

        ts_init = int(time.time() * 1e9)

        response = await self._client.get_account_instruments(instruments=oanda_symbols)
        instruments_data = response.get("instruments", [])

        for inst_data in instruments_data:
            try:
                instrument = parse_instrument(inst_data, ts_init)
                self._instruments[instrument.id] = instrument
                if instrument.base_currency:
                    self._currencies[instrument.base_currency.code] = instrument.base_currency
                self._currencies[instrument.quote_currency.code] = instrument.quote_currency
            except Exception as e:
                self._log.warning(
                    f"Failed to parse instrument {inst_data.get('name', '?')}: {e}",
                )
