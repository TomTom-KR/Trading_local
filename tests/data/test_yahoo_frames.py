"""Pure conversions of yfinance frames and quote payloads (shapes as verified by the market-data probe)."""

import numpy as np
import pandas as pd
import pytest

from trading_local.data.yahoo import (
    aliases_from_names,
    clean_name,
    instrument_from_yahoo,
    quote_from_yahoo,
    round_significant,
    split_download,
    to_daily_frame,
    to_minute_frame,
)

T0 = 1_795_186_800  # 2026-11-20 15:00:00 UTC
NAN = np.nan
FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download_frame(group_by: str = "ticker") -> pd.DataFrame:
    """A yf.download(['NVDA', 'SAP.DE'], interval='1m') result: union index in the most common exchange time
    zone (here Europe/Berlin, microsecond unit), NaN rows where a ticker did not trade, float columns."""
    index = pd.DatetimeIndex(pd.to_datetime([T0, T0 + 60, T0 + 120, T0 + 180], unit="s", utc=True)).tz_convert(
        "Europe/Berlin"
    )
    nvda = [
        [NAN] * 6,
        [180.0, 181.0, 179.5, 180.5, 180.5, 5_000.0],
        [180.5, 182.0, 180.0, 181.5, 181.5, NAN],
        [181.5, 181.5, 181.5, 181.5, 181.5, 0.0],  # still-forming minute: flat snapshot
    ]
    sap = [
        [200.0, 201.0, 199.0, 200.5, 200.5, 1_000.0],
        [200.5, 200.8, 200.1, 200.2, 200.2, 800.0],
        [NAN] * 6,
        [NAN] * 6,
    ]
    blocks = {"NVDA": pd.DataFrame(nvda, index=index, columns=FIELDS), "SAP.DE": pd.DataFrame(sap, index=index, columns=FIELDS)}
    frame = pd.concat(blocks.values(), axis=1, keys=blocks.keys(), names=["Ticker", "Price"])
    frame.index.name = "Datetime"
    if group_by == "column":
        frame.columns = frame.columns.swaplevel(0, 1)
        frame = frame.sort_index(level=0, axis=1)
    return frame


@pytest.mark.parametrize("group_by", ["ticker", "column"])
def test_multi_symbol_download_is_split_and_converted(group_by: str) -> None:
    parts = split_download(download_frame(group_by), ["NVDA", "SAP.DE"])
    assert set(parts) == {"NVDA", "SAP.DE"}
    nvda = to_minute_frame(parts["NVDA"])
    assert nvda.index.name == "ts" and nvda.index.dtype == np.int64
    assert nvda.index.tolist() == [T0 + 60, T0 + 120, T0 + 180]
    assert list(nvda.columns) == ["open", "high", "low", "close", "volume"]
    assert all(dtype == np.float64 for dtype in nvda.dtypes)
    assert nvda.loc[T0 + 120, "volume"] == 0.0  # NaN volume becomes 0
    sap = to_minute_frame(parts["SAP.DE"])
    assert sap.index.tolist() == [T0, T0 + 60]
    assert sap.loc[T0, "close"] == 200.5


def test_the_forming_bar_is_dropped_with_before_ts() -> None:
    nvda = to_minute_frame(split_download(download_frame(), ["NVDA", "SAP.DE"])["NVDA"], before_ts=T0 + 180)
    assert nvda.index.tolist() == [T0 + 60, T0 + 120]


def test_single_ticker_with_flat_columns_and_unaligned_rows() -> None:
    index = pd.DatetimeIndex(pd.to_datetime([T0, T0 + 60, T0 + 112], unit="s", utc=True)).tz_convert("America/New_York")
    flat = pd.DataFrame(
        [[1.0, 2.0, 0.5, 1.5, 1.5, 10.0], [1.5, 1.6, 1.4, 1.5, 1.5, 0.0], [1.5, 1.5, 1.5, 1.5, 1.5, 0.0]],
        index=index,
        columns=FIELDS,
    )
    parts = split_download(flat, ["BTC-USD"])
    frame = to_minute_frame(parts["BTC-USD"])
    assert frame.index.tolist() == [T0, T0 + 60]  # the unaligned live row is no bar
    assert split_download(flat, ["A", "B"]) == {}


def test_rows_with_invalid_prices_are_dropped_and_nothing_is_invented() -> None:
    index = pd.DatetimeIndex(pd.to_datetime([T0, T0 + 60, T0 + 120, T0 + 300], unit="s", utc=True))
    data = pd.DataFrame(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 5.0], [0.0, 1.0, 1.0, 1.0, 1.0, 5.0], [1.0, NAN, 1.0, 1.0, 1.0, 5.0],
         [2.0, 2.0, 2.0, 2.0, 2.0, -3.0]],
        index=index,
        columns=FIELDS,
    )  # fmt: skip
    frame = to_minute_frame(data)
    assert frame.index.tolist() == [T0, T0 + 300]  # no rows for the gap
    assert frame.loc[T0 + 300, "volume"] == 0.0


def test_32_bit_chart_prices_are_restored() -> None:
    assert round_significant(np.array([60_740.66015625, 1.0974539518356323, 0.000123456789])).tolist() == [
        60_740.66, 1.097454, 0.0001234568,
    ]  # fmt: skip
    index = pd.DatetimeIndex(pd.to_datetime([T0], unit="s", utc=True))
    data = pd.DataFrame([[60_740.66015625, 60_741.0, 60_700.0, 60_740.66015625, 0.0, 12.5]], index=index, columns=FIELDS)
    frame = to_minute_frame(data)
    assert frame.loc[T0, "open"] == 60_740.66 and frame.loc[T0, "volume"] == 12.5


def test_empty_and_missing_inputs() -> None:
    assert split_download(None, ["X"]) == {}
    assert split_download(pd.DataFrame(), ["X"]) == {}
    assert to_minute_frame(pd.DataFrame()).empty
    assert to_daily_frame(pd.DataFrame()).empty


def test_daily_frames_from_naive_and_aware_indexes() -> None:
    naive = pd.DatetimeIndex(["2026-11-19", "2026-11-20", "2026-11-21", "2026-11-22"]).as_unit("s")
    data = pd.DataFrame(
        [[10.0, 11.0, 9.0, 10.5, 10.5, 100.0], [10.5, 11.5, 10.0, 11.0, 11.0, 120.0], [NAN] * 6, [NAN] * 6],
        index=naive,
        columns=FIELDS,
    )
    frame = to_daily_frame(data)
    assert frame.index.name == "day" and frame.index.tolist() == ["2026-11-19", "2026-11-20"]
    assert frame.loc["2026-11-20", "close"] == 11.0
    aware = pd.DatetimeIndex(["2026-11-19 00:00", "2026-11-20 00:00"], tz="Asia/Tokyo").as_unit("s")
    tokyo = to_daily_frame(pd.DataFrame(data.iloc[:2].to_numpy(), index=aware, columns=FIELDS))
    assert tokyo.index.tolist() == ["2026-11-19", "2026-11-20"]  # the venue's session date, not the UTC date


def test_quote_from_yahoo() -> None:
    raw = {
        "symbol": "^GDAXI", "regularMarketPrice": 23_450.5, "regularMarketTime": T0, "regularMarketVolume": 0,
        "regularMarketPreviousClose": 23_300.0, "regularMarketOpen": 23_310.0, "regularMarketDayHigh": 23_500.0,
        "regularMarketDayLow": 23_280.0, "regularMarketChangePercent": 0.64, "bid": 0.0, "ask": 0.0,
        "marketState": "REGULAR",
    }  # fmt: skip
    quote = quote_from_yahoo(raw, received_ts=T0 + 5)
    assert quote is not None
    assert (quote.symbol, quote.ts, quote.received_ts, quote.price) == ("^GDAXI", T0, T0 + 5, 23_450.5)
    assert quote.bid is None and quote.ask is None  # indices report 0.0
    assert quote.day_volume == 0.0 and quote.change_pct == 0.64 and quote.market_state == "REGULAR"
    assert quote.source == "quote"
    assert quote_from_yahoo({"symbol": "BRK.B", "quoteType": "EQUITY"}, T0) is None  # stub without price


PENCE_QUOTE = {
    "symbol": "SHEL.L", "quoteType": "EQUITY", "currency": "GBp", "exchange": "LSE", "regularMarketPrice": 3574.5,
    "exchangeTimezoneName": "Europe/London", "exchangeDataDelayedBy": 15, "longName": "Shell plc",
    "shortName": "SHELL PLC ORD EUR0.07          ",
}  # fmt: skip


def test_instrument_from_a_pence_quote() -> None:
    instrument = instrument_from_yahoo(PENCE_QUOTE)
    assert instrument is not None
    assert (instrument.currency, instrument.price_scale) == ("GBP", 0.01)
    assert (instrument.exchange, instrument.region, instrument.timezone) == ("XLON", "EU", "Europe/London")
    assert instrument.asset_class == "equity" and instrument.tradable
    assert instrument.delay_min == 15 and instrument.benchmark == "^STOXX50E"
    assert instrument.name == "Shell plc"
    assert "Shell" in instrument.aliases and "Shell plc" in instrument.aliases


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"symbol": "BTC-USD", "quoteType": "CRYPTOCURRENCY", "currency": "USD", "exchange": "CCC",
          "regularMarketPrice": 61_000.0, "longName": "Bitcoin USD"},
         ("crypto", "CRYPTO", "GLOBAL", "UTC", True, None)),
        ({"symbol": "^GDAXI", "quoteType": "INDEX", "currency": "EUR", "exchange": "GER",
          "regularMarketPrice": 23_000.0, "shortName": "DAX P"},
         ("index", "XETR", "EU", "Europe/Berlin", True, None)),
        ({"symbol": "^FCHI", "quoteType": "INDEX", "currency": "EUR", "exchange": "PAR", "regularMarketPrice": 8000.0},
         ("index", "XPAR", "EU", "Europe/Paris", False, None)),
        ({"symbol": "EURUSD=X", "quoteType": "CURRENCY", "currency": "USD", "exchange": "CCY",
          "regularMarketPrice": 1.05},
         ("fx", "FX", "GLOBAL", "UTC", False, None)),
        ({"symbol": "GC=F", "quoteType": "FUTURE", "currency": "USD", "exchange": "CMX", "regularMarketPrice": 2600.0},
         ("future", "CME", "US", "America/Chicago", False, None)),
        ({"symbol": "7203.T", "quoteType": "EQUITY", "currency": "JPY", "exchange": "JPX",
          "regularMarketPrice": 2900.0, "exchangeDataDelayedBy": 20, "longName": "Toyota Motor Corporation"},
         ("equity", "XTKS", "ASIA", "Asia/Tokyo", True, "^N225")),
    ],
)  # fmt: skip
def test_instrument_classification(raw: dict[str, object], expected: tuple[object, ...]) -> None:
    instrument = instrument_from_yahoo(raw)
    assert instrument is not None
    actual = (
        instrument.asset_class, instrument.exchange, instrument.region, instrument.timezone,
        instrument.tradable, instrument.benchmark,
    )  # fmt: skip
    assert actual == expected


@pytest.mark.parametrize(
    "change",
    [
        {"exchange": "PNK"},  # OTC: no calendar
        {"quoteType": "MUTUALFUND"},
        {"regularMarketPrice": None},  # stub
        {"currency": None},
    ],
)
def test_unsupported_lookups_give_none(change: dict[str, object]) -> None:
    assert instrument_from_yahoo({**PENCE_QUOTE, **change}) is None


def test_names_and_aliases() -> None:
    assert clean_name("SAP SE                        I") == "SAP SE"
    assert aliases_from_names("Toyota Motor Corporation", "TOYOTA MOTOR CORP") == (
        "Toyota Motor Corporation", "Toyota Motor", "TOYOTA MOTOR CORP", "TOYOTA MOTOR",
    )  # fmt: skip
    assert aliases_from_names("Apple Inc.", None) == ("Apple Inc.", "Apple")
    assert aliases_from_names("Novo Nordisk A/S", "") == ("Novo Nordisk A/S", "Novo Nordisk")
