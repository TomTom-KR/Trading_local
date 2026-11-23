from dataclasses import replace
from datetime import UTC, datetime

import pytest

from trading_local.alerts import formatting as fmt
from trading_local.testing import SAMPLE_INSTRUMENTS

INSTRUMENTS = {instrument.symbol: instrument for instrument in SAMPLE_INSTRUMENTS}


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (1234.5, 2, "1.234,50"),
        (1234567.891, 1, "1.234.567,9"),
        (-1.26, 1, "-1,3"),
        (-0.04, 1, "0,0"),  # never '-0,0'
        (float("nan"), 2, "–"),
        (float("inf"), 2, "–"),
    ],
)
def test_format_number_uses_german_separators(value: float, decimals: int, expected: str) -> None:
    assert fmt.format_number(value, decimals) == expected


@pytest.mark.parametrize(
    ("price", "decimals", "text"),
    [
        (97345.12, 0, "97.345"),
        (23456.78, 0, "23.457"),
        (2850.5, 1, "2.850,5"),
        (142.35, 2, "142,35"),
        (1.08453, 4, "1,0845"),
        (0.054321, 5, "0,05432"),
        (0.00012346, 7, "0,0001235"),
    ],
)
def test_price_precision_follows_the_magnitude(price: float, decimals: int, text: str) -> None:
    assert fmt.price_decimals(price) == decimals
    assert fmt.format_price(price) == text


def test_price_decimals_edge_cases() -> None:
    assert fmt.price_decimals(0.0) == 2
    assert fmt.price_decimals(float("nan")) == 2
    assert fmt.price_decimals(1e-12) == 8  # capped
    assert fmt.format_price(None) == "–"
    assert fmt.format_price(142.349, 1) == "142,3"


def test_percent_r_and_ratio() -> None:
    assert fmt.format_percent(58.0, 0) == "58 %"
    assert fmt.format_percent(1.94, 1, signed=True) == "+1,9 %"
    assert fmt.format_percent(-1.0, 1, signed=True) == "-1,0 %"
    assert fmt.format_percent(0.0, 1, signed=True) == "0,0 %"
    assert fmt.format_percent(None) == "–"
    assert fmt.format_r(1.94) == "+1,9 R"
    assert fmt.format_r(-1.0) == "-1,0 R"
    assert fmt.format_r(0.32, 2) == "+0,32 R"
    assert fmt.format_r(-0.01) == "0,0 R"
    assert fmt.format_r(None) == "–"
    assert fmt.format_ratio(2.0) == "2,0 : 1"


def test_datetimes_are_shown_in_the_users_zone_with_daylight_saving() -> None:
    winter = int(datetime(2026, 11, 20, 15, 5, tzinfo=UTC).timestamp())
    summer = int(datetime(2026, 7, 1, 15, 5, tzinfo=UTC).timestamp())
    assert fmt.format_datetime(winter, "Europe/Berlin") == "20.11.2026 16:05"
    assert fmt.format_datetime(summer, "Europe/Berlin") == "01.07.2026 17:05"
    assert fmt.format_datetime(winter, "America/New_York") == "20.11.2026 10:05"
    assert fmt.format_time(winter, "Europe/Berlin") == "16:05"
    assert fmt.format_datetime(None, "Europe/Berlin") == "–"


@pytest.mark.parametrize(
    ("minutes", "text"),
    [(0, "0 Min."), (45, "45 Min."), (60, "1 Std."), (390, "6 Std. 30 Min."), (1440, "1 Tag"),
     (2880, "2 Tage"), (3000, "2 Tage 2 Std."), (-5, "0 Min.")],
)
def test_format_duration(minutes: int, text: str) -> None:
    assert fmt.format_duration(minutes) == text


def test_price_unit_names_the_quoted_unit() -> None:
    assert fmt.price_unit(INSTRUMENTS["NVDA"]) == "USD"
    assert fmt.price_unit(INSTRUMENTS["SHEL.L"]) == "GBp"
    assert fmt.price_unit(replace(INSTRUMENTS["NVDA"], currency="ZAR", price_scale=0.01)) == "ZAR × 0,01"
