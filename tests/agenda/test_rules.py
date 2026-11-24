"""Agenda rules: published values, surprises, matching, affected instruments and the entry block."""

from dataclasses import replace

import pytest

from trading_local.agenda import rules
from trading_local.agenda.rules import Value, parse_value
from trading_local.db.repo_agenda import AgendaEvent
from trading_local.models import Instrument
from trading_local.testing import SAMPLE_INSTRUMENTS

NVDA, SAP, SHEL, TOYOTA, BTC = SAMPLE_INSTRUMENTS[:5]
EURUSD = SAMPLE_INSTRUMENTS[7]
TENCENT = Instrument("0700.HK", "Tencent", "equity", "ASIA", "XHKG", "HKD", timezone="Asia/Hong_Kong")
T0 = 1_795_190_400  # 2026-11-20 16:00 UTC


def event(kind: str = "macro", *, currency: str = "USD", symbol: str | None = None, at: int = T0,
          importance: int = 3, time_known: bool = True) -> AgendaEvent:  # fmt: skip
    return AgendaEvent(
        uid=f"{kind}:{currency}:{symbol}:{at}", kind=kind, source="test", title="CPI m/m", event_ts=at,
        first_seen_ts=0, updated_ts=0, currency=currency, symbol=symbol, importance=importance,
        time_known=time_known,
    )  # fmt: skip


@pytest.mark.parametrize(
    ("text", "number", "unit", "step"),
    [
        ("0.3%", 0.3, "%", 0.1),
        ("-2", -2.0, "", 1.0),
        ("103.3", 103.3, "", 0.1),
        ("21k", 21_000.0, "", 1000.0),
        ("23.75K", 23_750.0, "", 10.0),
        ("1.2B", 1.2e9, "", 1e8),
        ("<0.10%", 0.10, "%", 0.01),
        ("1,234.5", 1234.5, "", 0.1),
    ],
)
def test_parse_value(text: str, number: float, unit: str, step: float) -> None:
    value = parse_value(text)
    assert value is not None
    assert value.number == pytest.approx(number) and value.unit == unit and value.step == pytest.approx(step)


@pytest.mark.parametrize("text", ["", "-", None, "n/a", "4.5%%", "12 days"])
def test_parse_value_rejects_text(text: str | None) -> None:
    assert parse_value(text) is None


def test_surprise_and_label() -> None:
    actual, forecast = parse_value("0.4%"), parse_value("0.3%")
    assert rules.surprise(actual, forecast) == pytest.approx(0.1)
    assert rules.surprise_label(actual, forecast) == "above"
    assert rules.surprise_label(parse_value("2.56%"), parse_value("2.71%")) == "below"
    assert rules.surprise_label(parse_value("0.30%"), parse_value("0.3%")) == "inline"  # equal at 1 decimal
    assert rules.surprise_label(parse_value("0.34%"), parse_value("0.3%")) == "inline"
    assert rules.surprise_label(parse_value("0.36%"), parse_value("0.3%")) == "above"
    assert rules.surprise(parse_value("21k"), parse_value("0.3%")) is None  # different units
    assert rules.surprise_label(actual, None) == ""
    assert rules.earnings_label(5.8) == "above" and rules.earnings_label(-1.5) == "inline"
    assert rules.earnings_label(-9.3) == "below" and rules.earnings_label(None) == ""


def test_standardised_surprise_needs_history() -> None:
    assert rules.standardised(0.2, [0.1, -0.1, 0.1, -0.1, 0.1]) is None  # five earlier surprises: too few
    assert rules.standardised(0.2, [0.1, -0.1, 0.1, -0.1, 0.1, -0.1]) == pytest.approx(2.0)
    assert rules.standardised(0.2, [0.0] * 8) is None and rules.standardised(None, [0.1] * 8) is None


def test_central_bank_titles() -> None:
    for title in ("Federal Funds Rate", "Main Refinancing Rate", "BOJ Policy Rate", "FOMC Press Conference",
                  "Fed Interest Rate Decision", "Official Bank Rate", "Cash Rate"):  # fmt: skip
        assert rules.is_central_bank(title), title
    for title in ("CPI m/m", "Core PCE Price Index m/m", "Unemployment Rate", "Prelim GDP q/q"):
        assert not rules.is_central_bank(title), title


def test_matching_titles_and_values() -> None:
    assert rules.title_tokens("US CPI MoM") == rules.title_tokens("CPI m/m") == {"cpi", "mm"}
    assert rules.title_similarity("CB Consumer Confidence", "US CB Consumer Confidence") == 1.0
    assert rules.title_similarity("CPI m/m", "CPI y/y") == pytest.approx(1 / 3)
    # headline abbreviations and country words (live 2026-11-24: 'Japanese Mfg PMI Flash', 'German IFO ...')
    assert rules.title_tokens("Mfg PMI Flash") == rules.title_tokens("Flash Manufacturing PMI")
    assert rules.title_similarity("IFO Business Climate", "German ifo Business Climate") == 1.0
    forecast, previous = parse_value("-2"), parse_value("-2")
    # weak title similarity, but the same forecast and previous value
    assert rules.match_score("Richmond Fed Comp. Index", forecast, previous, "Richmond Manufacturing Index",
                             parse_value("-2"), parse_value("-2")) > 0  # fmt: skip
    assert rules.match_score("CPI y/y", parse_value("2.9%"), None, "CPI m/m", parse_value("0.3%"), None) == 0.0
    assert rules.same_value(Value(0.3, "%", 0.1), Value(0.3, "%", 0.01))
    assert not rules.same_value(Value(0.3, "%", 0.1), Value(0.3, "", 0.1))


def test_which_instruments_an_event_concerns() -> None:
    usd, eur, cny = event(currency="USD"), event(currency="EUR"), event(currency="CNY")
    assert rules.affects(usd, NVDA) and not rules.affects(usd, SAP)
    assert rules.affects(eur, SAP) and not rules.affects(eur, SHEL)
    assert rules.affects(usd, BTC) and not rules.affects(eur, replace(BTC, currency="EUR"))  # crypto: USD only
    assert rules.affects(usd, EURUSD) and rules.affects(eur, EURUSD) and not rules.affects(cny, EURUSD)
    assert rules.affects(cny, TENCENT)  # Hong Kong trades China
    earnings = event("earnings", currency="", symbol="NVDA")
    assert rules.affects(earnings, NVDA) and not rules.affects(earnings, SAP)
    assert not rules.affects(event(currency=""), NVDA)


def test_macro_events_block_shortly_before_the_release() -> None:
    before = 30 * 60
    cpi = event(at=T0)
    kwargs = {"planned_exit_ts": T0 + 4 * 3600, "before_s": before}
    assert rules.blocking_event([cpi], NVDA, t_issue=T0 - before, **kwargs) is cpi
    assert rules.blocking_event([cpi], NVDA, t_issue=T0, **kwargs) is cpi
    assert rules.blocking_event([cpi], NVDA, t_issue=T0 - before - 1, **kwargs) is None  # earlier: allowed
    assert rules.blocking_event([cpi], NVDA, t_issue=T0 + 1, **kwargs) is None  # released: allowed
    assert rules.blocking_event([cpi], SAP, t_issue=T0 - 60, **kwargs) is None  # not its currency
    for other in (replace(cpi, importance=2), replace(cpi, time_known=False)):
        assert rules.blocking_event([other], NVDA, t_issue=T0 - 60, **kwargs) is None
    assert rules.blocking_event([cpi], NVDA, t_issue=T0 - 60, planned_exit_ts=T0 + 3600, before_s=0) is None


def test_own_earnings_block_until_the_planned_exit() -> None:
    release = event("earnings", currency="", symbol="NVDA", at=T0 + 3 * 3600)
    assert rules.blocking_event([release], NVDA, t_issue=T0, planned_exit_ts=T0 + 4 * 3600, before_s=1800) is release
    # an intraday trade flat before a release after the close
    assert rules.blocking_event([release], NVDA, t_issue=T0, planned_exit_ts=T0 + 2 * 3600, before_s=1800) is None
    assert rules.blocking_event([release], SAP, t_issue=T0, planned_exit_ts=T0 + 4 * 3600, before_s=1800) is None
    # a release without known time covers its day
    day_start = T0 - 16 * 3600
    untimed = event("earnings", currency="", symbol="NVDA", at=day_start, time_known=False)
    assert rules.blocking_event([untimed], NVDA, t_issue=T0, planned_exit_ts=T0 + 60, before_s=0) is untimed
    assert rules.blocking_event([untimed], NVDA, t_issue=day_start + 86_400, planned_exit_ts=T0 + 86_460,
                                before_s=0) is None  # fmt: skip
    first = rules.blocking_event([release, event(at=T0 + 600)], NVDA, t_issue=T0, planned_exit_ts=T0 + 4 * 3600,
                                 before_s=1800)  # fmt: skip
    assert first is not None and first.kind == "macro"  # the earliest blocking event
