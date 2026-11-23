import dataclasses
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from trading_local import presenters
from trading_local.dto import NewsImpactOut, QuoteOut, SignalBrief, SignalDetailOut, SignalOut
from trading_local.models import ChartAnalysis, Instrument, NewsAnalysis, NewsItem, Quote, SessionState, Signal

T0 = 1_800_000_000

NVDA = Instrument(
    "NVDA", "NVIDIA", "equity", "US", "XNAS", "USD", timezone="America/New_York", benchmark="^GSPC", sort_order=1
)
SHELL = Instrument(
    "SHEL.L",
    "Shell",
    "equity",
    "EU",
    "XLON",
    "GBP",
    price_scale=0.01,
    timezone="Europe/London",
    delay_min=15,
    tradable=False,
    active=False,
    aliases=("Shell", "Royal Dutch Shell"),
    sort_order=3,
)


def make_quote(price: float = 104.0, **fields: Any) -> Quote:
    return Quote(symbol="NVDA", ts=T0, received_ts=T0 + 1, price=price, **fields)


def make_signal(**fields: Any) -> Signal:
    values: dict[str, Any] = {
        "created_ts": T0,
        "data_ts": T0,
        "symbol": "NVDA",
        "direction": "LONG",
        "setup": "breakout",
        "timeframe": "15m",
        "horizon_min": 240,
        "ref_price": 100.0,
        "atr": 1.5,
        "stop_dist": 2.0,
        "target_dist": 3.0,
        "score": 0.66,
        "confidence": 0.58,
        "engine_version": "v1",
        "id": 7,
    }
    values.update(fields)
    return Signal(**values)


def make_item(**fields: Any) -> NewsItem:
    values: dict[str, Any] = {
        "uid": "u1",
        "title_key": "k1",
        "source": "reuters-biz",
        "url": "https://news.example/1",
        "title": "Nvidia hebt die Prognose an",
        "fetched_ts": T0,
        "id": 11,
    }
    values.update(fields)
    return NewsItem(**values)


# ------------------------------------------------------------------------------------------------
# Labels, time, mail addresses
# ------------------------------------------------------------------------------------------------
def test_setup_labels() -> None:
    assert presenters.SETUP_LABELS_DE == {
        "trend_pullback": "Trend-Rücksetzer",
        "breakout": "Ausbruch",
        "mean_reversion": "Gegenbewegung",
        "news_momentum": "Nachrichten-Impuls",
        "daily_trend": "Tagestrend",
    }
    assert presenters.setup_label_de("mean_reversion") == "Gegenbewegung"
    assert presenters.setup_label_de("gap_fade") == "gap_fade"  # unknown: the raw name


def test_to_ts() -> None:
    berlin_evening = datetime(2026, 11, 20, 16, 0, tzinfo=timezone(timedelta(hours=1)))

    assert presenters.to_ts(None) is None
    assert presenters.to_ts(datetime(2026, 11, 20, 15, 0, 0, 999_000, tzinfo=UTC)) == 1_795_186_800
    assert presenters.to_ts(berlin_evening) == 1_795_186_800
    with pytest.raises(ValueError, match="naive"):
        presenters.to_ts(datetime(2026, 11, 20, 15, 0))  # noqa: DTZ001


@pytest.mark.parametrize(
    ("address", "masked"),
    [
        ("max@example.org", "m***@example.org"),
        ("  Max.Mustermann+boerse@mail.example.org ", "M***@mail.example.org"),
        ("m@example.org", "m***@example.org"),
        ("Max Mustermann <max@example.org>", "m***@example.org"),
        ('"Mustermann, Max" <max@example.org>', "m***@example.org"),
        ("max@example.org, erika@example.com", "m***@example.org, e***@example.com"),
        ("max@example.org; erika@example.com", "m***@example.org, e***@example.com"),
        ("@example.org", "***@example.org"),
        ("", ""),
        ("   ", ""),
        ("kein-postfach", "***"),
        ("max@", "***"),
        ("max@example.org@example.com", "***"),
    ],
)
def test_mask_email(address: str, masked: str) -> None:
    assert presenters.mask_email(address) == masked


@pytest.mark.parametrize(
    "address", ["max.mustermann@example.org", "Max Mustermann <max.mustermann@example.org>", "mustermann"]
)
def test_mask_email_never_shows_the_local_part(address: str) -> None:
    assert "mustermann" not in presenters.mask_email(address).lower()


# ------------------------------------------------------------------------------------------------
# Market data
# ------------------------------------------------------------------------------------------------
def test_quote_out_copies_all_fields() -> None:
    quote = Quote(
        symbol="NVDA",
        ts=T0,
        received_ts=T0 + 2,
        price=104.5,
        bid=104.4,
        ask=104.6,
        prev_close=101.0,
        day_open=101.5,
        day_high=105.0,
        day_low=101.2,
        day_volume=1_250_000.0,
        change_pct=3.47,
        market_state="REGULAR",
        source="stream",
    )

    out = presenters.quote_out(quote)

    expected = dataclasses.asdict(quote)
    del expected["received_ts"]  # internal: not part of the API model
    assert out == QuoteOut(**expected)
    assert presenters.quote_out(make_quote(99.0)) == QuoteOut(symbol="NVDA", price=99.0, ts=T0)


def test_instrument_out_without_context_is_closed() -> None:
    out = presenters.instrument_out(SHELL)

    assert out.model_dump() == {
        "symbol": "SHEL.L",
        "name": "Shell",
        "asset_class": "equity",
        "region": "EU",
        "exchange": "XLON",
        "exchange_name": "",
        "currency": "GBP",
        "price_scale": 0.01,
        "timezone": "Europe/London",
        "delay_min": 15,
        "tradable": False,
        "active": False,
        "benchmark": None,
        "sort_order": 3,
        "phase": "closed",
        "is_open": False,
        "quote": None,
        "open_signal": None,
    }


def test_instrument_out_with_session_quote_and_open_signal() -> None:
    state = SessionState(
        exchange="XNAS",
        phase="open",
        is_open=True,
        since=datetime(2026, 11, 20, 14, 30, tzinfo=UTC),
        until=datetime(2026, 11, 20, 21, 0, tzinfo=UTC),
        next_open=datetime(2026, 11, 23, 14, 30, tzinfo=UTC),
        next_close=datetime(2026, 11, 20, 21, 0, tzinfo=UTC),
    )
    signal = make_signal(status="active", actionable=True, created_ts=T0 - 600)

    out = presenters.instrument_out(
        NVDA, quote=make_quote(104.0), state=state, open_signal=signal, exchange_name="Nasdaq"
    )

    assert (out.symbol, out.exchange, out.exchange_name, out.benchmark) == ("NVDA", "XNAS", "Nasdaq", "^GSPC")
    assert (out.phase, out.is_open) == ("open", True)
    assert out.quote == QuoteOut(symbol="NVDA", price=104.0, ts=T0)
    assert out.open_signal == SignalBrief(id=7, direction="LONG", status="active", confidence=0.58, created_ts=T0 - 600)


def test_instrument_out_shows_a_lunch_break_as_not_open() -> None:
    state = SessionState("XTKS", "break", False, None, None, None, None, note="Mittagspause")

    out = presenters.instrument_out(NVDA, state=state)

    assert (out.phase, out.is_open) == ("break", False)


def test_instrument_out_needs_a_stored_signal() -> None:
    with pytest.raises(ValueError, match="signal has no id"):
        presenters.instrument_out(NVDA, open_signal=make_signal(id=None))


# ------------------------------------------------------------------------------------------------
# News
# ------------------------------------------------------------------------------------------------
def test_news_out_without_analysis() -> None:
    item = make_item(
        summary="Der Chiphersteller erwartet mehr Umsatz.",
        lang="de",
        published_ts=T0 - 300,
        hinted_symbols=["NVDA"],
        status="queued",
        status_reason="internal",
        duplicate_of=None,
    )

    out = presenters.news_out(item, None, source_names={"reuters-biz": "Reuters Business", "other": "Other"})

    assert out.model_dump() == {
        "id": 11,
        "source": "reuters-biz",
        "source_name": "Reuters Business",
        "url": "https://news.example/1",
        "title": "Nvidia hebt die Prognose an",
        "summary": "Der Chiphersteller erwartet mehr Umsatz.",
        "lang": "de",
        "published_ts": T0 - 300,
        "fetched_ts": T0,
        "status": "queued",
        "hinted_symbols": ["NVDA"],
        "analysis": None,
    }


@pytest.mark.parametrize("source_names", [None, {}, {"another-feed": "Another Feed"}])
def test_news_out_falls_back_to_the_feed_id(source_names: dict[str, str] | None) -> None:
    assert presenters.news_out(make_item(), None, source_names=source_names).source_name == "reuters-biz"


def test_news_out_orders_the_impacts_by_strength() -> None:
    analysis = NewsAnalysis(
        news_id=11,
        analyzed_ts=T0 + 45,
        model="test-model",
        relevance=4,
        sentiment=0.3,
        category="guidance",
        horizon="days",
        summary_de="Prognose angehoben",
        confidence=0.8,
        symbols={"TSM": 0.2, "NVDA": 0.7, "INTC": -0.9, "AMD": 0.2, "SAP.DE": 0.0},
        latency_ms=2300,
    )

    out = presenters.news_out(make_item(status="analyzed"), analysis)

    assert out.analysis is not None
    assert out.analysis.symbols == [
        NewsImpactOut(symbol="INTC", sentiment=-0.9),  # strongest first, whatever the direction
        NewsImpactOut(symbol="NVDA", sentiment=0.7),
        NewsImpactOut(symbol="AMD", sentiment=0.2),  # equal strength: alphabetical
        NewsImpactOut(symbol="TSM", sentiment=0.2),
        NewsImpactOut(symbol="SAP.DE", sentiment=0.0),
    ]
    assert out.analysis.model_dump(exclude={"symbols"}) == {
        "analyzed_ts": T0 + 45,
        "relevance": 4,
        "sentiment": 0.3,
        "confidence": 0.8,
        "category": "guidance",
        "horizon": "days",
        "summary_de": "Prognose angehoben",
    }
    assert list(analysis.symbols) == ["TSM", "NVDA", "INTC", "AMD", "SAP.DE"]  # the input is left alone


def test_news_out_needs_a_stored_item() -> None:
    with pytest.raises(ValueError, match="news item has no id"):
        presenters.news_out(make_item(id=None), None)


# ------------------------------------------------------------------------------------------------
# Signals
# ------------------------------------------------------------------------------------------------
def full_signal() -> Signal:
    return make_signal(
        status="active",
        intraday=False,
        components={"trend": 0.8, "volume": 0.4},
        features={"rsi": 61.5, "ema20": 101.2, "note": "Rücksetzer"},
        expected_r=0.21,
        issued_ts=T0 + 30,
        llm_verdict="confirm",
        llm_conviction=0.7,
        llm_latency_ms=8123,
        rationale_de="Ausbruch über das Vortageshoch",
        risks_de="Zahlen am Abend",
        news_ids=[11],
        actionable=True,
        shadow_reason="",
        status_reason="Einstieg erfolgt",
        entry_deadline_ts=T0 + 900,
        checked_ts=T0 + 540,
        entry_ts=T0 + 60,
        entry_price=100.2,
        stop_price=98.2,
        target_price=103.2,
        expires_ts=T0 + 14_460,
        mfe_r=0.9,
        mae_r=-0.2,
        mailed_ts=T0 + 31,
    )


def test_signal_out_maps_the_signal_and_adds_the_instrument() -> None:
    signal = full_signal()

    out = presenters.signal_out(signal, SHELL)

    assert type(out) is SignalOut
    assert out.model_dump() == {
        "id": 7,
        "created_ts": T0,
        "issued_ts": T0 + 30,
        "data_ts": T0,
        "symbol": "NVDA",
        "name": "Shell",
        "region": "EU",
        "currency": "GBP",
        "delay_min": 15,
        "direction": "LONG",
        "setup": "breakout",
        "setup_label_de": "Ausbruch",
        "timeframe": "15m",
        "intraday": False,
        "horizon_min": 240,
        "ref_price": 100.0,
        "atr": 1.5,
        "stop_dist": 2.0,
        "target_dist": 3.0,
        "rr": 1.5,
        "score": 0.66,
        "confidence": 0.58,
        "expected_r": 0.21,
        "llm_verdict": "confirm",
        "llm_conviction": 0.7,
        "rationale_de": "Ausbruch über das Vortageshoch",
        "risks_de": "Zahlen am Abend",
        "actionable": True,
        "shadow_reason": "",
        "status": "active",
        "status_reason": "Einstieg erfolgt",
        "entry_ts": T0 + 60,
        "entry_price": 100.2,
        "stop_price": 98.2,
        "target_price": 103.2,
        "expires_ts": T0 + 14_460,
        "exit_ts": None,
        "exit_price": None,
        "outcome_r": None,
        "mfe_r": 0.9,
        "mae_r": -0.2,
        "mailed_ts": T0 + 31,
        "engine_version": "v1",
        "last_price": None,
        "open_r": None,
    }


def test_signal_out_without_instrument_keeps_the_neutral_defaults() -> None:
    out = presenters.signal_out(make_signal(setup="gap_fade"), None)

    assert (out.name, out.region, out.currency, out.delay_min) == ("", "GLOBAL", "", 0)
    assert out.setup_label_de == "gap_fade"
    assert (out.last_price, out.open_r) == (None, None)


@pytest.mark.parametrize(
    ("direction", "last_price", "expected_r"),
    [
        ("LONG", 104.0, 2.0),  # (104 - 100) / 2
        ("LONG", 99.0, -0.5),
        ("LONG", 100.0, 0.0),
        ("SHORT", 104.0, -2.0),  # a rising price is a loss for a short
        ("SHORT", 97.0, 1.5),
    ],
)
def test_signal_out_open_r_of_an_active_signal(direction: str, last_price: float, expected_r: float) -> None:
    signal = make_signal(direction=direction, status="active", entry_price=100.0, stop_dist=2.0)

    out = presenters.signal_out(signal, NVDA, quote=make_quote(last_price))

    assert out.last_price == last_price
    assert out.open_r == pytest.approx(expected_r)


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "pending_entry"},  # not entered yet
        {"status": "review"},
        {"status": "active", "entry_price": None},  # entry unknown
        {"status": "target", "entry_price": 100.0, "exit_price": 103.0, "outcome_r": 1.4},  # closed: realised instead
        {"status": "active", "entry_price": 100.0, "stop_dist": 0.0},  # degenerate distance: no division by zero
    ],
)
def test_signal_out_has_no_open_r_unless_active_with_an_entry(fields: dict[str, Any]) -> None:
    out = presenters.signal_out(make_signal(**fields), NVDA, quote=make_quote(104.0))

    assert out.last_price == 104.0
    assert out.open_r is None


def test_signal_out_without_quote_has_no_live_fields() -> None:
    out = presenters.signal_out(make_signal(status="active", entry_price=100.0), NVDA)

    assert (out.last_price, out.open_r) == (None, None)


def test_signal_detail_out_adds_components_features_and_news() -> None:
    signal = full_signal()
    news = [presenters.news_out(make_item(), None)]

    detail = presenters.signal_detail_out(signal, NVDA, news=news, quote=make_quote(101.2))

    assert type(detail) is SignalDetailOut
    assert detail.components == {"trend": 0.8, "volume": 0.4}
    assert detail.features == {"rsi": 61.5, "ema20": 101.2, "note": "Rücksetzer"}
    assert detail.news == news
    assert (detail.name, detail.setup_label_de, detail.rr, detail.last_price) == ("NVIDIA", "Ausbruch", 1.5, 101.2)
    assert detail.open_r == pytest.approx(0.5)
    # everything a list entry shows is shown identically in the detail view
    summary = presenters.signal_out(signal, NVDA, quote=make_quote(101.2))
    assert detail.model_dump(exclude={"components", "features", "news"}) == summary.model_dump()


def test_signal_detail_out_copies_the_component_and_feature_maps() -> None:
    signal = full_signal()

    detail = presenters.signal_detail_out(signal, None)
    signal.components["trend"] = 0.0
    signal.features["rsi"] = 0.0

    assert detail.components["trend"] == 0.8 and detail.features["rsi"] == 61.5
    assert detail.news == []


@pytest.mark.parametrize("present", [presenters.signal_out, presenters.signal_detail_out])
def test_signals_must_be_stored_before_they_are_presented(present: Any) -> None:
    with pytest.raises(ValueError, match="signal has no id"):
        present(make_signal(id=None), NVDA)


# ------------------------------------------------------------------------------------------------
# Chart analyses
# ------------------------------------------------------------------------------------------------
def test_chart_analysis_out() -> None:
    analysis = ChartAnalysis(
        symbol="NVDA",
        ts=T0,
        timeframe="15m",
        model="vision-model",
        bias="bullish",
        strength=4,
        comment_de="Aufwärtstrend intakt",
        pattern="bull_flag",
        support=98.5,
        resistance=104.0,
        latency_ms=12_500,
        id=3,
    )

    assert presenters.chart_analysis_out(analysis).model_dump() == dataclasses.asdict(analysis)


def test_chart_analysis_out_needs_a_stored_analysis() -> None:
    analysis = ChartAnalysis(
        symbol="NVDA", ts=T0, timeframe="15m", model="m", bias="neutral", strength=2, comment_de="Seitwärts"
    )

    with pytest.raises(ValueError, match="chart analysis has no id"):
        presenters.chart_analysis_out(analysis)


# ------------------------------------------------------------------------------------------------
# Review additions: purity, units, boundaries
# ------------------------------------------------------------------------------------------------
def test_presenters_leave_their_inputs_alone_and_share_no_state_with_them() -> None:
    signal, quote = full_signal(), make_quote(101.2, bid=101.1)
    item = make_item(hinted_symbols=["NVDA"])
    analysis = NewsAnalysis(11, T0 + 45, "m", 4, 0.3, "guidance", "days", "Prognose angehoben", symbols={"NVDA": 0.7})
    before = [dataclasses.asdict(obj) for obj in (signal, quote, item, analysis)]

    news = presenters.news_out(item, analysis)
    detail = presenters.signal_detail_out(signal, NVDA, news=[news], quote=quote)
    watchlist_entry = presenters.instrument_out(NVDA, quote=quote, open_signal=signal)

    assert [dataclasses.asdict(obj) for obj in (signal, quote, item, analysis)] == before
    # changing the outputs must not reach the domain objects either
    news.hinted_symbols.append("AMD")
    detail.components["trend"] = 0.0
    detail.features["rsi"] = 0.0
    assert watchlist_entry.quote is not None
    watchlist_entry.quote.price = 0.0
    assert [dataclasses.asdict(obj) for obj in (signal, quote, item, analysis)] == before


def test_presenting_twice_gives_equal_but_independent_objects() -> None:
    signal = full_signal()

    first, second = presenters.signal_detail_out(signal, NVDA), presenters.signal_detail_out(signal, NVDA)
    first.news.append(presenters.news_out(make_item(), None))

    assert first.model_dump(exclude={"news"}) == second.model_dump(exclude={"news"})
    assert second.news == []  # no shared default list


def test_open_r_is_measured_in_quoted_units_whatever_the_price_scale() -> None:
    # London quotes pence: entry, stop distance and quote are all in pence, so the scale cancels out
    signal = make_signal(symbol="SHEL.L", status="active", entry_price=2500.0, stop_dist=25.0, direction="SHORT")

    out = presenters.signal_out(signal, SHELL, quote=Quote(symbol="SHEL.L", ts=T0, received_ts=T0, price=2450.0))

    assert (out.currency, out.delay_min, out.last_price) == ("GBP", 15, 2450.0)
    assert out.open_r == pytest.approx(2.0)


def test_signal_brief_of_a_shadow_signal_in_review() -> None:
    out = presenters.instrument_out(NVDA, open_signal=make_signal(direction="SHORT", confidence=0.0))

    assert out.open_signal == SignalBrief(id=7, direction="SHORT", status="review", confidence=0.0, created_ts=T0)
    assert (out.phase, out.is_open, out.quote) == ("closed", False, None)


@pytest.mark.parametrize(
    ("address", "masked"),
    [
        ("max@example.org, kein-postfach", "m***@example.org, ***"),
        ("<max@example.org>", "m***@example.org"),
        ("MAX@EXAMPLE.ORG", "M***@EXAMPLE.ORG"),
        ("müller@example.org", "m***@example.org"),
        # a stray separator must not turn a valid recipient list into '***'
        ("max@example.org,", "m***@example.org"),
        ("max@example.org;", "m***@example.org"),
        ("; max@example.org ;; erika@example.com ; ", "m***@example.org, e***@example.com"),
        ('"Mustermann, Max" <max@example.org>; erika@example.com;', "m***@example.org, e***@example.com"),
        (" ;, ", ""),
        # two addresses without a separator are no recipient list
        ("max@example.org erika@example.com", "***"),
        ("max@example.org <erika@example.com>", "***"),
    ],
)
def test_mask_email_with_mixed_and_unusual_input(address: str, masked: str) -> None:
    assert presenters.mask_email(address) == masked


@pytest.mark.parametrize(
    "address",
    [
        "max.mustermann@example.org,",
        "max.mustermann@example.org (Max Mustermann)",
        "Max Mustermann max.mustermann@example.org",
        "mailto:max.mustermann@example.org",
        "team: max.mustermann@example.org, erika.mustermann@example.com;",
        "max.mustermann@example.org\nerika.mustermann@example.com",
    ],
)
def test_mask_email_hides_the_local_part_of_malformed_input_too(address: str) -> None:
    assert "mustermann" not in presenters.mask_email(address).lower()


def test_address_masker_masks_the_configured_addresses_inside_free_text() -> None:
    mask = presenters.address_masker(["Max <max.mustermann@example.org>; erika@example.com", "", "kein-address"])
    reply = "553 5.7.1 <MAX.MUSTERMANN@EXAMPLE.ORG>: Sender address rejected; {'erika@example.com': (550, b'x')}"
    masked = mask(reply)
    assert masked == "553 5.7.1 <M***@EXAMPLE.ORG>: Sender address rejected; {'e***@example.com': (550, b'x')}"
    assert mask(masked) == masked  # masking twice changes nothing
    assert mask("andere@example.net bleibt") == "andere@example.net bleibt"  # only the configured addresses


def test_address_masker_prefers_the_longer_of_overlapping_addresses() -> None:
    mask = presenters.address_masker(["ax@example.org", "max@example.org"])
    assert mask("an max@example.org und ax@example.org") == "an m***@example.org und a***@example.org"


def test_address_masker_without_addresses_changes_nothing() -> None:
    text = "Verbindung abgelehnt (max@example.org)"
    assert presenters.address_masker(["", "  ", "ohne Adresse"])(text) == text


def test_news_out_keeps_an_empty_impact_list_and_ties_stable() -> None:
    def analysis(symbols: dict[str, float]) -> NewsAnalysis:
        return NewsAnalysis(11, T0 + 45, "m", 1, 0.0, "other", "weeks", "Ohne Bezug", symbols=symbols)

    without = presenters.news_out(make_item(), analysis({}))
    opposite = presenters.news_out(make_item(), analysis({"B": 0.5, "A": -0.5, "C": 0.5}))

    assert without.analysis is not None and without.analysis.symbols == []
    assert opposite.analysis is not None
    assert [impact.symbol for impact in opposite.analysis.symbols] == ["A", "B", "C"]  # equal strength: by symbol
