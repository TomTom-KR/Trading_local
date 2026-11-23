"""Request schemas and the validation / clamping / filtering of model answers."""

import math
from typing import Any

import pytest

from tests.llm.fakes import chart_answer, news_answer, review_answer
from trading_local.llm import schemas
from trading_local.llm.client import schema_violation
from trading_local.llm.schemas import OutputError

CANDIDATES = ["NVDA", "AMD", "SAP.DE"]


def _news(data: dict[str, Any], candidates: list[str] = CANDIDATES) -> Any:
    return schemas.to_news_analysis(
        data, candidates=candidates, news_id=7, analyzed_ts=1000, model="qwen3.5:9b", latency_ms=1500
    )


def _chart(data: dict[str, Any], last_price: float | None = 100.0) -> Any:
    return schemas.to_chart_analysis(
        data, symbol="NVDA", timeframe="15m", ts=1000, model="qwen3.5:9b", last_price=last_price, latency_ms=3000
    )


# -- request schemas ------------------------------------------------------------------------------
def test_news_schema_limits_impacts_to_exactly_the_candidates() -> None:
    schema = schemas.news_schema(CANDIDATES)
    impacts = schema["properties"]["impacts"]
    assert impacts["items"]["properties"]["symbol"]["enum"] == CANDIDATES
    assert impacts["maxItems"] == 5
    assert impacts.get("minItems", 0) == 0  # an empty list is allowed
    assert set(schema["required"]) == set(schema["properties"])
    assert list(schema["properties"])[0] == "summary_de"  # text first: cheap reasoning before the scores


def test_news_schema_without_candidates_allows_only_an_empty_list() -> None:
    impacts = schemas.news_schema([])["properties"]["impacts"]
    assert impacts["maxItems"] == 0
    assert "enum" not in impacts["items"]["properties"]["symbol"]  # an empty enum would break the grammar


@pytest.mark.parametrize(
    "schema", [schemas.news_schema(CANDIDATES), schemas.trade_review_schema(), schemas.chart_schema()]
)
def test_schemas_carry_no_keywords_the_grammar_ignores_or_mangles(schema: dict[str, Any]) -> None:
    text = repr(schema)
    for keyword in ("maxLength", "minLength", "uniqueItems", "$ref", "pattern"):
        assert keyword not in text
    numbers = [s for s in _walk(schema) if s.get("type") == "number"]
    assert numbers and all("minimum" not in s and "maximum" not in s for s in numbers)  # clamped in code


def test_the_sample_answers_conform_to_their_schemas() -> None:
    assert schema_violation(news_answer(), schemas.news_schema(CANDIDATES)) == ""
    assert schema_violation(review_answer(), schemas.trade_review_schema()) == ""
    assert schema_violation(chart_answer(), schemas.chart_schema()) == ""


def _walk(schema: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(schema, dict):
        found.append(schema)
        for value in schema.values():
            found.extend(_walk(value))
    return found


# -- news -----------------------------------------------------------------------------------------
def test_news_answer_is_mapped_to_news_analysis() -> None:
    data = news_answer(impacts=[{"symbol": "NVDA", "sentiment": 0.7}, {"symbol": "AMD", "sentiment": -0.2}])
    analysis = _news(data)
    assert analysis.news_id == 7 and analysis.analyzed_ts == 1000 and analysis.model == "qwen3.5:9b"
    assert analysis.symbols == {"NVDA": 0.7, "AMD": -0.2}
    assert (analysis.relevance, analysis.sentiment, analysis.confidence) == (4, 0.6, 0.8)
    assert (analysis.category, analysis.horizon) == ("earnings", "days")
    assert analysis.summary_de == "Nvidia übertrifft die Erwartungen."
    assert analysis.latency_ms == 1500


def test_news_scores_are_clamped() -> None:
    data = news_answer(
        sentiment=3.5, confidence=-0.2, relevance=9, impacts=[{"symbol": "NVDA", "sentiment": -4.0}]
    )
    analysis = _news(data)
    assert (analysis.sentiment, analysis.confidence, analysis.relevance) == (1.0, 0.0, 5)
    assert analysis.symbols == {"NVDA": -1.0}
    assert _news(news_answer(relevance=0)).relevance == 1


def test_news_impacts_outside_the_candidates_are_dropped_and_duplicates_keep_the_first() -> None:
    impacts = [
        {"symbol": "TSLA", "sentiment": 0.9},
        {"symbol": "NVDA", "sentiment": 0.5},
        {"symbol": "NVDA", "sentiment": -0.5},
        {"symbol": "SAP.DE", "sentiment": 0.1},
    ]
    assert _news(news_answer(impacts=impacts)).symbols == {"NVDA": 0.5, "SAP.DE": 0.1}


def test_news_keeps_at_most_five_impacts_and_accepts_none() -> None:
    many = [f"S{i}" for i in range(8)]
    impacts = [{"symbol": symbol, "sentiment": 0.1} for symbol in many]
    assert len(_news(news_answer(impacts=impacts), many).symbols) == 5
    assert _news(news_answer(impacts=[])).symbols == {}


def test_news_summary_is_normalised_and_capped() -> None:
    summary = _news(news_answer(summary_de="  Nvidia \n übertrifft   die Erwartungen. ")).summary_de
    assert summary == "Nvidia übertrifft die Erwartungen."
    long = _news(news_answer(summary_de="Wort " * 200)).summary_de
    assert len(long) <= 300 and long.endswith("…")


@pytest.mark.parametrize(
    "data",
    [
        news_answer(summary_de="   "),
        news_answer(summary_de="Die Aktie pr채gt den Markt."),  # garbled umlaut seen with default sampling
        news_answer(summary_de="Die n盲chste Sitzung."),
        news_answer(category="rumour"),
        news_answer(horizon="years"),
        {k: v for k, v in news_answer().items() if k != "sentiment"},
        news_answer(sentiment=math.nan),
        news_answer(sentiment="bullish"),
        news_answer(impacts=[{"symbol": "NVDA"}]),
    ],
)
def test_unusable_news_answers_raise_output_error(data: dict[str, Any]) -> None:
    with pytest.raises(OutputError):
        _news(data)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Anmeldung unter https://nvda-bonus.example/login.", "Anmeldung unter [Link entfernt]"),
        ("Details: http://evil.example und www.evil.example/x", "Details: [Link entfernt] und [Link entfernt]"),
        ("Kontakt Investor@Evil-Bank.example sofort", "Kontakt [Link entfernt] sofort"),
        ("Bonus auf nvda-bonus.example/claim?id=7 abholen", "Bonus auf [Link entfernt] abholen"),
        ("hxxps://evil.example/x getarnt", "[Link entfernt] getarnt"),
        ("WWW.EVIL.EXAMPLE groß geschrieben", "[Link entfernt] groß geschrieben"),
    ],
)
def test_links_and_addresses_are_removed_from_free_texts(raw: str, expected: str) -> None:
    assert schemas.german_text(raw, "summary_de", 300) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Amazon.com Inc. hebt die Prognose an.",
        "EUR/USD notiert bei 1.0845/1.0850.",
        "Chance/Risiko-Verhältnis 2.0, Umsatz 3.5 Mrd./Quartal.",
        "S&P 500 +0,4 %, Kurs-Gewinn-Verhältnis 21,5.",
    ],
)
def test_ordinary_text_with_dots_and_slashes_is_kept(text: str) -> None:
    assert schemas.german_text(text, "summary_de", 300) == text


def test_a_link_from_a_hostile_headline_never_reaches_the_recommendation_mail() -> None:
    """Prompt-injection path: news summary -> trade review ('reuse their wording') -> entry mail."""
    from trading_local.alerts.templates import entry_mail
    from trading_local.models import Signal
    from trading_local.testing import SAMPLE_INSTRUMENTS

    evil = "https://nvda-bonus.example/login"
    news = _news(news_answer(summary_de=f"NVIDIA kündigt Sonderdividende an, Anmeldung unter {evil}."))
    review = schemas.to_trade_review(
        review_answer(reasons_de=[f"Sonderdividende, Details unter {evil}"], risks_de=["Phishing unter www.evil.example"]),
        latency_ms=1,
    )
    assert evil not in news.summary_de and "[Link entfernt]" in news.summary_de
    nvda = next(instrument for instrument in SAMPLE_INSTRUMENTS if instrument.symbol == "NVDA")
    signal = Signal(
        created_ts=1_795_000_000, issued_ts=1_795_000_000, data_ts=1_794_999_940, symbol="NVDA", direction="LONG",
        setup="news_momentum", timeframe="5m", horizon_min=240, ref_price=142.35, atr=1.2, stop_dist=1.8,
        target_dist=3.6, score=0.66, confidence=0.6, engine_version="t", expected_r=0.3, llm_verdict="confirm",
        llm_conviction=0.8, rationale_de=review.rationale_de, risks_de=review.risks_de, status="pending_entry",
        actionable=True, id=7,
    )
    mail = entry_mail(signal, nvda, timezone="Europe/Berlin", gui_base="http://127.0.0.1:8000")
    for body in (mail.text, mail.html):
        assert "evil.example" not in body and "nvda-bonus" not in body
        assert "Sonderdividende, Details unter [Link entfernt]" in body


def test_output_errors_carry_a_german_text_for_the_user() -> None:
    with pytest.raises(OutputError) as garbled:
        _news(news_answer(summary_de="Die n盲chste Sitzung."))
    assert str(garbled.value) == "summary_de contains garbled characters"
    assert garbled.value.reason_de == "Feld summary_de enthält verstümmelte Zeichen"
    with pytest.raises(OutputError) as invalid:
        _news(news_answer(category="rumour", horizon="years"))
    assert str(invalid.value).startswith("validation: category:")
    assert invalid.value.reason_de == "Ungültige Felder: category, horizon"


# -- trade review ---------------------------------------------------------------------------------
def test_review_answer_is_mapped_with_points_joined_by_line_breaks() -> None:
    review = schemas.to_trade_review(review_answer(), latency_ms=2500)
    assert review.verdict == "confirm"
    assert review.conviction == 0.85
    assert review.rationale_de == "Trend 15 Min aufwärts\nChance/Risiko-Verhältnis 2.0"
    assert review.risks_de == "Keine wesentlichen Gegenargumente"
    assert review.latency_ms == 2500


def test_review_lists_are_trimmed_and_cleaned() -> None:
    data = review_answer(
        reasons_de=["A", " ", "B", "A", "C", "D"],
        risks_de=["X", "Y", "Z"],
    )
    review = schemas.to_trade_review(data, latency_ms=None)
    assert review.rationale_de == "A\nB\nC"
    assert review.risks_de == "X\nY"


@pytest.mark.parametrize(
    ("verdict", "conviction", "expected"),
    [
        ("confirm", 0.3, 0.7),
        ("confirm", 1.4, 1.0),
        ("reduce", 0.95, 0.69),
        ("reduce", 0.1, 0.4),
        ("veto", 0.8, 0.39),
        ("veto", -1.0, 0.0),
    ],
)
def test_review_conviction_is_clamped_into_the_band_of_the_verdict(
    verdict: str, conviction: float, expected: float
) -> None:
    review = schemas.to_trade_review(review_answer(verdict=verdict, conviction=conviction), latency_ms=None)
    assert review.verdict == verdict
    assert review.conviction == expected


@pytest.mark.parametrize(
    "data",
    [
        review_answer(verdict="unavailable"),
        review_answer(verdict="buy"),
        review_answer(reasons_de=["Gut 好"]),
        {"verdict": "veto"},
    ],
)
def test_unusable_review_answers_raise_output_error(data: dict[str, Any]) -> None:
    with pytest.raises(OutputError):
        schemas.to_trade_review(data, latency_ms=None)


# -- chart ----------------------------------------------------------------------------------------
def test_chart_answer_is_mapped_with_nearest_plausible_levels() -> None:
    analysis = _chart(chart_answer(support=[95.0, 98.5, 99.0], resistance=[110.0, 103.0]))
    assert (analysis.symbol, analysis.timeframe, analysis.ts, analysis.model) == ("NVDA", "15m", 1000, "qwen3.5:9b")
    assert (analysis.bias, analysis.strength, analysis.pattern) == ("bullish", 4, "Aufwärtstrend")
    assert (analysis.support, analysis.resistance) == (99.0, 103.0)
    assert analysis.comment_de.startswith("Der Kurs steigt")


def test_chart_levels_further_than_25_percent_or_on_the_wrong_side_are_discarded() -> None:
    analysis = _chart(chart_answer(support=[70.0, 101.0], resistance=[130.0, 99.0]))
    assert analysis.support is None and analysis.resistance is None
    edge = _chart(chart_answer(support=[75.0], resistance=[125.0]))
    assert (edge.support, edge.resistance) == (75.0, 125.0)


def test_chart_levels_are_dropped_without_a_last_price() -> None:
    analysis = _chart(chart_answer(), last_price=None)
    assert analysis.support is None and analysis.resistance is None


def test_chart_strength_is_clamped_and_structures_have_german_labels() -> None:
    assert _chart(chart_answer(strength=9)).strength == 5
    assert set(schemas.STRUCTURE_LABELS_DE) == set(schemas.chart_schema()["properties"]["structure"]["enum"])
    assert _chart(chart_answer(structure="breakdown")).pattern == "Bruch nach unten"


@pytest.mark.parametrize(
    "data",
    [chart_answer(bias="up"), chart_answer(structure="flag"), chart_answer(comment_de=""), chart_answer(support="98")],
)
def test_unusable_chart_answers_raise_output_error(data: dict[str, Any]) -> None:
    with pytest.raises(OutputError):
        _chart(data)
