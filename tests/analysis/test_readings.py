"""German readings, price levels, rationale and LLM fact sheets of one feature row (pure functions)."""

import math
from typing import Any

import pytest

from trading_local.analysis import readings as rd
from trading_local.analysis.features import FEATURE_COLUMNS
from trading_local.dto import ChartAnalysisOut
from trading_local.models import Candidate, Direction, Signal
from trading_local.testing import SAMPLE_INSTRUMENTS

NVDA = SAMPLE_INSTRUMENTS[0]
NAN = math.nan


def uptrend_row(**overrides: Any) -> dict[str, Any]:
    """A complete feature row of a calm uptrend (price 101, slow ATR 1.0)."""
    row: dict[str, Any] = {
        "price": 101.0, "data_ts": 1_795_192_500, "session": "2026-11-20", "minutes_since_open": 125.0,
        "minutes_to_close": 265.0, "in_session": True, "atr_fast": 0.4, "atr_slow": 1.0, "atr_daily": 2.5,
        "ema_fast_f": 100.8, "ema_slow_f": 100.2, "ema_fast_s": 100.5, "ema_slow_s": 99.5, "ema_fast_s_slope": 0.3,
        "rsi_fast": 60.0, "rsi_slow": 62.0, "rsi_daily": 58.0, "macd_hist_s": 0.2, "adx_slow": 24.0,
        "bb_mid_s": 100.0, "bb_upper_s": 102.0, "bb_lower_s": 98.0, "vwap": 100.4, "prev_high_f": 100.9,
        "prev_low_f": 100.3, "prev_price_f": 100.6, "high_f": 101.1, "low_f": 100.5, "open_f": 100.6,
        "pullback_low": 99.8, "pullback_high": 101.1, "donchian_upper_s": 101.05, "donchian_lower_s": 97.0,
        "day_high": 101.5, "day_low": 99.0, "or_high": 100.2, "or_low": 99.05, "prev_day_high": 100.0,
        "prev_day_low": 98.0, "prev_day_close": 99.6, "gap_pct": 0.3, "ema_daily_fast": 97.0, "ema_daily_slow": 92.0,
        "ema_daily_pullback": 98.0, "rel_volume": 1.8, "atr_pct_rank": 0.55, "ret_60": 0.6, "bench_trend": 0.8,
        "bench_ret_60": 0.4, "rel_strength_60": 0.2,
    }  # fmt: skip
    row.update(overrides)
    return row


def downtrend_row() -> dict[str, Any]:
    return uptrend_row(
        ema_fast_f=100.0, ema_slow_f=100.6, ema_fast_s=100.0, ema_slow_s=101.5, prev_day_close=99.0,
        ema_daily_fast=100.0, ema_daily_slow=104.0, rsi_slow=40.0, vwap=101.5, bench_trend=-0.5, bench_ret_60=-0.3,
    )  # fmt: skip


COMPONENTS = {
    "trend": 0.9551, "momentum": 0.6855, "volume": 1.0, "structure": 1.0, "market": 0.9857, "news": 0.5,
    "volatility": 1.0,
}  # fmt: skip


def make_signal(components: dict[str, float] = COMPONENTS, direction: Direction = "LONG") -> Signal:
    return Signal(
        created_ts=1, data_ts=1, symbol="NVDA", direction=direction, setup="breakout", timeframe="15m",
        horizon_min=180, ref_price=101.0, atr=1.0, stop_dist=1.2, target_dist=2.4, score=0.86, confidence=0.48,
        engine_version="v", components=dict(components),
    )  # fmt: skip


def make_candidate(**features: Any) -> Candidate:
    return Candidate(
        symbol="NVDA", direction="LONG", setup="breakout", timeframe="15m", intraday=True, horizon_min=180,
        decision_ts=1_795_192_503, data_ts=1_795_192_500 - 300, ref_price=101.0, atr=1.0, stop_dist=1.2,
        target_dist=2.4, score=0.8644, features=uptrend_row(**features),
    )  # fmt: skip


def by_key(items: list[Any]) -> dict[str, Any]:
    return {item.key: item for item in items}


# ------------------------------------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------------------------------------
def test_the_test_row_has_every_feature_column() -> None:
    assert set(uptrend_row()) == set(FEATURE_COLUMNS)


def test_german_number_format() -> None:
    assert rd.de_number(1234.567) == "1.234,57"
    assert rd.de_number(0.4, 2, signed=True) == "+0,40"
    assert rd.de_number(-1.26, 1, signed=True) == "-1,3"
    assert rd.de_price(90123.4) == "90.123,40"
    assert rd.de_price(1.08567) == "1,0857"
    assert rd.direction_de("LONG") == "Long" and rd.direction_de("SHORT") == "Short"


# ------------------------------------------------------------------------------------------------
# Readings
# ------------------------------------------------------------------------------------------------
def test_readings_of_an_uptrend() -> None:
    out = rd.readings(uptrend_row(), news_score=0.35, news_count=3, benchmark_name="S&P 500")
    assert [r.key for r in out] == [
        "trend_5m", "trend_15m", "trend_daily", "momentum", "trend_strength", "volume", "vwap", "volatility",
        "benchmark", "news",
    ]  # fmt: skip
    r = by_key(out)
    assert r["trend_5m"].label_de == "Trend 5 Min"
    assert r["trend_5m"].value_de == "aufwärts (EMA20 über EMA50, Abstand 1,5 ATR)"
    assert r["trend_15m"].value_de == "aufwärts (EMA20 über EMA50, Abstand 1,0 ATR)"
    assert r["trend_daily"].value_de == "aufwärts (Schluss über EMA50 über EMA200)"
    assert (r["momentum"].label_de, r["momentum"].value_de) == ("Momentum (RSI 15 Min)", "62 – positiv")
    assert (r["trend_strength"].value_de, r["trend_strength"].stance) == ("24 – Trendphase", "neutral")
    assert r["volume"].value_de == "1,8-faches des Üblichen – hoch"
    assert r["vwap"].value_de == "über dem VWAP (+0,60 ATR, VWAP 100,40)"
    assert r["volatility"].value_de == "normal (55. Perzentil der letzten 10 Handelstage)"
    assert r["benchmark"].label_de == "Markt (S&P 500)"
    assert r["benchmark"].value_de == "15-Min-Trend aufwärts, 60 Min +0,40 %"
    assert r["news"].value_de == "positiv (+0,35 aus 3 Meldungen)"
    stances = {key: reading.stance for key, reading in r.items()}
    assert stances == {
        "trend_5m": "bullish", "trend_15m": "bullish", "trend_daily": "bullish", "momentum": "bullish",
        "trend_strength": "neutral", "volume": "neutral", "vwap": "bullish", "volatility": "neutral",
        "benchmark": "bullish", "news": "bullish",
    }  # fmt: skip


def test_readings_of_a_downtrend_are_bearish() -> None:
    r = by_key(rd.readings(downtrend_row(), news_score=-0.4, news_count=1))
    assert r["trend_5m"].value_de.startswith("abwärts (EMA20 unter EMA50")
    assert r["trend_daily"].value_de == "abwärts (Schluss unter EMA50 unter EMA200)"
    assert r["momentum"].value_de == "40 – negativ" and r["vwap"].value_de.startswith("unter dem VWAP (-0,50 ATR")
    assert r["benchmark"].label_de == "Markt" and r["news"].value_de == "negativ (-0,40 aus 1 Meldung)"
    assert {reading.stance for key, reading in r.items() if key not in ("trend_strength", "volume", "volatility")} == {
        "bearish"
    }


@pytest.mark.parametrize(
    ("overrides", "key", "value", "stance"),
    [
        ({"ema_fast_s": 99.55}, "trend_15m", "seitwärts (EMA20 nahe EMA50)", "neutral"),
        ({"rsi_slow": 74.0}, "momentum", "74 – überkauft", "neutral"),
        ({"rsi_slow": 25.0}, "momentum", "25 – überverkauft", "neutral"),
        ({"rsi_slow": 50.0}, "momentum", "50 – neutral", "neutral"),
        ({"adx_slow": 12.0}, "trend_strength", "12 – Seitwärtsphase", "neutral"),
        ({"rel_volume": 0.5}, "volume", "0,5-faches des Üblichen – niedrig", "neutral"),
        ({"vwap": 101.05}, "vwap", "am VWAP (-0,05 ATR, VWAP 101,05)", "neutral"),
        ({"atr_pct_rank": 0.9}, "volatility", "hoch (90. Perzentil der letzten 10 Handelstage)", "neutral"),
        ({"atr_pct_rank": 0.1}, "volatility", "niedrig (10. Perzentil der letzten 10 Handelstage)", "neutral"),
        (
            {"ema_daily_slow": 99.0},
            "trend_daily",
            "uneinheitlich (Schluss, EMA50 und EMA200 nicht geordnet)",
            "neutral",
        ),
        ({"ema_daily_slow": NAN}, "trend_daily", "aufwärts (Schluss über EMA50)", "bullish"),
        ({"ema_daily_slow": NAN, "ema_daily_fast": 100.0}, "trend_daily", "abwärts (Schluss unter EMA50)", "bearish"),
        ({"bench_trend": NAN}, "benchmark", "60 Min +0,40 %", "neutral"),
        ({"bench_trend": 0.05, "bench_ret_60": NAN}, "benchmark", "15-Min-Trend seitwärts", "neutral"),
    ],
)
def test_reading_variants(overrides: dict[str, float], key: str, value: str, stance: str) -> None:
    reading = by_key(rd.readings(uptrend_row(**overrides), news_score=None, news_count=0))[key]
    assert (reading.value_de, reading.stance) == (value, stance)


@pytest.mark.parametrize(
    ("score", "count", "value", "stance"),
    [
        (None, 0, "keine aktuellen Meldungen", "neutral"),
        (0.3, 0, "keine aktuellen Meldungen", "neutral"),
        (0.05, 2, "ausgeglichen (+0,05 aus 2 Meldungen)", "neutral"),
        (NAN, 2, "keine aktuellen Meldungen", "neutral"),
    ],
)
def test_news_reading(score: float | None, count: int, value: str, stance: str) -> None:
    reading = by_key(rd.readings(uptrend_row(), news_score=score, news_count=count))["news"]
    assert (reading.value_de, reading.stance) == (value, stance)


def test_missing_inputs_leave_readings_out_but_news_always_appears() -> None:
    empty = {key: NAN for key in FEATURE_COLUMNS}
    assert [r.key for r in rd.readings(empty, news_score=None, news_count=0)] == ["news"]
    assert [r.key for r in rd.readings({}, news_score=None, news_count=0)] == ["news"]


# ------------------------------------------------------------------------------------------------
# Levels
# ------------------------------------------------------------------------------------------------
def test_levels_are_sorted_deduplicated_and_classified() -> None:
    levels = rd.levels(uptrend_row())
    assert [(level.price, level.kind, level.label_de) for level in levels] == [
        (101.5, "day_high", "Tageshoch"),
        (101.05, "resistance", "20-Balken-Hoch (15 Min)"),
        (100.4, "vwap", "VWAP"),
        (100.2, "support", "Hoch der Eröffnungsspanne (30 Min)"),
        (100.0, "prev_high", "Vortageshoch"),
        (99.6, "prev_close", "Vortagesschluss"),
        (99.0, "day_low", "Tagestief"),  # the opening-range low 0.05 ATR above it is dropped
        (98.0, "prev_low", "Vortagestief"),
        (97.0, "support", "20-Balken-Tief (15 Min)"),
    ]


def test_levels_without_atr_drop_only_identical_prices() -> None:
    levels = rd.levels(uptrend_row(atr_slow=NAN, atr_fast=NAN, or_low=99.0))
    prices = [level.price for level in levels]
    assert prices.count(99.0) == 1 and 99.05 not in prices
    assert [level.kind for level in levels if level.price == 99.0] == ["day_low"]


def test_levels_skip_missing_values_and_relative_levels_without_price() -> None:
    levels = rd.levels(uptrend_row(price=NAN, vwap=NAN, day_high=NAN))
    kinds = {level.kind for level in levels}
    assert "vwap" not in kinds and "day_high" not in kinds and not kinds & {"support", "resistance"}


# ------------------------------------------------------------------------------------------------
# Analysis payload
# ------------------------------------------------------------------------------------------------
def test_finite_features_keep_real_numbers_only() -> None:
    features = rd.finite_features(uptrend_row(gap_pct=NAN, rel_volume=math.inf))
    assert "gap_pct" not in features and "rel_volume" not in features
    assert {"session", "in_session", "data_ts"}.isdisjoint(features)
    assert features["price"] == 101.0 and all(isinstance(value, float) for value in features.values())


def test_analysis_out_combines_all_parts() -> None:
    chart = ChartAnalysisOut(
        id=1, symbol="NVDA", ts=5, timeframe="15m", model="m", bias="bullish", strength=3, comment_de="x"
    )
    out = rd.analysis_out(
        "NVDA", uptrend_row(), as_of=1_795_192_503, news_score=0.123456, news_count=2, benchmark_name="S&P 500",
        chart_analysis=chart, note_de="Hinweis",
    )  # fmt: skip
    assert out.available and out.symbol == "NVDA" and out.as_of == 1_795_192_503 and out.data_ts == 1_795_192_500
    assert out.news_score == 0.1235 and out.chart_analysis == chart and out.note_de == "Hinweis"
    assert len(out.readings) == 10 and len(out.levels) == 9 and out.features["atr_slow"] == 1.0
    assert rd.analysis_out("X", {}, as_of=1, news_score=None, news_count=0).data_ts is None


# ------------------------------------------------------------------------------------------------
# Rationale
# ------------------------------------------------------------------------------------------------
def test_rationale_names_the_setup_the_two_strongest_and_the_weakest_component() -> None:
    text = rd.rationale_de(make_signal(), uptrend_row())
    assert text == (
        "Ausbruch (Long) bei NVDA: Am stärksten sprechen Kursstruktur (1,00) und "
        "Volatilität (1,00; ATR-Perzentil 55) für den Einstieg. Schwächster Faktor: Nachrichtenlage (0,50)."
    )


def test_rationale_details_come_from_the_row_and_are_optional() -> None:
    components = {"trend": 0.8, "volume": 0.9, "market": 0.3}
    with_row = rd.rationale_de(make_signal(components, "SHORT"), uptrend_row())
    assert with_row == (
        "Ausbruch (Short) bei NVDA: Am stärksten sprechen Volumen (0,90; 1,8-faches Volumen) und "
        "Trend (0,80; 15 Min aufwärts) für den Einstieg. "
        "Schwächster Faktor: Marktumfeld (0,30; Index 60 Min +0,4 %)."
    )
    assert rd.rationale_de(make_signal(components)).endswith("Schwächster Faktor: Marktumfeld (0,30).")


def test_rationale_edge_cases() -> None:
    assert rd.rationale_de(make_signal({})) == "Ausbruch (Long) bei NVDA. Keine Score-Komponenten verfügbar."
    single = rd.rationale_de(make_signal({"momentum": 0.7}), {"rsi_slow": 61.0})
    assert single.startswith("Ausbruch (Long) bei NVDA: Am stärksten spricht Momentum (0,70; RSI 61) für den")


# ------------------------------------------------------------------------------------------------
# LLM fact sheets
# ------------------------------------------------------------------------------------------------
def test_review_facts() -> None:
    facts = rd.review_facts(
        make_candidate(), instrument=NVDA, minutes_to_close=264, benchmark_name="S&P 500", news_score=0.25
    )
    assert facts == {
        "Setup": "Ausbruch", "Richtung": "Long (Kauf)", "Zeiteinheit": "15m", "Kurs": 101.0, "Währung": "USD",
        "ATR": 1.0, "Stopp-Abstand %": 1.19, "Ziel-Abstand %": 2.38, "Chance/Risiko": 2.0, "Score": 0.86,
        "Trend 5 Min": "aufwärts", "Trend 15 Min": "aufwärts", "Trend Tag": "aufwärts", "RSI 15 Min": 62.0,
        "RSI 5 Min": 60.0, "ADX 15 Min": 24.0, "Relatives Volumen": 1.8, "Eröffnungslücke %": 0.3,
        "Vergleichsindex 60 Min %": 0.4, "Abstand zum VWAP (ATR)": 0.6, "Lage zur Eröffnungsspanne": "darüber",
        "Lage in der Tagesspanne %": 80, "Vergleichsindex": "S&P 500", "Trend Vergleichsindex": "aufwärts",
        "Nachrichten-Score": 0.25, "Datenalter Min": 5, "Minuten bis Börsenschluss": 264,
    }  # fmt: skip


def test_review_facts_leave_out_what_is_unknown() -> None:
    candidate = make_candidate(rel_volume=NAN, vwap=NAN, or_high=NAN, bench_trend=NAN, ema_fast_f=NAN)
    facts = rd.review_facts(candidate, instrument=NVDA, minutes_to_close=None, benchmark_name=None, news_score=None)
    missing = {"Relatives Volumen", "Abstand zum VWAP (ATR)", "Lage zur Eröffnungsspanne", "Trend Vergleichsindex",
               "Trend 5 Min", "Vergleichsindex", "Nachrichten-Score", "Minuten bis Börsenschluss"}  # fmt: skip
    assert missing.isdisjoint(facts) and facts["Trend 15 Min"] == "aufwärts"
    inside = rd.review_facts(make_candidate(or_high=102.0, or_low=100.0), instrument=NVDA, minutes_to_close=1,
                             benchmark_name=None, news_score=None)  # fmt: skip
    assert inside["Lage zur Eröffnungsspanne"] == "innerhalb (50 %)"


def test_market_note() -> None:
    assert rd.market_note_de(uptrend_row(), "S&P 500") == "S&P 500: 15-Min-Trend aufwärts, 60 Min +0,40 %"
    assert rd.market_note_de(uptrend_row(), None) == "Kein Vergleichsindex hinterlegt."
    no_data = uptrend_row(bench_trend=NAN, bench_ret_60=NAN)
    assert rd.market_note_de(no_data, "DAX") == "Keine aktuellen Daten zum Vergleichsindex DAX."


def test_chart_facts() -> None:
    facts = rd.chart_facts(uptrend_row(price=101.123456789))
    assert facts["Kurs"] == 101.123 and facts["EMA20 (15 Min)"] == 100.5 and facts["ATR (15 Min)"] == 1.0
    assert facts["Swing-Hoch (20 x 15 Min)"] == 101.05 and facts["EMA200 (Tag)"] == 92.0
    assert "Vortageshoch" in rd.chart_facts(uptrend_row()) and "ATR (15 Min)" not in rd.chart_facts({"price": 5.0})
    assert rd.chart_facts(None, 99.5) == {"Kurs": 99.5}
    assert rd.chart_facts(None, None) == {} and rd.chart_facts(None, NAN) == {}
