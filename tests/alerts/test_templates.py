import re
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from trading_local.alerts import templates
from trading_local.models import Signal
from trading_local.testing import SAMPLE_INSTRUMENTS

INSTRUMENTS = {instrument.symbol: instrument for instrument in SAMPLE_INSTRUMENTS}
TS = int(datetime(2026, 11, 20, 15, 5, tzinfo=UTC).timestamp())  # 16:05 in Berlin
GUI = "http://127.0.0.1:8000"
TZ = "Europe/Berlin"
HOSTILE = '<script>alert("x")</script><img src="http://evil.example/p.png" onerror="steal()"> & </td></table>'


def signal(**overrides: Any) -> Signal:
    values: dict[str, Any] = {
        "id": 42,
        "created_ts": TS,
        "issued_ts": TS,
        "data_ts": TS,
        "symbol": "NVDA",
        "direction": "LONG",
        "setup": "trend_pullback",
        "timeframe": "15m",
        "horizon_min": 390,
        "ref_price": 142.35,
        "atr": 1.2,
        "stop_dist": 1.45,
        "target_dist": 2.9,
        "score": 0.64,
        "confidence": 0.58,
        "engine_version": "test",
        "expected_r": 0.32,
        "llm_verdict": "confirm",
        "llm_conviction": 0.7,
        "rationale_de": "Aufwärtstrend intakt.\nRücksetzer an die EMA 20.",
        "risks_de": "Quartalszahlen in zwei Tagen.",
        "status": "pending_entry",
        "actionable": True,
    }
    values.update(overrides)
    return Signal(**values)


def entry(sig: Signal, symbol: str = "NVDA", cid: str | None = None) -> templates.MailContent:
    return templates.entry_mail(sig, INSTRUMENTS[symbol], timezone=TZ, gui_base=GUI, chart_cid=cid)


def visible_text(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


# ------------------------------------------------------------------------------------------------
# Entry mail
# ------------------------------------------------------------------------------------------------
def test_long_subject_is_readable_on_a_lock_screen() -> None:
    assert entry(signal()).subject == "LONG NVDA 142,35 USD | Stopp 140,90 | Ziel 145,25 | Konfidenz 58 %"


def test_short_subject_puts_the_stop_above_and_the_target_below() -> None:
    content = entry(signal(direction="SHORT"))
    assert content.subject == "SHORT NVDA 142,35 USD | Stopp 143,80 | Ziel 139,45 | Konfidenz 58 %"
    assert "Leerverkaufen (Short): NVIDIA (NVDA) – Nasdaq" in content.text
    assert templates.SHORT_HINT_DE in content.text
    assert templates.SHORT_HINT_DE in content.html


def test_entry_body_lists_all_facts() -> None:
    text = entry(signal()).text
    assert text.startswith("Kaufen (Long): NVIDIA (NVDA) – Nasdaq\n")
    for expected in (
        "Referenzkurs:        142,35 USD",
        "140,90 (-1,0 %)",
        "145,25 (+2,0 %)",
        "Chance/Risiko:       2,0 : 1",
        "6 Std. 30 Min. Handelszeit, Schließen vor Handelsschluss",
        "Konfidenz:           58 %",
        "Erwartungswert:      +0,32 R",
        "Trend-Rücksetzer (15 Min)",
        "Score:               0,64",
        "Datenstand: 20.11.2026 16:05",
        "Begründung:\nAufwärtstrend intakt.\nRücksetzer an die EMA 20.",
        "Risiken:\nQuartalszahlen in zwei Tagen.",
        "KI-Prüfung: bestätigt (Überzeugung 70 %)",
        "http://127.0.0.1:8000/#/signale/42",
        templates.FOOTER_DE,
        templates.LEVELS_NOTE_DE,
    ):
        assert expected in text, expected
    assert templates.SHORT_HINT_DE not in text
    assert "verzögert" not in text  # NVDA has a real-time feed


def test_delayed_market_gets_the_delay_sentence_and_quoted_units() -> None:
    sap = entry(signal(symbol="SAP.DE", ref_price=182.35), "SAP.DE")
    assert (
        "Kursdaten dieses Marktes sind ca. 15 Minuten verzögert - der Einstieg erfolgt zum nächsten verfügbaren Kurs."
        in sap.text
    )
    assert "Kaufen (Long): SAP (SAP.DE) – Xetra" in sap.text
    shell = entry(signal(symbol="SHEL.L", ref_price=2650.0, stop_dist=30.0, target_dist=60.0), "SHEL.L")
    assert shell.subject.startswith("LONG SHEL.L 2.650,0 GBp | Stopp 2.620,0 | Ziel 2.710,0")


def test_daily_signal_has_no_session_close_note() -> None:
    daily = signal(intraday=False, horizon_min=5 * 390, timeframe="1d", setup="daily_trend")
    text = entry(daily).text
    assert "Maximale Haltedauer: ca. 5 Handelstage\n" in text
    assert "Schließen vor Handelsschluss" not in text
    assert "Tagestrend (1 Tag)" in text


@pytest.mark.parametrize(
    ("horizon", "features", "expected"),
    [
        # Xetra: 510 trading minutes per session; 5 sessions are 2,550 trading minutes (about 7 calendar days)
        (2550, {"minutes_since_open": 120.0, "minutes_to_close": 390.0}, "ca. 5 Handelstage"),
        (1950, {"minutes_since_open": 15.0, "minutes_to_close": 375.0}, "ca. 5 Handelstage"),  # New York
        (300, {"minutes_since_open": 100.0, "minutes_to_close": 200.0}, "ca. 1 Handelstag"),
        (1950, {}, "ca. 5 Handelstage"),  # no session length known: the engine's default of 390 minutes
        (1950, {"minutes_since_open": "kaputt", "minutes_to_close": None}, "ca. 5 Handelstage"),
    ],
)
def test_daily_holding_period_counts_trading_days(horizon: int, features: dict[str, Any], expected: str) -> None:
    """horizon_min counts trading minutes (time stop via add_trading_minutes); '1 Tag 18 Std.' was wrong."""
    daily = signal(intraday=False, horizon_min=horizon, timeframe="1d", setup="daily_trend", features=features)
    assert templates.holding_period_de(daily) == expected
    assert f"Maximale Haltedauer: {expected}" in entry(daily).text
    assert "Std." not in templates.holding_period_de(daily)


def test_intraday_holding_period_is_trading_time() -> None:
    text = entry(signal(horizon_min=240)).text
    assert "Maximale Haltedauer: 4 Std. Handelszeit, Schließen vor Handelsschluss" in text


def test_default_session_length_matches_the_engine() -> None:
    from trading_local.analysis.params import ENGINE_PARAMS

    assert templates.DEFAULT_SESSION_MINUTES == ENGINE_PARAMS.default_session_minutes


@pytest.mark.parametrize(
    ("verdict", "conviction", "line"),
    [
        ("confirm", None, "KI-Prüfung: bestätigt"),
        ("reduce", 0.55, "KI-Prüfung: reduzierte Positionsgröße empfohlen (Überzeugung 55 %)"),
        ("unavailable", None, "KI-Prüfung: ohne KI-Prüfung (Modell nicht erreichbar)"),
        (None, None, "KI-Prüfung: ohne KI-Prüfung"),
    ],
)
def test_llm_verdict_in_words(verdict: str | None, conviction: float | None, line: str) -> None:
    text = entry(signal(llm_verdict=verdict, llm_conviction=conviction)).text
    assert line in text.splitlines()


def test_empty_free_texts_are_left_out_and_long_ones_cut() -> None:
    text = entry(signal(rationale_de="  ", risks_de="x" * 5000)).text
    assert "Begründung" not in text
    risks = text.split("Risiken:\n", 1)[1].split("\n", 1)[0]
    assert len(risks) == templates.MAX_FREE_TEXT_CHARS
    assert risks.endswith("…")


def test_unstored_signal_has_no_link() -> None:
    content = entry(signal(id=None))
    assert "/#/signale/" not in content.text
    assert "/#/signale/" not in content.html


def test_html_escapes_hostile_llm_text() -> None:
    content = entry(signal(rationale_de=HOSTILE, risks_de=HOSTILE), cid="chart.abc@trading.local")
    html = content.html
    assert "<script" not in html
    assert '<img src="http://evil' not in html
    assert html.count("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;") == 2
    assert "&lt;/td&gt;&lt;/table&gt;" in html
    # the only image is the inline chart; no external resources at all
    assert re.findall(r'src="([^"]*)"', html) == ["cid:chart.abc@trading.local"]
    assert "<link" not in html and "url(" not in html


def test_html_contains_the_same_facts_and_no_image_without_chart() -> None:
    content = entry(signal())
    text = visible_text(content.html)
    for expected in ("Referenzkurs", "142,35 USD", "140,90 (-1,0 %)", "2,0 : 1", "KI-Prüfung: bestätigt",
                     templates.FOOTER_DE, "Kaufen (Long): NVIDIA (NVDA) – Nasdaq"):
        assert expected in text, expected
    assert "<img" not in content.html
    assert 'href="http://127.0.0.1:8000/#/signale/42"' in content.html
    assert content.html.startswith("<!DOCTYPE html>")
    assert 'name="color-scheme" content="light dark"' in content.html


# ------------------------------------------------------------------------------------------------
# Exit mail
# ------------------------------------------------------------------------------------------------
def closed(**overrides: Any) -> Signal:
    values: dict[str, Any] = {
        "status": "target",
        "entry_ts": TS + 60,
        "entry_price": 142.40,
        "exit_ts": TS + 5400,
        "exit_price": 145.30,
        "outcome_r": 1.94,
        "mailed_ts": TS,
    }
    values.update(overrides)
    return signal(**values)


def exit_content(sig: Signal) -> templates.MailContent:
    return templates.exit_mail(sig, INSTRUMENTS["NVDA"], timezone=TZ, gui_base=GUI)


@pytest.mark.parametrize(
    ("overrides", "subject"),
    [
        ({}, "SCHLIESSEN NVDA (Long): Ziel erreicht, +1,9 R"),
        ({"status": "stopped", "exit_price": 140.9, "outcome_r": -1.04},
         "SCHLIESSEN NVDA (Long): Stopp ausgelöst, -1,0 R"),
        ({"status": "expired", "outcome_r": 0.4}, "SCHLIESSEN NVDA (Long): Haltedauer abgelaufen, +0,4 R"),
        ({"status": "expired", "status_reason": "session_close", "outcome_r": 0.4},
         "SCHLIESSEN NVDA (Long): Handelsschluss, +0,4 R"),
        ({"status": "expired", "status_reason": "Glattstellung vor Handelsschluss", "outcome_r": None},
         "SCHLIESSEN NVDA (Long): Handelsschluss"),
        ({"status": "cancelled", "entry_ts": None, "entry_price": None, "exit_ts": None, "exit_price": None,
          "outcome_r": None}, "ENTFÄLLT NVDA (Long): kein Einstieg zustande gekommen"),
    ],
)
def test_exit_subjects(overrides: dict[str, Any], subject: str) -> None:
    assert exit_content(closed(**overrides)).subject == subject


def test_exit_body_reports_prices_times_and_result() -> None:
    content = exit_content(closed(direction="SHORT", entry_price=145.30, exit_price=142.40))
    text = content.text
    assert text.startswith("Short-Position schließen: NVIDIA (NVDA)\n")
    assert "Ergebnis:   +1,9 R (+2,0 %)" in text  # a falling price is a gain for a short
    assert "Einstieg:   145,30 USD (20.11.2026 16:06)" in text
    assert "Ausstieg:   142,40 USD (20.11.2026 17:35)" in text
    assert "Grund:      Ziel erreicht" in text
    assert "http://127.0.0.1:8000/#/signale/42" in text
    assert templates.FOOTER_DE in text
    assert "Ziel erreicht" in visible_text(content.html)


def test_exit_html_escapes_names() -> None:
    hostile = replace(INSTRUMENTS["NVDA"], name="<b>NVIDIA</b>")
    html = templates.exit_mail(closed(), hostile, timezone=TZ, gui_base=GUI).html
    assert "<b>NVIDIA</b>" not in html
    assert "&lt;b&gt;NVIDIA&lt;/b&gt;" in html


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("host", "url"),
    [
        ("127.0.0.1", "http://127.0.0.1:8000"),
        ("0.0.0.0", "http://127.0.0.1:8000"),
        ("::", "http://[::1]:8000"),
        ("::1", "http://[::1]:8000"),
        ("tradingpc.local", "http://tradingpc.local:8000"),
    ],
)
def test_gui_base_url(host: str, url: str) -> None:
    assert templates.gui_base_url(host, 8000) == url


def test_venue_names() -> None:
    assert templates.venue_name("XETR") == "Xetra"
    assert templates.venue_name("XYZZ") == "XYZZ"


def test_check_mail() -> None:
    content = templates.check_mail(
        transport_de="SMTP über mail.gmx.net:465 (SSL)", now_ts=TS, timezone=TZ, gui_base=GUI
    )
    assert content.subject == templates.TEST_SUBJECT_DE
    assert "SMTP über mail.gmx.net:465 (SSL)" in content.text
    assert "20.11.2026 16:05" in content.text
    assert templates.FOOTER_DE in content.html
