"""Mail content: pure functions that turn a signal into subject, plain text and HTML (no I/O, no clock).

All texts are German. Every dynamic value that goes into the HTML is escaped: rationale and risk texts
come from the LLM, which read untrusted news, so they are treated as hostile input. Free texts are also
capped in length.

The HTML is deliberately plain so that GMX, Gmail and Outlook render it alike: one centred table, inline
styles only, no external resources (the chart is referenced by Content-ID), explicit text AND background
colours on every block so that clients which switch to dark mode invert it consistently.

Stop and target prices in an entry mail are computed from the reference price and the planned distances:
the real entry happens on the next bar, and the distances then apply from the entry price.

The maximum holding period (``Signal.horizon_min``) counts TRADING minutes: the tracker and the AI portfolio
turn it into the time stop with ``MarketCalendar.add_trading_minutes``. The mail therefore never shows it as
wall-clock time: daily signals say 'ca. 5 Handelstage', intraday signals '4 Std. Handelszeit'.
"""

import html
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from trading_local.alerts.formatting import (
    MISSING,
    TIMEFRAME_LABELS_DE,
    format_datetime,
    format_duration,
    format_number,
    format_percent,
    format_price,
    format_r,
    format_ratio,
    price_decimals,
    price_unit,
)
from trading_local.models import Instrument, Signal
from trading_local.presenters import setup_label_de

FOOTER_DE: Final = "Automatisch erzeugte Analyse einer lokalen Simulation. Keine Anlageberatung."
SHORT_HINT_DE: Final = (
    "Short-Positionen lassen sich z. B. über CFDs, Knock-out-Zertifikate oder Short-ETFs umsetzen; "
    "in der Simulation wird direkt leerverkauft."
)
LEVELS_NOTE_DE: Final = (
    "Stopp und Ziel sind vom Referenzkurs aus berechnet. Der Einstieg erfolgt zum nächsten verfügbaren Kurs; "
    "die Abstände gelten dann ab dem Einstiegskurs."
)
R_NOTE_DE: Final = (
    "R = Vielfaches des geplanten Risikos (Abstand Einstieg bis Stopp); Prozent = Kursbewegung in Handelsrichtung. "
    "Beide nach der simulierten Slippage, die in Einstiegs- und Ausstiegskurs enthalten ist."
)
TEST_SUBJECT_DE: Final = "Testmail von Trading Local"

ACTION_DE: Final[dict[str, str]] = {"LONG": "Kaufen (Long)", "SHORT": "Leerverkaufen (Short)"}
SIDE_DE: Final[dict[str, str]] = {"LONG": "Long", "SHORT": "Short"}
VERDICT_DE: Final[dict[str, str]] = {
    "confirm": "bestätigt",
    "reduce": "reduzierte Positionsgröße empfohlen",
    "veto": "abgelehnt",
    "unavailable": "ohne KI-Prüfung (Modell nicht erreichbar)",
}
NO_REVIEW_DE: Final = "ohne KI-Prüfung"

# Readable names of the calendar keys (MIC or pseudo venue); unknown keys are shown as they are.
VENUE_NAMES_DE: Final[dict[str, str]] = {
    "XNAS": "Nasdaq",
    "XNYS": "NYSE",
    "XASE": "NYSE American",
    "ARCX": "NYSE Arca",
    "XETR": "Xetra",
    "XFRA": "Börse Frankfurt",
    "XPAR": "Euronext Paris",
    "XAMS": "Euronext Amsterdam",
    "XBRU": "Euronext Brüssel",
    "XMIL": "Borsa Italiana",
    "XMAD": "Börse Madrid",
    "XSWX": "SIX Swiss Exchange",
    "XLON": "London Stock Exchange",
    "XCSE": "Nasdaq Kopenhagen",
    "XSTO": "Nasdaq Stockholm",
    "XHEL": "Nasdaq Helsinki",
    "XOSL": "Börse Oslo",
    "XWBO": "Wiener Börse",
    "XTKS": "Börse Tokio",
    "XOSE": "Börse Osaka",
    "XHKG": "Börse Hongkong",
    "XSHG": "Börse Shanghai",
    "XSHE": "Börse Shenzhen",
    "XKRX": "Korea Exchange",
    "XTAI": "Börse Taiwan",
    "XBOM": "Börse Bombay",
    "XNSE": "National Stock Exchange of India",
    "XASX": "ASX Sydney",
    "XSES": "Börse Singapur",
    "CRYPTO": "Kryptohandel (rund um die Uhr)",
    "KRAKEN": "Kraken (Kryptobörse, rund um die Uhr)",
    "FX": "Devisenmarkt",
    "CME": "CME (Terminbörse)",
}

MAX_FREE_TEXT_CHARS: Final = 2000  # LLM texts longer than this are cut
# Regular trading minutes per session the engine assumes when it knows no session length
# (analysis.params.EngineParams.default_session_minutes; a test keeps both equal).
DEFAULT_SESSION_MINUTES: Final = 390

# Colours (light scheme; explicit everywhere so that dark-mode inversion stays consistent).
_PAGE_BG = "#f3f4f6"
_CARD_BG = "#ffffff"
_BORDER = "#d1d5db"
_RULE = "#e5e7eb"
_INK = "#111827"
_MUTED = "#4b5563"
_LINK_BG = "#1d4ed8"
_BADGE_BG: Final[dict[str, str]] = {"LONG": "#15803d", "SHORT": "#b91c1c", "EXIT": "#374151"}
_FONT = "Arial,Helvetica,sans-serif"
_CONTENT_WIDTH_PX = 592  # 640 px card minus 2 x 24 px padding

type Row = tuple[str, str]


@dataclass(frozen=True, slots=True)
class MailContent:
    subject: str
    text: str  # plain-text body
    html: str  # complete HTML document


# ------------------------------------------------------------------------------------------------
# Small public helpers
# ------------------------------------------------------------------------------------------------
def venue_name(exchange: str) -> str:
    """Readable German name of a calendar key ('XNAS' -> 'Nasdaq'); unknown keys are returned unchanged."""
    return VENUE_NAMES_DE.get(exchange, exchange)


def gui_base_url(host: str, port: int) -> str:
    """Base URL of the local GUI. Wildcard bind addresses are replaced by the loopback address."""
    reachable = {"": "127.0.0.1", "0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    if ":" in reachable and not reachable.startswith("["):
        reachable = f"[{reachable}]"  # IPv6 literal
    return f"http://{reachable}:{port}"


def signal_url(gui_base: str, signal_id: int | None) -> str | None:
    """Deep link to a signal in the GUI; None for a signal that has not been stored."""
    return None if signal_id is None else f"{gui_base}/#/signale/{signal_id}"


def verdict_de(signal: Signal) -> str:
    """The LLM verdict in words, with the model's conviction when it gave one."""
    text = VERDICT_DE.get(signal.llm_verdict or "", NO_REVIEW_DE)
    if signal.llm_verdict in ("confirm", "reduce") and signal.llm_conviction is not None:
        text += f" (Überzeugung {format_percent(signal.llm_conviction * 100, 0)})"
    return text


def holding_period_de(signal: Signal) -> str:
    """The maximum holding period in words, in trading time (see the module docstring).

    Daily signals: the horizon divided by the session length of the decision day (``minutes_since_open`` +
    ``minutes_to_close`` from the feature snapshot, as the engine computed it), rounded to whole trading days.
    Intraday signals: the trading time plus the reminder to close before the end of the session.
    """
    if signal.intraday:
        return f"{format_duration(signal.horizon_min)} Handelszeit, Schließen vor Handelsschluss"
    days = max(1, round(signal.horizon_min / _session_minutes(signal)))
    return "ca. 1 Handelstag" if days == 1 else f"ca. {days} Handelstage"


def _session_minutes(signal: Signal) -> float:
    """Regular trading minutes of the signal's decision day; the engine's default when the snapshot lacks them."""
    try:
        total = float(signal.features["minutes_since_open"]) + float(signal.features["minutes_to_close"])
    except (KeyError, TypeError, ValueError):
        return DEFAULT_SESSION_MINUTES
    return total if math.isfinite(total) and total > 0 else DEFAULT_SESSION_MINUTES


def exit_reason_de(signal: Signal) -> str:
    """Why a signal ended, from its status (and, for expired and cancelled signals, its ``status_reason``)."""
    match signal.status:
        case "target":
            return "Ziel erreicht"
        case "stopped":
            return "Stopp ausgelöst"
        case "expired":
            return "Handelsschluss" if _is_session_close(signal.status_reason) else "Haltedauer abgelaufen"
        case "cancelled":
            if _is_drift(signal.status_reason):
                return "Kurs vor dem Einstieg davongelaufen"
            return "kein Einstieg zustande gekommen"
        case _:
            return "Signal beendet"


# ------------------------------------------------------------------------------------------------
# Entry mail
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EntryLevels:
    """Stop and target of an entry recommendation, computed from the reference price."""

    ref: float
    stop: float
    target: float
    decimals: int

    @classmethod
    def of(cls, signal: Signal) -> "EntryLevels":
        ref = signal.ref_price
        return cls(
            ref=ref,
            stop=ref - signal.sign * signal.stop_dist,
            target=ref + signal.sign * signal.target_dist,
            decimals=price_decimals(ref),
        )

    def price(self, value: float) -> str:
        return format_price(value, self.decimals)

    def distance(self, value: float) -> str:
        """Signed distance from the reference price in percent."""
        return format_percent((value / self.ref - 1.0) * 100.0 if self.ref else float("nan"), 1, signed=True)


def entry_subject(signal: Signal, instrument: Instrument) -> str:
    """'LONG NVDA 142,35 USD | Stopp 140,90 | Ziel 145,25 | Konfidenz 58 %' (readable on a lock screen)."""
    levels = EntryLevels.of(signal)
    return (
        f"{signal.direction} {signal.symbol} {levels.price(levels.ref)} {price_unit(instrument)} | "
        f"Stopp {levels.price(levels.stop)} | Ziel {levels.price(levels.target)} | "
        f"Konfidenz {format_percent(signal.confidence * 100, 0)}"
    )


def entry_mail(
    signal: Signal, instrument: Instrument, *, timezone: str, gui_base: str, chart_cid: str | None = None
) -> MailContent:
    """Entry recommendation. ``chart_cid`` (bare Content-ID, without angle brackets) embeds the chart image."""
    levels = EntryLevels.of(signal)
    unit = price_unit(instrument)
    headline = f"{ACTION_DE[signal.direction]}: {instrument.name} ({signal.symbol}) – {venue_name(instrument.exchange)}"
    holding = holding_period_de(signal)
    rows: list[Row] = [
        ("Referenzkurs", f"{levels.price(levels.ref)} {unit}"),
        ("Stopp", f"{levels.price(levels.stop)} ({levels.distance(levels.stop)})"),
        ("Ziel", f"{levels.price(levels.target)} ({levels.distance(levels.target)})"),
        ("Chance/Risiko", format_ratio(signal.rr)),
        ("Maximale Haltedauer", holding),
        ("Konfidenz", format_percent(signal.confidence * 100, 0)),
        ("Erwartungswert", format_r(signal.expected_r, 2)),
        ("Setup", f"{setup_label_de(signal.setup)} ({TIMEFRAME_LABELS_DE.get(signal.timeframe, signal.timeframe)})"),
        ("Score", format_number(signal.score, 2)),
    ]
    freshness = [f"Datenstand: {format_datetime(signal.data_ts, timezone)} (letzte abgeschlossene Kerze)."]
    if instrument.delay_min > 0:
        freshness.append(
            f"Kursdaten dieses Marktes sind ca. {instrument.delay_min} Minuten verzögert - "
            "der Einstieg erfolgt zum nächsten verfügbaren Kurs."
        )
    sections = _sections([("Begründung", signal.rationale_de), ("Risiken", signal.risks_de)])
    verdict_line = f"KI-Prüfung: {verdict_de(signal)}"
    notes = [LEVELS_NOTE_DE, *freshness]
    hints = [SHORT_HINT_DE] if signal.direction == "SHORT" else []
    url = signal_url(gui_base, signal.id)
    subject = entry_subject(signal, instrument)

    text_parts = [headline, _text_rows(rows), "\n".join(notes)]
    text_parts += [f"{heading}:\n{body}" for heading, body in sections]
    text_parts += [verdict_line, *hints]
    if url is not None:
        text_parts.append(f"Signal in der Oberfläche öffnen: {url}")
    text_parts.append(f"--\n{FOOTER_DE}")

    blocks = [_html_badge(signal.direction, signal.direction), _html_headline(headline), _html_rows(rows)]
    if chart_cid is not None:
        blocks.append(_html_image(chart_cid, f"Chart {signal.symbol}"))
    blocks += [_html_paragraph(note, muted=True) for note in notes]
    blocks += [_html_section(heading, body) for heading, body in sections]
    blocks += [_html_paragraph(verdict_line, bold=True), *(_html_paragraph(hint) for hint in hints)]
    if url is not None:
        blocks.append(_html_link(url, "Signal in der Oberfläche öffnen"))
    return MailContent(subject=subject, text=_join_text(text_parts), html=_html_document(subject, blocks))


# ------------------------------------------------------------------------------------------------
# Exit mail
# ------------------------------------------------------------------------------------------------
def exit_mail(signal: Signal, instrument: Instrument, *, timezone: str, gui_base: str) -> MailContent:
    """Exit recommendation for a signal whose entry was mailed (target, stop, time stop, session close).

    A cancelled signal (no entry before the entry deadline, or the market ran away before the entry) gets a
    short 'ENTFÄLLT' notice instead.
    """
    side = SIDE_DE[signal.direction]
    reason = exit_reason_de(signal)
    name = f"{instrument.name} ({signal.symbol})"
    unit = price_unit(instrument)
    decimals = price_decimals(signal.entry_price or signal.ref_price)
    if signal.status == "cancelled":
        subject = f"ENTFÄLLT {signal.symbol} ({side}): {reason}"
        badge = "ENTFÄLLT"
        headline = f"{side}-Empfehlung entfällt: {name}"
        why = (
            "Der Kurs hatte sich vor dem Einstieg zu weit vom Referenzkurs entfernt."
            if _is_drift(signal.status_reason)
            else "Bis zum Ablauf der Einstiegsfrist kam kein Einstieg zustande."
        )
        summary = (
            f"Die {side}-Empfehlung für {name} entfällt: {why} "
            "Wer trotzdem eingestiegen ist, sollte die Position schließen."
        )
        rows: list[Row] = [("Grund", reason)]
        notes: list[str] = []
    else:
        subject = f"SCHLIESSEN {signal.symbol} ({side}): {reason}"
        if signal.outcome_r is not None:
            subject += f", {format_r(signal.outcome_r)}"
        badge = "SCHLIESSEN"
        headline = f"{side}-Position schließen: {name}"
        summary = (
            f"Die {side}-Empfehlung für {name} ist beendet: {reason}. "
            "Wer ihr gefolgt ist, sollte die Position jetzt schließen."
        )
        rows = [
            ("Ergebnis", f"{format_r(signal.outcome_r)} ({format_percent(_move_pct(signal), 1, signed=True)})"),
            ("Einstieg", _price_at(signal.entry_price, signal.entry_ts, unit, decimals, timezone)),
            ("Ausstieg", _price_at(signal.exit_price, signal.exit_ts, unit, decimals, timezone)),
            ("Grund", reason),
        ]
        notes = [R_NOTE_DE]
    rows.append(("Signal vom", format_datetime(signal.issued_ts or signal.created_ts, timezone)))
    url = signal_url(gui_base, signal.id)

    text_parts = [headline, summary, _text_rows(rows), *notes]
    if url is not None:
        text_parts.append(f"Signal in der Oberfläche öffnen: {url}")
    text_parts.append(f"--\n{FOOTER_DE}")

    blocks = [_html_badge(badge, "EXIT"), _html_headline(headline), _html_paragraph(summary), _html_rows(rows)]
    blocks += [_html_paragraph(note, muted=True) for note in notes]
    if url is not None:
        blocks.append(_html_link(url, "Signal in der Oberfläche öffnen"))
    return MailContent(subject=subject, text=_join_text(text_parts), html=_html_document(subject, blocks))


# ------------------------------------------------------------------------------------------------
# Test mail
# ------------------------------------------------------------------------------------------------
def check_mail(*, transport_de: str, now_ts: int, timezone: str, gui_base: str) -> MailContent:
    """Test message; ``transport_de`` describes the delivery path, e.g. 'SMTP über mail.gmx.net:465 (SSL)'."""
    intro = "Diese Testnachricht bestätigt, dass Trading Local E-Mails erzeugen und zustellen kann."
    rows: list[Row] = [
        ("Versandweg", transport_de),
        ("Erzeugt am", format_datetime(now_ts, timezone)),
        ("Oberfläche", f"{gui_base}/"),
    ]
    text = _join_text([intro, _text_rows(rows), f"--\n{FOOTER_DE}"])
    blocks = [_html_headline(intro), _html_rows(rows), _html_link(f"{gui_base}/", "Oberfläche öffnen")]
    return MailContent(subject=TEST_SUBJECT_DE, text=text, html=_html_document(TEST_SUBJECT_DE, blocks))


# ------------------------------------------------------------------------------------------------
# Helpers: content
# ------------------------------------------------------------------------------------------------
def _is_session_close(status_reason: str) -> bool:
    """True if a status reason says the position was flattened at the end of the trading session."""
    reason = status_reason.lower()
    return "session_close" in reason or "session close" in reason or "handelsschluss" in reason


def _is_drift(status_reason: str) -> bool:
    """True if a status reason says the signal was cancelled because the market ran away before the entry."""
    reason = status_reason.lower()
    return "entlaufen" in reason or "drift" in reason


def _move_pct(signal: Signal) -> float | None:
    """Price move between entry and exit in the trade's direction, in percent. Both prices are fill prices
    that already include the simulated slippage (``trading_local.execution``)."""
    if not signal.entry_price or signal.exit_price is None:
        return None
    return (signal.exit_price / signal.entry_price - 1.0) * 100.0 * signal.sign


def _price_at(price: float | None, ts: int | None, unit: str, decimals: int, timezone: str) -> str:
    if price is None:
        return MISSING
    text = f"{format_price(price, decimals)} {unit}"
    return text if ts is None else f"{text} ({format_datetime(ts, timezone)})"


def _clip(text: str) -> str:
    """Strip a free text and cut it to ``MAX_FREE_TEXT_CHARS``."""
    text = text.strip()
    return text if len(text) <= MAX_FREE_TEXT_CHARS else text[: MAX_FREE_TEXT_CHARS - 1].rstrip() + "…"


def _sections(candidates: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Headed free-text sections; empty texts are left out."""
    return [(heading, _clip(body)) for heading, body in candidates if body.strip()]


# ------------------------------------------------------------------------------------------------
# Helpers: plain text
# ------------------------------------------------------------------------------------------------
def _text_rows(rows: Sequence[Row]) -> str:
    width = max(len(label) for label, _ in rows) + 2
    return "\n".join(f"{label + ':':<{width}}{value}" for label, value in rows)


def _join_text(parts: Sequence[str]) -> str:
    return "\n\n".join(part for part in parts if part) + "\n"


# ------------------------------------------------------------------------------------------------
# Helpers: HTML (every dynamic value passes through _esc)
# ------------------------------------------------------------------------------------------------
def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _cell(content: str, padding: str = "8px 24px") -> str:
    return (
        f'<tr><td style="padding:{padding};background-color:{_CARD_BG};color:{_INK};'
        f'font-family:{_FONT};font-size:15px;line-height:1.45;">{content}</td></tr>'
    )


def _html_badge(label: str, kind: str) -> str:
    return _cell(
        f'<span style="display:inline-block;padding:4px 12px;border-radius:4px;background-color:{_BADGE_BG[kind]};'
        f'color:#ffffff;font-weight:bold;font-size:14px;letter-spacing:0.5px;">{_esc(label)}</span>',
        padding="20px 24px 4px 24px",
    )


def _html_headline(text: str) -> str:
    return _cell(f'<span style="font-size:18px;font-weight:bold;color:{_INK};">{_esc(text)}</span>')


def _html_rows(rows: Sequence[Row]) -> str:
    body = "".join(
        f'<tr><td style="padding:6px 12px 6px 0;border-bottom:1px solid {_RULE};color:{_MUTED};'
        f'font-family:{_FONT};font-size:14px;width:42%;vertical-align:top;">{_esc(label)}</td>'
        f'<td style="padding:6px 0;border-bottom:1px solid {_RULE};color:{_INK};font-family:{_FONT};'
        f'font-size:15px;font-weight:bold;vertical-align:top;">{_esc(value)}</td></tr>'
        for label, value in rows
    )
    return _cell(
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;">{body}</table>'
    )


def _html_image(cid: str, alt: str) -> str:
    return _cell(
        f'<img src="cid:{_esc(cid)}" width="{_CONTENT_WIDTH_PX}" alt="{_esc(alt)}" '
        f'style="display:block;width:100%;max-width:{_CONTENT_WIDTH_PX}px;height:auto;border:1px solid {_RULE};">'
    )


def _html_paragraph(text: str, *, muted: bool = False, bold: bool = False) -> str:
    style = f"color:{_MUTED if muted else _INK};font-size:{13 if muted else 15}px;"
    if bold:
        style += "font-weight:bold;"
    return _cell(f'<span style="{style}">{_multiline(text)}</span>')


def _html_section(heading: str, body: str) -> str:
    return _cell(
        f'<span style="font-weight:bold;color:{_INK};">{_esc(heading)}</span><br>'
        f'<span style="color:{_INK};">{_multiline(body)}</span>'
    )


def _html_link(url: str, label: str) -> str:
    href = _esc(url)
    return _cell(
        f'<a href="{href}" style="display:inline-block;padding:10px 16px;border-radius:4px;'
        f'background-color:{_LINK_BG};color:#ffffff;font-weight:bold;text-decoration:none;">{_esc(label)}</a>'
        f'<br><span style="font-size:12px;color:{_MUTED};">{href}</span>',
        padding="12px 24px",
    )


def _multiline(text: str) -> str:
    return "<br>".join(_esc(line) for line in text.splitlines())


def _html_document(title: str, blocks: Sequence[str]) -> str:
    footer = _cell(
        f'<span style="font-size:12px;color:{_MUTED};">{_esc(FOOTER_DE)}</span>', padding="16px 24px 20px 24px"
    )
    return (
        '<!DOCTYPE html>\n<html lang="de">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="light dark">\n<meta name="supported-color-schemes" content="light dark">\n'
        f"<title>{_esc(title)}</title>\n</head>\n"
        f'<body style="margin:0;padding:0;background-color:{_PAGE_BG};">\n'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="background-color:{_PAGE_BG};"><tr><td align="center" style="padding:16px 8px;">\n'
        f'<table role="presentation" width="640" cellpadding="0" cellspacing="0" border="0" '
        f'style="width:100%;max-width:640px;background-color:{_CARD_BG};border:1px solid {_BORDER};'
        f'border-radius:6px;border-collapse:separate;">\n'
        + "\n".join(blocks)
        + f"\n{footer}\n</table>\n</td></tr></table>\n</body>\n</html>\n"
    )
