"""German texts about agenda events (GUI, activity log, trade review). Pure functions."""

from datetime import datetime
from typing import Final
from zoneinfo import ZoneInfo

from trading_local.agenda import rules
from trading_local.analysis.readings import de_number
from trading_local.db.repo_agenda import AgendaEvent

KIND_LABELS_DE: Final[dict[str, str]] = {
    "macro": "Konjunkturdaten",
    "central_bank": "Notenbank",
    "earnings": "Quartalszahlen",
}
_WEEKDAYS_DE: Final = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")


def event_name_de(event: AgendaEvent) -> str:
    """'Quartalszahlen' for earnings, otherwise the published title with its currency: 'CPI m/m (USD)'."""
    if event.kind == "earnings":
        return "Quartalszahlen"
    return f"{event.title} ({event.currency})" if event.currency else event.title


def when_de(event: AgendaEvent, *, now_ts: int, timezone: str) -> str:
    """'heute 14:30', 'morgen 08:00', 'Do 28.01. 22:00' (user's time zone); 'heute' / 'Do 28.01.' without time."""
    zone = ZoneInfo(timezone)
    moment = datetime.fromtimestamp(event.event_ts, zone)
    today = datetime.fromtimestamp(now_ts, zone).date()
    days = (moment.date() - today).days
    day = "heute" if days == 0 else "morgen" if days == 1 else f"{_WEEKDAYS_DE[moment.weekday()]} {moment:%d.%m.}"
    return f"{day} {moment:%H:%M}" if event.time_known else day


def in_text_de(seconds: int) -> str:
    """'in 12 Min', 'in 5 Std', 'in 3 Tagen' (rounded down; 'jetzt' within a minute)."""
    if seconds < 60:
        return "jetzt"
    if seconds < 3600:
        return f"in {seconds // 60} Min"
    if seconds < 2 * 86_400:
        return f"in {seconds // 3600} Std"
    return f"in {seconds // 86_400} Tagen"


def surprise_label(event: AgendaEvent) -> rules.SurpriseLabel:
    """Side of the surprise of a released event ('' without actual value or forecast)."""
    if event.actual is None:
        return ""
    if event.kind == "earnings":
        return rules.earnings_label(event.surprise)
    return rules.surprise_label(rules.parse_value(event.actual_text), rules.parse_value(event.forecast_text))


def result_de(event: AgendaEvent) -> str:
    """The released value against the expectation, e.g. '0.4% statt 0.3% – über Erwartung (+1,8σ)' or for
    earnings 'EPS 2.10 statt 1.98 (+6,1 %) – über Erwartung'; '' without actual value."""
    if event.actual is None:
        return ""
    label = rules.SURPRISE_LABELS_DE[surprise_label(event)]
    if event.kind == "earnings":
        text = f"EPS {event.actual_text}"
        if event.forecast_text:
            text += f" statt {event.forecast_text}"
        if event.surprise is not None:
            text += f" ({de_number(event.surprise, 1, signed=True)} %)"
    else:
        text = event.actual_text + (f" statt {event.forecast_text}" if event.forecast_text else "")
    if label:
        text += f" – {label}"
    if event.surprise_z is not None and event.kind != "earnings":
        text += f" ({de_number(event.surprise_z, 1, signed=True)}σ)"
    return text


def upcoming_de(event: AgendaEvent, *, now_ts: int, timezone: str) -> str:
    """'CPI m/m (USD) heute 14:30 (in 95 Min)'."""
    return f"{event_name_de(event)} {when_de(event, now_ts=now_ts, timezone=timezone)} ({in_text_de(event.event_ts - now_ts)})"


def released_de(event: AgendaEvent) -> str:
    """'CPI m/m (USD): 0.4% statt 0.3% – über Erwartung'."""
    return f"{event_name_de(event)}: {result_de(event)}"
