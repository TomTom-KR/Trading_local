"""Runtime settings and the e-mail endpoints."""

from fastapi.testclient import TestClient

from trading_local.config import RuntimeSettings
from trading_local.db import repo_ops
from trading_local.services import Services

from .conftest import MAIL_FROM, MAIL_TO, NOW_TS


def test_get_returns_the_current_settings(client: TestClient) -> None:
    assert client.get("/api/settings").json() == RuntimeSettings().model_dump(mode="json")


def test_put_merges_a_subset_and_persists_it(client: TestClient, services: Services) -> None:
    response = client.put("/api/settings", json={"mail_max_per_hour": 5, "mail_regions": ["EU", "US"]})
    assert response.status_code == 200
    assert (response.json()["mail_max_per_hour"], response.json()["mail_regions"]) == (5, ["EU", "US"])
    assert response.json()["signals_enabled"] is True
    assert services.runtime_settings.get().mail_max_per_hour == 5
    assert client.get("/api/settings").json()["mail_max_per_hour"] == 5


def test_invalid_values_are_422_naming_the_field(client: TestClient, services: Services) -> None:
    response = client.put("/api/settings", json={"mail_max_per_hour": 500, "allow_long": True})
    assert response.status_code == 422
    assert response.json()["detail"] == "Ungültige Eingabe: Feld 'mail_max_per_hour' darf höchstens 120 sein."
    response = client.put("/api/settings", json={"mail_regions": ["MARS"]})
    assert response.status_code == 422 and "Feld 'mail_regions.0'" in response.json()["detail"]
    assert services.runtime_settings.get() == RuntimeSettings()


def test_unknown_keys_and_non_objects_are_rejected(client: TestClient) -> None:
    response = client.put("/api/settings", json={"smtp_password": "x", "mail_enabled": False})
    assert response.status_code == 422
    assert response.json()["detail"] == "Unbekannte Einstellung: 'smtp_password'."
    response = client.put("/api/settings", json=[1, 2])
    assert response.status_code == 422 and "JSON-Objekt" in response.json()["detail"]
    response = client.put("/api/settings", content=b"{broken", headers={"content-type": "application/json"})
    assert response.status_code == 422 and "kein gültiges JSON" in response.json()["detail"]


def test_mail_test_returns_the_mailer_result(client: TestClient) -> None:
    response = client.post("/api/mail/test")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "status": "file", "detail_de": "Testmail in den Postausgang geschrieben."}


def test_mail_log_masks_recipients(client: TestClient, services: Services) -> None:
    for offset, recipient in ((30, MAIL_TO), (20, f"Max <{MAIL_TO}>; zweite@example.invalid"), (10, "kaputt")):
        repo_ops.log_mail(services.db, ts=NOW_TS - offset, kind="test", recipient=recipient, subject="Test",
                          status="file")
    body = client.get("/api/mail/log").json()
    assert [entry["recipient"] for entry in body] == [
        "***", "e***@example.invalid, z***@example.invalid", "e***@example.invalid"
    ]
    assert len(client.get("/api/mail/log", params={"limit": 1}).json()) == 1
    assert client.get("/api/mail/log", params={"limit": 0}).status_code == 422


def test_mail_log_masks_addresses_inside_the_details(client: TestClient, services: Services) -> None:
    """Rows written before the mailer masked its texts: SMTP replies that repeat the configured addresses."""
    repo_ops.log_mail(
        services.db, ts=NOW_TS - 10, kind="signal", recipient=MAIL_TO, subject="LONG NVDA", status="failed",
        detail=f"Der Mailserver hat die Absenderadresse abgelehnt (553: 5.7.1 <{MAIL_FROM.upper()}>: Sender address "
               f"rejected), Empfänger {MAIL_TO}.",
    )
    detail = client.get("/api/mail/log").json()[0]["detail"]
    assert MAIL_TO not in detail and MAIL_FROM.upper() not in detail
    assert "<A***@EXAMPLE.INVALID>: Sender address rejected" in detail and "Empfänger e***@example.invalid." in detail
