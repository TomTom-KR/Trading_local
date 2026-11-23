"""Application wiring: lifespan hooks, GUI files, routing fallbacks and error responses."""

from pathlib import Path

from fastapi.testclient import TestClient

from trading_local.services import Services
from trading_local.web import create_app
from trading_local.web.app import INTERNAL_ERROR_DE

from .conftest import BASE_URL, Fakes


def test_lifespan_runs_hooks_exactly_once_and_binds_the_bus(services: Services, static_dir: Path) -> None:
    calls: list[str] = []

    async def on_startup() -> None:
        calls.append("start")

    async def on_shutdown() -> None:
        calls.append("stop")

    app = create_app(services, on_startup=on_startup, on_shutdown=on_shutdown, static_dir=static_dir)
    with TestClient(app, base_url=BASE_URL) as client:
        assert calls == ["start"]
        assert client.get("/api/status").status_code == 200
        assert services.bus._loop is not None and app.state.hub.running
    assert calls == ["start", "stop"]
    assert not app.state.hub.running


def test_index_is_served_without_caching(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-cache"
    assert "Trading Local" in response.text


def test_assets_have_correct_media_types_and_a_short_cache_lifetime(client: TestClient) -> None:
    expected = {
        "/app.mjs": "text/javascript",
        "/assets/main.js": "text/javascript",
        "/assets/style.css": "text/css",
        "/assets/icon.svg": "image/svg+xml",
        "/assets/data.json": "application/json",
        "/assets/app.mjs.map": "application/json",
        "/assets/font.woff2": "font/woff2",
        "/assets/logo.png": "image/png",
        "/favicon.ico": "image/x-icon",
    }
    for path, media_type in expected.items():
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers["content-type"].split(";")[0] == media_type, path
        assert response.headers["cache-control"] == "public, max-age=60", path


def test_assets_answer_conditional_requests_with_304(client: TestClient) -> None:
    etag = client.get("/assets/style.css").headers["etag"]
    response = client.get("/assets/style.css", headers={"if-none-match": etag})
    assert response.status_code == 304
    assert response.headers["cache-control"] == "public, max-age=60"


def test_unknown_files_and_traversal_attempts_are_404(client: TestClient) -> None:
    assert client.get("/missing.js").status_code == 404
    assert client.get("/..%2f..%2fpyproject.toml").status_code == 404
    response = client.get("/assets/../../conftest.py")
    assert response.status_code == 404


def test_missing_gui_directory_gives_a_german_note(services: Services, tmp_path: Path) -> None:
    app = create_app(services, static_dir=tmp_path / "does-not-exist")
    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Oberfläche ist nicht installiert" in response.text
        assert client.get("/app.js").status_code == 404
        assert client.get("/api/status").status_code == 200


def test_missing_index_gives_the_note_too(services: Services, tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    app = create_app(services, static_dir=tmp_path / "empty")
    with TestClient(app, base_url=BASE_URL) as client:
        assert "Oberfläche ist nicht installiert" in client.get("/").text


def test_api_and_ws_paths_never_fall_through_to_files(client: TestClient, static_dir: Path) -> None:
    (static_dir / "api").mkdir()
    (static_dir / "api" / "status2").write_text("file", encoding="utf-8")
    for path in ("/api", "/api/unknown", "/api/status2", "/ws", "/ws/other"):
        response = client.get(path)
        assert response.status_code == 404, path
        assert response.json() == {"detail": "Unbekannter API-Pfad."}, path


def test_wrong_method_on_a_known_route_is_405_in_german(client: TestClient) -> None:
    response = client.get("/api/validation/run")
    assert response.status_code == 405
    assert response.json()["detail"] == "Diese Methode ist für diese Adresse nicht erlaubt."


def test_unhandled_errors_become_a_generic_german_500(client: TestClient, fakes: Fakes) -> None:
    fakes.engine.analysis_error = RuntimeError("internal detail /secret/path")
    response = client.get("/api/analysis/NVDA")
    assert response.status_code == 500
    assert response.json() == {"detail": INTERNAL_ERROR_DE}
    assert "secret" not in response.text
    assert response.headers["x-content-type-options"] == "nosniff"


def test_openapi_schema_is_under_api_and_docs_pages_are_off(client: TestClient) -> None:
    assert client.get("/api/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 404
