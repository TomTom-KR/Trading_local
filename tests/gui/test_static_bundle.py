"""Static checks of the browser bundle in trading_local/web/static.

The GUI has no build step and runs under a strict Content-Security-Policy (script-src 'self', no inline
scripts, no eval, no import maps) and fully offline. These tests make sure that every file the browser
loads exists, every ES-module import resolves (and names an export that exists), nothing is fetched from
another host, and the vendored libraries are the reviewed versions with their licences.
"""

import hashlib
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "trading_local" / "web" / "static"
VENDOR = STATIC / "vendor"
OWN_JS = sorted(p for p in STATIC.rglob("*.js") if VENDOR not in p.parents)
ALL_MODULES = OWN_JS + sorted(VENDOR.glob("*.mjs"))
TEXT_ASSETS = [*OWN_JS, *STATIC.rglob("*.css"), *STATIC.rglob("*.html"), *STATIC.rglob("*.svg")]
# URLs that may appear in our own files: the XML namespace of inline SVG and the TradingView attribution
# link that the lightweight-charts licence requires (a link the user can click, nothing is loaded from it).
ALLOWED_URLS = {"http://www.w3.org/2000/svg", "https://www.tradingview.com/"}
# Modules that are referenced only from JSDoc comments (types), never imported at runtime.
TYPE_ONLY_MODULES = {"lib/types.js"}

IMPORT_RE = re.compile(
    r"""(?:^|[;\s])(?:import|export)\s*(?:[\w*{}\s,$]+?\s*from\s*)?['"]([^'"]+)['"]"""
    r"""|\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)""",
    re.MULTILINE,
)
NAMED_IMPORT_RE = re.compile(r"import\s*\{([^}]*)\}\s*from\s*['\"]([^'\"]+)['\"]")


def _strip_comments(source: str) -> str:
    """Remove block comments and whole-line // comments (enough for our own, hand-written modules)."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", source, flags=re.MULTILINE)


def _code(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text if VENDOR in path.parents else _strip_comments(text)


def _specifiers(path: Path) -> list[str]:
    return [a or b for a, b in IMPORT_RE.findall(_code(path))]


def _resolve(base: Path, specifier: str) -> Path:
    return (base.parent / specifier).resolve()


def _exports(path: Path) -> set[str]:
    code = _code(path)
    names = set(re.findall(r"export\s+(?:async\s+)?(?:function\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)", code))
    for block in re.findall(r"export\s*\{([^}]*)\}", code):
        for part in block.split(","):
            part = part.strip()
            if part:
                names.add(part.split(" as ")[-1].strip())
    return names


class _IndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[tuple[dict[str, str | None], str]] = []
        self.links: list[dict[str, str | None]] = []
        self.inline_styles: list[str] = []
        self.handlers: list[str] = []
        self._in_script: dict[str, str | None] | None = None
        self._script_text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        self.handlers += [name for name in attributes if name.startswith("on")]
        if "style" in attributes:
            self.inline_styles.append(tag)
        if tag == "style":
            self.inline_styles.append("<style>")
        if tag == "link":
            self.links.append(attributes)
        if tag == "script":
            self._in_script, self._script_text = attributes, ""

    def handle_data(self, data: str) -> None:
        if self._in_script is not None:
            self._script_text += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_script is not None:
            self.scripts.append((self._in_script, self._script_text))
            self._in_script = None


@pytest.fixture(scope="module")
def index() -> _IndexParser:
    parser = _IndexParser()
    parser.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    return parser


def test_index_loads_the_app_as_one_module_script(index: _IndexParser) -> None:
    assert [attrs.get("src") for attrs, _ in index.scripts] == ["app.js"]
    attrs, body = index.scripts[0]
    assert attrs.get("type") == "module"
    assert body.strip() == "", "inline script content is blocked by the CSP"


def test_index_has_no_inline_code_or_styles(index: _IndexParser) -> None:
    assert index.handlers == [], "inline event handlers are blocked by the CSP"
    assert index.inline_styles == [], "inline styles would need style-src 'unsafe-inline'"
    assert not any(attrs.get("type") == "importmap" for attrs, _ in index.scripts)


def test_every_file_referenced_from_index_exists(index: _IndexParser) -> None:
    refs = [attrs.get("src") for attrs, _ in index.scripts] + [link.get("href") for link in index.links]
    assert refs, "index.html references nothing"
    for ref in refs:
        assert ref and not re.match(r"^[a-z]+:", ref), f"absolute or external reference: {ref}"
        assert (STATIC / ref).is_file(), f"index.html references a missing file: {ref}"


@pytest.mark.parametrize("module", ALL_MODULES, ids=lambda p: p.relative_to(STATIC).as_posix())
def test_every_import_resolves_to_a_bundled_file(module: Path) -> None:
    for spec in _specifiers(module):
        assert spec.startswith(("./", "../")), f"{module.name}: bare or absolute import '{spec}' (no import maps)"
        target = _resolve(module, spec)
        assert target.is_file(), f"{module.name}: import '{spec}' does not resolve"
        assert STATIC in target.parents, f"{module.name}: import '{spec}' leaves the static directory"


def test_vendor_modules_are_self_contained() -> None:
    for module in VENDOR.glob("*.mjs"):
        assert _specifiers(module) == [], f"{module.name} imports other files"


@pytest.mark.parametrize("module", OWN_JS, ids=lambda p: p.relative_to(STATIC).as_posix())
def test_named_imports_exist_and_are_used(module: Path) -> None:
    code = _code(module)
    for names, spec in NAMED_IMPORT_RE.findall(code):
        exported = _exports(_resolve(module, spec))
        for part in filter(None, (n.strip() for n in names.split(","))):
            imported, _, local = part.partition(" as ")
            assert imported.strip() in exported, f"{module.name}: '{imported.strip()}' is not exported by {spec}"
            name = (local or imported).strip()
            uses = len(re.findall(rf"(?<![\w$.]){re.escape(name)}(?![\w$])", code))
            assert uses >= 2, f"{module.name}: '{name}' is imported but never used"


def test_every_own_module_is_reachable_from_app_js() -> None:
    seen: set[Path] = set()
    todo = [STATIC / "app.js"]
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        todo += [_resolve(current, s) for s in _specifiers(current) if s.startswith(".")]
    unreachable = {p.relative_to(STATIC).as_posix() for p in OWN_JS} - {p.relative_to(STATIC).as_posix() for p in seen}
    assert unreachable == TYPE_ONLY_MODULES


def test_css_urls_resolve_locally() -> None:
    for css in STATIC.rglob("*.css"):
        for ref in re.findall(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", css.read_text(encoding="utf-8")):
            assert not re.match(r"^[a-z]+:", ref), f"{css.name}: external url({ref})"
            assert (css.parent / ref).is_file(), f"{css.name}: url({ref}) does not exist"
        assert "@import" not in css.read_text(encoding="utf-8"), f"{css.name}: use <link> instead of @import"


@pytest.mark.parametrize("path", TEXT_ASSETS, ids=lambda p: p.relative_to(STATIC).as_posix())
def test_no_external_urls_in_own_files(path: Path) -> None:
    urls = set(re.findall(r"(?:https?:)?//[A-Za-z0-9.-]+\.[a-z]{2,}[^\s'\"`)<>]*", _code(path)))
    unexpected = {u for u in urls if u not in ALLOWED_URLS}
    assert not unexpected, f"{path.name} references other hosts: {sorted(unexpected)}"


@pytest.mark.parametrize("module", ALL_MODULES, ids=lambda p: p.relative_to(STATIC).as_posix())
def test_no_eval_like_constructs(module: Path) -> None:
    code = _code(module)
    assert not re.search(r"(?<![\w$.])eval\s*\(", code), f"{module.name} uses eval (blocked by the CSP)"
    assert "new Function" not in code, f"{module.name} uses new Function (blocked by the CSP)"
    assert not re.search(r"setTimeout\(\s*['\"]", code), f"{module.name} passes a string to setTimeout"


def test_vendored_files_match_the_manifest_and_carry_licences() -> None:
    manifest = json.loads((VENDOR / "VERSIONS.json").read_text(encoding="utf-8"))["files"]
    assert set(manifest) >= {"vendor/preact-htm.mjs", "vendor/lightweight-charts.mjs"}
    for rel, entry in manifest.items():
        digest = hashlib.sha256((STATIC / rel).read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"{rel} differs from the reviewed version {entry['version']}"
        for licence in entry["licence_files"]:
            assert (STATIC / licence).is_file() and (STATIC / licence).stat().st_size > 50, f"missing licence {licence}"


def _object_keys(source: str, name: str) -> set[str]:
    block = re.search(rf"const {name} = \{{(.*?)\}};", source, flags=re.DOTALL)
    assert block, f"{name} not found"
    return set(re.findall(r"(\w+):\s*'", block.group(1)))


def test_german_labels_cover_the_engines_components_and_features() -> None:
    """The signal drawer must never show raw engine identifiers ('market', 'ema_fast_s_slope') to the user."""
    from trading_local.analysis.features import FEATURE_COLUMNS
    from trading_local.analysis.params import DEFAULT_WEIGHTS

    labels = (STATIC / "lib" / "labels.js").read_text(encoding="utf-8")
    assert _object_keys(labels, "COMPONENT_LABELS") == {name for name, _ in DEFAULT_WEIGHTS}
    missing = set(FEATURE_COLUMNS) - {"data_ts"} - _object_keys(labels, "FEATURE_LABELS")
    assert not missing, f"features without a German label: {sorted(missing)}"


def test_reason_codes_are_never_shown_raw() -> None:
    """INT-8: every place that shows a signal's shadow_reason / status_reason goes through the German labels."""
    for path in OWN_JS:
        if path.name in ("labels.js", "types.js"):
            continue
        code = _code(path)
        for match in re.finditer(r"\.(shadow_reason|status_reason)\b", code):
            line = code[code.rfind("\n", 0, match.start()) + 1: code.find("\n", match.end())]
            if path.name == "depot.js":  # order status reasons are German texts of the broker
                assert match.group(1) == "status_reason" and "o.status_reason" in line, line
                continue
            label = "shadowReasonLabel(" if match.group(1) == "shadow_reason" else "statusReasonLabel("
            raw = re.sub(r"(shadowReasonLabel|statusReasonLabel)\([^)]*\)", "", line)
            # a bare test such as `s.shadow_reason ? ... : ''` is fine, printing the code is not
            printed = re.search(rf"\$\{{[^}}]*\.{match.group(1)}\b(?!\s*\?)[^}}]*\}}", raw)
            assert label in line and not printed, f"{path.name}: raw {match.group(1)} in: {line.strip()}"


def test_r_results_are_labelled_without_order_fees() -> None:
    """MONEY-4 / e3: tracked and validated R include slippage and the percentage commission, not the flat fee."""
    for name in ("views/signals.js", "views/validation.js"):
        text = (STATIC / name).read_text(encoding="utf-8")
        assert "nach Kosten" not in text and "Kosten." not in text, name
        assert "nach Slippage und Provision, ohne feste Ordergebühren" in text, name


def test_settings_form_covers_every_runtime_setting_with_its_bounds() -> None:
    """Every RuntimeSettings field has a control, and the client-side bounds equal the pydantic constraints."""
    from annotated_types import Ge, Gt, Le

    from trading_local.config import RuntimeSettings

    source = (STATIC / "views" / "settings.js").read_text(encoding="utf-8")
    fields = {m.group(1): m.group(0) for m in re.finditer(r"\{ key: '(\w+)'[^}]*\}", source)}
    assert set(fields) == set(RuntimeSettings.model_fields), "settings form and RuntimeSettings differ"
    def number(spec: str, key: str) -> float | None:
        match = re.search(rf"\b{key}: ([\d.]+)", spec)
        return float(match.group(1)) if match else None

    for name, info in RuntimeSettings.model_fields.items():
        spec = fields[name]
        for bound in info.metadata:
            if isinstance(bound, Le):
                assert number(spec, "max") == bound.le, f"{name}: max differs"
            elif isinstance(bound, Ge):
                assert number(spec, "min") == bound.ge and "minExclusive" not in spec, f"{name}: min differs"
            elif isinstance(bound, Gt):
                assert number(spec, "min") == bound.gt and "minExclusive: true" in spec, f"{name}: exclusive min differs"


def test_lightweight_charts_attribution_is_shown() -> None:
    """The library's logo is disabled (it injects an inline <style>), so the system page must name TradingView."""
    system_view = (STATIC / "views" / "system.js").read_text(encoding="utf-8")
    assert "TradingView Lightweight Charts" in system_view and "https://www.tradingview.com/" in system_view
    assert "attributionLogo: false" in (STATIC / "lib" / "charts.js").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_own_modules_parse_as_javascript() -> None:
    node = shutil.which("node")
    for module in OWN_JS:
        proc = subprocess.run([node, "--check", str(module)], capture_output=True, text=True, timeout=30, check=False)
        assert proc.returncode == 0, f"{module.name}: {proc.stderr.strip().splitlines()[:5]}"
