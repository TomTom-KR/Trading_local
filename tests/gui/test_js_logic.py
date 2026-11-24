"""Behaviour of the GUI's pure JavaScript helpers, executed with Node.js (skipped without Node).

Covers what a user would get wrong silently: German number input ('1.500' EUR must not become 1,50 EUR),
German output formatting, and the REST client's cancellation and timeout rules (a cancelled request must
never deliver data that arrived after the cancellation; the timeout must also cover a stalled body).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "trading_local" / "web" / "static"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def _run_module(script: str, *, tz: str = "Europe/Berlin") -> object:
    """Run an ES-module snippet with Node and return the JSON it prints last."""
    assert NODE is not None
    env = {**os.environ, "TZ": tz}
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60, check=False, env=env
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _url(rel: str) -> str:
    return (STATIC / rel).as_uri()


def test_parse_decimal_understands_german_thousands_only_where_asked() -> None:
    cases = [
        ("1.500", True, 1500), ("100.000", True, 100000), ("1.234.567", True, 1234567), ("1.500", False, 1.5),
        ("0.500", True, 0.5), ("1.5", True, 1.5), ("12.3456", True, 12.3456), ("1.234,56", False, 1234.56),
        ("1,5", True, 1.5), ("142.35", False, 142.35), (" 2 000 ", True, 2000), ("1.2.3", False, None),
        ("abc", True, None), ("", True, None), ("1,2,3", False, None),
    ]
    script = f"""
        const {{ parseDecimal }} = await import({json.dumps(_url('lib/format.js'))});
        const cases = {json.dumps(cases)};
        console.log(JSON.stringify(cases.map(([t, g]) => {{ const v = parseDecimal(t, {{ grouping: g }}); return Number.isNaN(v) ? null : v; }})));
    """
    assert _run_module(script) == [expected for _, _, expected in cases]


def test_german_output_formats() -> None:
    script = f"""
        const f = await import({json.dumps(_url('lib/format.js'))});
        const ts = Date.UTC(2026, 10, 20) / 1000;  // a daily bar: session date 00:00 UTC
        console.log(JSON.stringify([
            f.fmtNum(-1234.5, 2), f.fmtNum(-0.001, 2), f.fmtPct(58), f.fmtPctSigned(-0.84), f.fmtPctSigned(1.2),
            f.fmtSigned(0.004, 2), f.fmtR(-1), f.fmtMoney(1840, 'EUR', 0), f.priceUnit('GBP', 0.01), f.priceUnit('USD', 1),
            f.fmtDuration(4320), f.fmtDuration(30), f.fmtFlex(0.35), f.fmtFlex(-0.00001), f.fmtSessionDate(ts),
        ]));
    """
    nbsp = " "
    expected = [
        "−1.234,50", "0,00", f"58{nbsp}%", f"−0,8{nbsp}%", f"+1,2{nbsp}%", "0,00", f"−1,0{nbsp}R",
        f"1.840{nbsp}EUR", "GBp", "USD", "1 Std 12 Min", "unter 1 Min", "0,35", "0", "20.11.2026",
    ]
    assert _run_module(script) == expected
    # a daily bar keeps its session date west of UTC, where local midnight would fall on the previous day
    assert _run_module(script, tz="America/Los_Angeles")[-1] == "20.11.2026"


def test_order_quantities_use_the_brokers_lot_sizes() -> None:
    """MONEY-5: an amount order for an index worth more than the amount floored to 0 'Stück' and was blocked."""
    from trading_local.models import Instrument
    from trading_local.sim import sizing

    script = f"""
        const f = await import({json.dumps(_url('lib/format.js'))});
        const classes = ['equity', 'index', 'crypto', 'fx', 'future'];
        console.log(JSON.stringify({{
            decimals: Object.fromEntries(classes.map((c) => [c, f.qtyDecimals(c)])),
            tradable: Object.fromEntries(classes.map((c) => [c, f.isTradableClass(c)])),
            dax: f.floorToLot(2000 / 24803, 'index'),  // 2.000 EUR of the DAX
            exact: f.floorToLot(0.03, 'index'),  // 2.9999999999999996 lots in binary floating point
            shares: [f.floorToLot(2.9999996, 'equity'), f.floorToLot(-1, 'equity'), f.floorToLot(Number.NaN, 'index')],
            btc: f.floorToLot(2000 / 91234.5, 'crypto'),
            text: [f.fmtQty(0.08, 'index'), f.fmtQty(12, 'equity'), f.fmtQty(0.021921, 'crypto'), f.fmtQty(0.5)],
        }}));
    """
    result = _run_module(script)
    for asset_class in ("equity", "index", "crypto", "fx", "future"):
        instrument = Instrument("X", "X", asset_class, "US", "XNYS", "USD")  # type: ignore[arg-type]
        lot = sizing.lot_size(instrument)
        assert result["tradable"][asset_class] is sizing.is_tradable_class(instrument), asset_class
        if lot is not None:
            assert 10 ** -result["decimals"][asset_class] == pytest.approx(lot), asset_class
    index = Instrument("^GDAXI", "DAX", "index", "EU", "XETR", "EUR")
    crypto = Instrument("BTC-USD", "Bitcoin", "crypto", "GLOBAL", "CRYPTO", "USD")
    assert result["dax"] == sizing.round_qty(index, 2000 / 24803) == 0.08
    assert result["btc"] == sizing.round_qty(crypto, 2000 / 91234.5)
    assert result["exact"] == 0.03
    assert result["shares"] == [2, 0, 0]
    assert result["text"] == ["0,08", "12", "0,021921", "0,5"]


def test_instrument_phases_follow_the_session_overview() -> None:
    script = f"""
        const s = await import({json.dumps(_url('lib/sessions.js'))});
        const sessions = {{ regions: [
            {{ region: 'EU', exchanges: [{{ exchange: 'XETR', phase: 'open' }}, {{ exchange: 'XLON', phase: 'open' }}] }},
            {{ region: 'ASIA', exchanges: [{{ exchange: 'XTKS', phase: 'break' }}] }},
        ] }};
        const instruments = [
            {{ symbol: 'SAP.DE', exchange: 'XETR', phase: 'closed', is_open: false }},
            {{ symbol: '7203.T', exchange: 'XTKS', phase: 'break', is_open: false }},
            {{ symbol: 'BTC-USD', exchange: 'CRYPTO', phase: 'open', is_open: true }},
        ];
        const phases = s.phasesFromSessions(sessions);
        const patched = s.applyPhases(instruments, phases);
        const again = s.applyPhases(patched, phases);
        console.log(JSON.stringify({{
            phases, patched, sameWhenUnchanged: again === patched, untouched: patched[1] === instruments[1] && patched[2] === instruments[2],
            closing: s.applyPhases(patched, {{ XETR: 'closed' }})[0], empty: s.phasesFromSessions(null),
        }}));
    """
    result = _run_module(script)
    assert result["phases"] == {"XETR": "open", "XLON": "open", "XTKS": "break"}
    assert result["patched"][0] == {"symbol": "SAP.DE", "exchange": "XETR", "phase": "open", "is_open": True}
    assert result["sameWhenUnchanged"] is True and result["untouched"] is True
    assert result["closing"]["phase"] == "closed" and result["closing"]["is_open"] is False
    assert result["empty"] == {}


RIBBON_HARNESS = """
    const r = await import(%s);
    const T0 = 0, T1 = 86400;  // a 24 h timeline, track 1188 px wide
    const pct = (ts) => ((Math.min(T1, Math.max(T0, ts)) - T0) / (T1 - T0)) * 100;
    const px = (p) => (p / 100) * 1188;
    const H = 3600;
    const widths = { 'Xetra · London · Paris · Amsterdam · Zürich · Kopenhagen': 299, 'New York · Nasdaq': 110,
        'Shanghai · Shenzhen': 122, Hongkong: 68 };
    const widthPx = (t) => widths[t] ?? t.length * 6;
    const place = (segments, names, now, trackPx = 1188) => {
        const p = r.labelPlacement(segments, r.labelCandidates(names), widthPx, pct, now, trackPx);
        return { ...p, leftPx: px(p.left), rightPx: px(p.left + p.width) };
    };
"""


def test_ribbon_labels_sit_on_or_right_next_to_their_bar_and_are_never_clipped() -> None:
    script = RIBBON_HARNESS % json.dumps(_url("lib/ribbon-layout.js")) + """
        const eu = ['Xetra', 'London', 'Paris', 'Amsterdam', 'Zürich', 'Kopenhagen'];
        console.log(JSON.stringify({
            candidates: r.labelCandidates(['Xetra', 'London', 'Paris']),
            // the live screenshot: Xetra 16:40 .. 24:00 (368 px) with the 299 px label, 'jetzt' at 6:00
            europe: place([{ start: 16.67 * H, end: T1, kind: 'regular' }], eu, 6 * H),
            // New York: pre-market 17:30..23:00 hatched, the regular session starts 23:00 and leaves the timeline
            us: place([{ start: 17.5 * H, end: 23 * H, kind: 'pre' }, { start: 23 * H, end: T1, kind: 'regular' }],
                ['New York', 'Nasdaq'], 6 * H),
            // lunch break: two short sessions, the label fits in neither nor in the break between them
            lunch: place([{ start: 10 * H, end: 12 * H, kind: 'regular' }, { start: 13.5 * H, end: 15.5 * H, kind: 'regular' }],
                ['Shanghai', 'Shenzhen'], 6 * H),
            // a narrow track: the full list fits nowhere, a shorter label is used
            narrow: place([{ start: 6 * H, end: 14 * H, kind: 'regular' }, { start: 15 * H, end: 23 * H, kind: 'regular' }],
                eu, 7 * H, 300),
            // nothing fits anywhere (short bars, short gaps, narrow track): the host is the last bar at the right
            // end of the timeline, and the label is shifted left so that it stays inside the timeline
            edge: place([[0, 3], [3.5, 6.5], [7, 10], [10.5, 13.5], [14, 17], [17.5, 20.5], [20.8, 24]]
                .map(([a, b]) => ({ start: a * H, end: b * H, kind: 'regular' })), ['Hongkong'], 0, 300),
        }));
    """
    result = _run_module(script)
    assert result["candidates"] == ["Xetra · London · Paris", "Xetra · London +1", "Xetra +2", "Xetra"]
    europe = result["europe"]  # fits on its bar: no floating label in front of it
    assert europe["outside"] is False and europe["text"].startswith("Xetra · London")
    assert europe["leftPx"] == pytest.approx(16.67 / 24 * 1188, abs=0.5) and europe["rightPx"] <= 1188
    us = result["us"]  # beside the bar (over the hatched pre-market), ending exactly where the session starts
    assert us["text"] == "New York · Nasdaq" and us["outside"] is True and us["align"] == "end"
    assert us["rightPx"] == pytest.approx(23 / 24 * 1188, abs=0.01) and us["leftPx"] >= 0
    lunch = result["lunch"]
    assert lunch["align"] == "end" and lunch["rightPx"] == pytest.approx(10 / 24 * 1188, abs=0.01)
    narrow = result["narrow"]
    assert narrow["text"] != "Xetra · London · Paris · Amsterdam · Zürich · Kopenhagen"
    assert narrow["text"].startswith("Xetra") and narrow["left"] >= 0 and narrow["left"] + narrow["width"] <= 100.0001
    edge = result["edge"]
    assert edge["text"] == "Hongkong" and edge["left"] >= 0 and edge["left"] + edge["width"] == pytest.approx(100)


def test_holding_period_is_shown_in_trading_time() -> None:
    """INT-2: horizon_min counts trading minutes; a five-session swing trade showed as '1 T 18 Std'."""
    script = f"""
        const s = await import({json.dumps(_url('lib/signals.js'))});
        console.log(JSON.stringify([
            s.holdingPeriodText({{ intraday: false, horizon_min: 2550, features: {{ minutes_since_open: 120, minutes_to_close: 390 }} }}),
            s.holdingPeriodText({{ intraday: false, horizon_min: 1950 }}),
            s.holdingPeriodText({{ intraday: false, horizon_min: 300, features: {{ minutes_since_open: 100, minutes_to_close: 200 }} }}),
            s.holdingPeriodText({{ intraday: true, horizon_min: 240 }}),
            s.holdingPeriodText({{ intraday: true }}),
        ]));
    """
    assert _run_module(script) == [
        "ca. 5 Handelstage", "ca. 5 Handelstage", "ca. 1 Handelstag", "4 Std Handelszeit, intraday", "–",
    ]


def test_every_live_minute_bar_reloads_the_daily_candle_even_before_its_utc_midnight() -> None:
    """INT-3 follow-up: a Sydney session (Monday 2026-11-23, 10:00 AEDT = Sunday 23:00 UTC) has its candle at Monday
    00:00 UTC. Its first hour of minute bars opens before that midnight and was dropped, so the candle froze."""
    monday, sunday_2305, friday = 1795392000, 1795388700, 1795132800
    script = f"""
        const b = await import({json.dumps(_url('lib/live-bars.js'))});
        console.log(JSON.stringify({{
            sydneyFirstHour: b.liveBarAction('1d', {sunday_2305}, {monday}),
            sessionStart: b.liveBarAction('1d', {sunday_2305}, {friday}),
            sameBucket: b.liveBarAction('15m', {monday} + 600, {monday}),
            nextBucket: b.liveBarAction('15m', {monday} + 960, {monday}),
            olderBucket: b.liveBarAction('15m', {monday} - 60, {monday}),
            minute: b.liveBarAction('1m', {monday} + 60, {monday}),
        }}));
    """
    result = _run_module(script)
    assert result["sydneyFirstHour"]["action"] == "refresh" and result["sessionStart"]["action"] == "refresh"
    assert result["sameBucket"] == {"action": "merge", "bucket": monday}
    assert result["nextBucket"] == {"action": "merge", "bucket": monday + 900}
    assert result["olderBucket"]["action"] == "ignore"
    assert result["minute"] == {"action": "merge", "bucket": monday + 60}


def test_engine_reason_codes_have_german_labels() -> None:
    """INT-8: the GUI printed 'verworfener Kandidat: erwartungswert' and 'zu 142,35 · time_stop'."""
    from typing import get_args

    from trading_local.analysis import engine, tracking
    from trading_local.execution import ExitReason

    shadow = sorted(value for name, value in vars(engine).items() if name.startswith("SHADOW_"))
    status = sorted([*get_args(ExitReason), tracking.REASON_NO_ENTRY, tracking.REASON_DRIFT])
    script = f"""
        const l = await import({json.dumps(_url('lib/labels.js'))});
        console.log(JSON.stringify({{
            shadow: {json.dumps(shadow)}.map(l.shadowReasonLabel),
            status: {json.dumps(status)}.map(l.statusReasonLabel),
            empty: [l.shadowReasonLabel(''), l.statusReasonLabel(null)],
            unknown: l.statusReasonLabel('neuer_grund'),
            examples: [l.shadowReasonLabel('erwartungswert'), l.statusReasonLabel('time_stop')],
        }}));
    """
    result = _run_module(script)
    assert len(shadow) == 7 and len(status) == 6
    for code, label in [*zip(shadow, result["shadow"], strict=True), *zip(status, result["status"], strict=True)]:
        assert label and label != code and "_" not in label, code
    assert result["empty"] == ["", ""] and result["unknown"] == "neuer_grund"
    assert result["examples"] == ["Erwartungswert nach Slippage nicht positiv", "Haltedauer abgelaufen"]


API_HARNESS = """
    const { api, isAbort, ApiError } = await import(%s);
    const calls = [];
    const respond = (status, body, delayMs = 0) => (url, init) => {
        calls.push({ url, method: init.method, headers: init.headers, body: init.body ?? null });
        return Promise.resolve({
            ok: status < 400, status,
            // the body arrives late and, like a real fetch body, fails once the request is aborted
            text: () => new Promise((resolve, reject) => {
                const t = setTimeout(() => resolve(body), delayMs);
                init.signal.addEventListener('abort', () => { clearTimeout(t); reject(new DOMException('aborted', 'AbortError')); });
            }),
        });
    };
    const outcome = (p) => p.then(
        (data) => ({ ok: true, data }),
        (err) => ({ ok: false, abort: isAbort(err), api: err instanceof ApiError, status: err.status ?? null, message: err.message }),
    );
"""


def _api_script(body: str) -> str:
    return API_HARNESS % json.dumps(_url("lib/api.js")) + body


def test_cancelled_request_never_delivers_late_data() -> None:
    script = _api_script("""
        globalThis.fetch = respond(200, '{"stale": true}', 60);
        const c = new AbortController();
        const p = outcome(api.get('/signals', { scope: 'open' }, { signal: c.signal }));
        setTimeout(() => c.abort(), 10);  // headers are in, the body is still on its way
        const late = await p;
        const c2 = new AbortController();
        c2.abort();
        const pre = await outcome(api.get('/signals', null, { signal: c2.signal }));
        console.log(JSON.stringify({ late, pre, calls: calls.length }));
    """)
    result = _run_module(script)
    assert result["late"]["ok"] is False and result["late"]["abort"] is True
    assert result["pre"]["abort"] is True and result["calls"] == 1, "an already cancelled request must not be sent"


def test_timeout_covers_a_stalled_body_and_reports_german_text() -> None:
    script = _api_script("""
        globalThis.fetch = respond(200, '{}', 10000);
        const r = await outcome(api.get('/status', null, { timeout: 30 }));
        console.log(JSON.stringify(r));
    """)
    result = _run_module(script)
    assert result["ok"] is False and result["api"] is True and result["abort"] is False
    assert result["status"] == 0 and "antwortet nicht" in result["message"]


def test_errors_carry_the_servers_german_detail_and_writes_send_json() -> None:
    script = _api_script("""
        globalThis.fetch = respond(400, '{"detail": "Kaufkraft reicht nicht."}');
        const rejected = await outcome(api.post('/portfolios/me/orders', { symbol: 'NVDA' }));
        globalThis.fetch = respond(500, 'Internal Server Error');
        const crashed = await outcome(api.del('/portfolios/me/orders/7'));
        globalThis.fetch = () => Promise.reject(new TypeError('Failed to fetch'));
        const offline = await outcome(api.get('/status'));
        console.log(JSON.stringify({ rejected, crashed, offline, calls }));
    """)
    result = _run_module(script)
    assert result["rejected"]["message"] == "Kaufkraft reicht nicht." and result["rejected"]["status"] == 400
    assert result["crashed"]["status"] == 500 and "Serverfehler" in result["crashed"]["message"]
    assert result["offline"]["status"] == 0 and "python -m trading_local" in result["offline"]["message"]
    post, delete = result["calls"]
    assert post["headers"]["Content-Type"] == "application/json" and json.loads(post["body"]) == {"symbol": "NVDA"}
    assert delete["method"] == "DELETE" and delete["headers"]["Content-Type"] == "application/json" and delete["body"] == "{}"


def test_the_24_7_footnote_stays_short_with_ten_kraken_coins() -> None:
    """KRAKEN brings ten coins: the ribbon's 'Rund um die Uhr handelbar' line names four and counts the rest."""
    coins = ["Bitcoin", "Ethereum", "Solana", "XRP", "Cardano", "Dogecoin", "Avalanche", "Chainlink", "Polkadot", "Litecoin"]
    script = f"""
        const f = await import({json.dumps(_url('lib/format.js'))});
        const coins = {json.dumps(coins)};
        console.log(JSON.stringify([f.fmtList(coins), f.fmtList(coins.slice(0, 5)), f.fmtList(coins.slice(0, 2)), f.fmtList([])]));
    """
    assert _run_module(script) == [
        "Bitcoin, Ethereum, Solana, XRP und 6 weitere", "Bitcoin, Ethereum, Solana, XRP, Cardano", "Bitcoin, Ethereum", "",
    ]


def test_agenda_groups_by_local_day_and_formats_published_figures() -> None:
    script = f"""
        const a = await import({json.dumps(_url('lib/agenda.js'))});
        const at = (d, h, m) => Date.UTC(2026, 10, d, h, m) / 1000;
        const events = [
            {{ id: 3, event_ts: at(25, 13, 30) }}, {{ id: 1, event_ts: at(24, 15, 0) }},
            {{ id: 2, event_ts: at(24, 22, 59) }}, {{ id: 4, event_ts: at(24, 23, 30) }},
        ];
        const days = a.groupByDay(events);
        const now = at(24, 12, 0);
        console.log(JSON.stringify({{
            days: days.map((d) => d.events.map((e) => e.id)),
            titles: days.map((d) => a.dayTitle(d.ts, now)),
            figures: ['0.3%', '-2', '103.3', '21.75K', '', '4.719%'].map(a.figure),
        }}));
    """
    result = _run_module(script)  # Berlin: 23:30 UTC is already the 25th
    assert result["days"] == [[1, 2], [4, 3]]
    assert result["titles"] == ["Heute, Di 24.11.2026", "Morgen, Mi 25.11.2026"]
    assert result["figures"] == ["0,3 %", "-2", "103,3", "21,75K", "–", "4,719 %"]
