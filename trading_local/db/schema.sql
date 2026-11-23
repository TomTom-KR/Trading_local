-- Trading Local - SQLite schema (version 1)
--
-- Conventions
--   * All timestamps are INTEGER epoch seconds in UTC unless a column comment says otherwise.
--   * Bar timestamps are the bar OPEN time. A 1-minute bar with ts = t is closed at t + 60.
--   * Prices are stored exactly as the data source quotes them (London shares in pence).
--     instruments.price_scale converts a quoted price into units of instruments.currency.
--   * JSON columns hold UTF-8 JSON text.
--   * The schema is applied idempotently at start-up; PRAGMA user_version tracks migrations.

-- ---------------------------------------------------------------------------------------------
-- Instruments: watchlist, context symbols (indices, FX, futures) and their trading venue
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS instruments (
    symbol       TEXT PRIMARY KEY,                 -- data-source symbol, e.g. 'SAP.DE', '^GDAXI', 'BTC-USD'
    name         TEXT NOT NULL,
    asset_class  TEXT NOT NULL CHECK (asset_class IN ('equity', 'index', 'crypto', 'fx', 'future')),
    region       TEXT NOT NULL CHECK (region IN ('US', 'EU', 'ASIA', 'GLOBAL')),
    exchange     TEXT NOT NULL,                    -- trading-calendar key: MIC such as 'XNYS', 'XETR', or 'CRYPTO', 'FX', 'CME'
    currency     TEXT NOT NULL,                    -- ISO 4217 code ('GBP' for London even though quotes are in pence)
    price_scale  REAL NOT NULL DEFAULT 1.0,        -- quoted price * price_scale = price in `currency` (0.01 for pence)
    timezone     TEXT NOT NULL,                    -- IANA time zone of the venue
    delay_min    INTEGER NOT NULL DEFAULT 0,       -- delay of the market-data feed in minutes (0 = real time)
    tradable     INTEGER NOT NULL DEFAULT 1 CHECK (tradable IN (0, 1)),   -- gets signals / may be traded in the simulation
    active       INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),     -- data collection on/off
    benchmark    TEXT,                             -- index symbol used as market context, e.g. '^GDAXI'
    aliases      TEXT NOT NULL DEFAULT '[]',       -- JSON list of names used to match news headlines
    sort_order   INTEGER NOT NULL DEFAULT 0
) STRICT, WITHOUT ROWID;

-- ---------------------------------------------------------------------------------------------
-- Market data
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bars_1m (
    symbol  TEXT    NOT NULL,
    ts      INTEGER NOT NULL,                      -- bar open time, epoch seconds UTC, multiple of 60
    open    REAL    NOT NULL,
    high    REAL    NOT NULL,
    low     REAL    NOT NULL,
    close   REAL    NOT NULL,
    volume  REAL    NOT NULL DEFAULT 0,
    source  TEXT    NOT NULL DEFAULT 'rest' CHECK (source IN ('rest', 'stream', 'quote')),
    PRIMARY KEY (symbol, ts)
) STRICT, WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS bars_1d (
    symbol  TEXT NOT NULL,
    day     TEXT NOT NULL,                         -- session date 'YYYY-MM-DD' in the venue's local time
    open    REAL NOT NULL,
    high    REAL NOT NULL,
    low     REAL NOT NULL,
    close   REAL NOT NULL,
    volume  REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, day)
) STRICT, WITHOUT ROWID;

-- Latest quote per symbol (persisted copy of the in-memory quote book)
CREATE TABLE IF NOT EXISTS quotes (
    symbol        TEXT PRIMARY KEY,
    ts            INTEGER NOT NULL,                -- time of the last trade/quote as reported by the source
    received_ts   INTEGER NOT NULL,                -- when this process received it
    price         REAL    NOT NULL,
    bid           REAL,
    ask           REAL,
    prev_close    REAL,
    day_open      REAL,
    day_high      REAL,
    day_low       REAL,
    day_volume    REAL,
    change_pct    REAL,                            -- percent versus prev_close
    market_state  TEXT,                            -- as reported by the source: REGULAR / PRE / POST / CLOSED ...
    source        TEXT NOT NULL                    -- 'stream' | 'quote' | 'rest'
) STRICT, WITHOUT ROWID;

-- ---------------------------------------------------------------------------------------------
-- News
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news (
    id              INTEGER PRIMARY KEY,
    uid             TEXT    NOT NULL UNIQUE,       -- stable dedupe key (hash of guid / link)
    title_key       TEXT    NOT NULL,              -- hash of the normalised title: near-duplicates across sources
    source          TEXT    NOT NULL,              -- feed id
    url             TEXT    NOT NULL,
    title           TEXT    NOT NULL,
    summary         TEXT    NOT NULL DEFAULT '',
    lang            TEXT    NOT NULL DEFAULT 'en',
    published_ts    INTEGER,                       -- as claimed by the feed; may be missing or wrong
    fetched_ts      INTEGER NOT NULL,              -- when WE first saw the item: the as-of time for any analysis
    hinted_symbols  TEXT    NOT NULL DEFAULT '[]', -- JSON list: deterministic alias matches / feed hints
    status          TEXT    NOT NULL DEFAULT 'new'
                    CHECK (status IN ('new', 'queued', 'analyzed', 'skipped', 'duplicate', 'error')),
    status_reason   TEXT    NOT NULL DEFAULT '',
    duplicate_of    INTEGER REFERENCES news (id) ON DELETE SET NULL
) STRICT;
CREATE INDEX IF NOT EXISTS news_fetched_idx ON news (fetched_ts DESC);
CREATE INDEX IF NOT EXISTS news_title_key_idx ON news (title_key, fetched_ts);
CREATE INDEX IF NOT EXISTS news_status_idx ON news (status, fetched_ts);
-- child-key index of the self-reference: without it every deleted news row scans the whole table
CREATE INDEX IF NOT EXISTS news_duplicate_of_idx ON news (duplicate_of) WHERE duplicate_of IS NOT NULL;

CREATE TABLE IF NOT EXISTS news_analysis (
    news_id      INTEGER PRIMARY KEY REFERENCES news (id) ON DELETE CASCADE,
    analyzed_ts  INTEGER NOT NULL,
    model        TEXT    NOT NULL,
    relevance    INTEGER NOT NULL CHECK (relevance BETWEEN 1 AND 5),
    sentiment    REAL    NOT NULL CHECK (sentiment BETWEEN -1.0 AND 1.0),
    confidence   REAL    NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0),
    category     TEXT    NOT NULL,
    horizon      TEXT    NOT NULL,
    summary_de   TEXT    NOT NULL,
    latency_ms   INTEGER
) STRICT;
CREATE INDEX IF NOT EXISTS news_analysis_ts_idx ON news_analysis (analyzed_ts DESC);

-- One row per (news item, affected symbol); analyzed_ts is denormalised for as-of queries
CREATE TABLE IF NOT EXISTS news_symbols (
    news_id      INTEGER NOT NULL REFERENCES news (id) ON DELETE CASCADE,
    symbol       TEXT    NOT NULL,
    analyzed_ts  INTEGER NOT NULL,
    relevance    INTEGER NOT NULL,
    sentiment    REAL    NOT NULL,
    PRIMARY KEY (news_id, symbol)
) STRICT, WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS news_symbols_symbol_idx ON news_symbols (symbol, analyzed_ts DESC);

-- Per-feed polling state (conditional GET, back-off)
CREATE TABLE IF NOT EXISTS feed_state (
    feed_id        TEXT PRIMARY KEY,
    etag           TEXT,
    last_modified  TEXT,
    last_ok_ts     INTEGER,
    last_error     TEXT    NOT NULL DEFAULT '',
    fail_count     INTEGER NOT NULL DEFAULT 0,
    next_poll_ts   INTEGER NOT NULL DEFAULT 0,
    items_total    INTEGER NOT NULL DEFAULT 0
) STRICT, WITHOUT ROWID;

-- ---------------------------------------------------------------------------------------------
-- LLM chart analyses (vision)
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chart_analyses (
    id          INTEGER PRIMARY KEY,
    symbol      TEXT    NOT NULL,
    ts          INTEGER NOT NULL,
    timeframe   TEXT    NOT NULL,                  -- '15m' | '1d'
    model       TEXT    NOT NULL,
    bias        TEXT    NOT NULL CHECK (bias IN ('bullish', 'bearish', 'neutral')),
    strength    INTEGER NOT NULL CHECK (strength BETWEEN 1 AND 5),
    pattern     TEXT    NOT NULL DEFAULT '',
    support     REAL,
    resistance  REAL,
    comment_de  TEXT    NOT NULL,
    latency_ms  INTEGER
) STRICT;
CREATE INDEX IF NOT EXISTS chart_analyses_symbol_idx ON chart_analyses (symbol, ts DESC);

-- ---------------------------------------------------------------------------------------------
-- Signals: the immutable journal of every candidate the engine detected.
-- actionable = 1  -> issued to the user (mail) and to the AI portfolio
-- actionable = 0  -> shadow signal (below threshold, LLM veto ...) that is tracked identically,
--                    so calibration and the LLM's value-add are measured without selection bias.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signals (
    id               INTEGER PRIMARY KEY,
    created_ts       INTEGER NOT NULL,             -- decision time (engine detected the candidate)
    issued_ts        INTEGER,                      -- when review finished and the signal was finalised
    data_ts          INTEGER NOT NULL,             -- close time of the newest bar used (shows data staleness)
    symbol           TEXT    NOT NULL REFERENCES instruments (symbol),
    direction        TEXT    NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
    setup            TEXT    NOT NULL,             -- e.g. 'trend_pullback', 'breakout', 'mean_reversion', 'news_momentum', 'daily_trend'
    timeframe        TEXT    NOT NULL,             -- decision timeframe: '5m' | '15m' | '1d'
    intraday         INTEGER NOT NULL DEFAULT 1 CHECK (intraday IN (0, 1)),  -- 1 = close before the session ends
    horizon_min      INTEGER NOT NULL,             -- maximum holding time in minutes (time stop)
    ref_price        REAL    NOT NULL,             -- last known price at decision time
    atr              REAL    NOT NULL,             -- ATR used for the distances (price units)
    stop_dist        REAL    NOT NULL CHECK (stop_dist > 0),     -- entry -> stop distance (price units)
    target_dist      REAL    NOT NULL CHECK (target_dist > 0),   -- entry -> target distance (price units)
    score            REAL    NOT NULL,             -- deterministic composite score 0..1
    components_json  TEXT    NOT NULL DEFAULT '{}',-- score components 0..1 by name
    features_json    TEXT    NOT NULL DEFAULT '{}',-- feature snapshot used (reproducibility)
    confidence       REAL    NOT NULL,             -- calibrated win probability 0..1 at decision time
    expected_r       REAL    NOT NULL DEFAULT 0,   -- expected value in R after costs at decision time
    llm_verdict      TEXT    CHECK (llm_verdict IN ('confirm', 'reduce', 'veto', 'unavailable')),
    llm_conviction   REAL,
    llm_latency_ms   INTEGER,
    rationale_de     TEXT    NOT NULL DEFAULT '',
    risks_de         TEXT    NOT NULL DEFAULT '',
    news_ids         TEXT    NOT NULL DEFAULT '[]',
    actionable       INTEGER NOT NULL DEFAULT 0 CHECK (actionable IN (0, 1)),
    shadow_reason    TEXT    NOT NULL DEFAULT '',  -- why not actionable: 'below_threshold', 'llm_veto', ...
    status           TEXT    NOT NULL
                     CHECK (status IN ('review', 'pending_entry', 'active', 'target', 'stopped', 'expired', 'cancelled')),
    status_reason    TEXT    NOT NULL DEFAULT '',
    entry_deadline_ts INTEGER,                     -- cancel if no entry bar arrived by then
    checked_ts       INTEGER,                      -- open time of the last 1-minute bar processed by the tracker
    entry_ts         INTEGER,
    entry_price      REAL,
    stop_price       REAL,
    target_price     REAL,
    expires_ts       INTEGER,                      -- time stop (set at entry)
    exit_ts          INTEGER,
    exit_price       REAL,
    outcome_r        REAL,                         -- realised result in R multiples after costs
    mfe_r            REAL,                         -- maximum favourable excursion in R
    mae_r            REAL,                         -- maximum adverse excursion in R
    mailed_ts        INTEGER,
    engine_version   TEXT    NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS signals_created_idx ON signals (created_ts DESC);
CREATE INDEX IF NOT EXISTS signals_symbol_idx ON signals (symbol, created_ts DESC);
CREATE INDEX IF NOT EXISTS signals_status_idx ON signals (status);

-- ---------------------------------------------------------------------------------------------
-- Paper trading: two portfolios ('ai' and 'me') with identical rules and costs
-- Cash accounting: opening a long reduces cash by the notional, opening a short increases it.
-- equity = cash + sum(signed_qty * price * price_scale * fx)
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolios (
    id             INTEGER PRIMARY KEY,
    key            TEXT    NOT NULL UNIQUE,        -- 'ai' | 'me'
    name           TEXT    NOT NULL,
    base_currency  TEXT    NOT NULL,
    start_cash     REAL    NOT NULL,
    cash           REAL    NOT NULL,
    fees_paid      REAL    NOT NULL DEFAULT 0,
    created_ts     INTEGER NOT NULL,
    reset_ts       INTEGER NOT NULL                -- start of the current comparison period
) STRICT;

CREATE TABLE IF NOT EXISTS orders (
    id               INTEGER PRIMARY KEY,
    portfolio_id     INTEGER NOT NULL REFERENCES portfolios (id),
    created_ts       INTEGER NOT NULL,
    symbol           TEXT    NOT NULL REFERENCES instruments (symbol),
    intent           TEXT    NOT NULL CHECK (intent IN ('open', 'close')),
    side             TEXT    NOT NULL CHECK (side IN ('LONG', 'SHORT')),   -- side of the position opened / closed
    qty              REAL,                         -- units; for 'open' either qty or notional_base is set
    notional_base    REAL,                         -- amount in base currency to invest (alternative to qty)
    ref_price        REAL,                         -- last known price when the order was placed
    stop_price       REAL,                         -- absolute protective levels (manual orders) ...
    target_price     REAL,
    stop_dist        REAL,                         -- ... or distances applied to the fill price (signal orders)
    target_dist      REAL,
    max_drift        REAL,                         -- reject the fill if |fill - ref_price| exceeds this (price units)
    time_stop_ts     INTEGER,                      -- position must be closed at/after this time
    flatten_at_close INTEGER NOT NULL DEFAULT 0 CHECK (flatten_at_close IN (0, 1)),
    position_id      INTEGER,                      -- for 'close' orders
    signal_id        INTEGER REFERENCES signals (id),
    origin           TEXT    NOT NULL
                     CHECK (origin IN ('manual', 'signal', 'stop', 'target', 'time_stop', 'session_close', 'reset')),
    status           TEXT    NOT NULL CHECK (status IN ('pending', 'filled', 'cancelled', 'rejected')),
    status_reason    TEXT    NOT NULL DEFAULT '',
    expire_ts        INTEGER,                      -- pending orders are cancelled after this time
    filled_ts        INTEGER,
    filled_qty       REAL,
    fill_price       REAL,                         -- quoted units, after slippage
    fill_fx          REAL,                         -- base currency per unit of instrument currency at the fill
    fee_base         REAL
) STRICT;
CREATE INDEX IF NOT EXISTS orders_portfolio_idx ON orders (portfolio_id, created_ts DESC);
CREATE INDEX IF NOT EXISTS orders_pending_idx ON orders (status) WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS positions (
    id               INTEGER PRIMARY KEY,
    portfolio_id     INTEGER NOT NULL REFERENCES portfolios (id),
    symbol           TEXT    NOT NULL REFERENCES instruments (symbol),
    side             TEXT    NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    qty              REAL    NOT NULL CHECK (qty > 0),
    entry_price      REAL    NOT NULL,             -- quoted units
    entry_fx         REAL    NOT NULL,             -- base currency per unit of instrument currency
    opened_ts        INTEGER NOT NULL,
    stop_price       REAL,
    target_price     REAL,
    time_stop_ts     INTEGER,
    flatten_at_close INTEGER NOT NULL DEFAULT 0 CHECK (flatten_at_close IN (0, 1)),
    signal_id        INTEGER REFERENCES signals (id),
    checked_ts       INTEGER,                      -- open time of the last 1-minute bar checked for stop/target
    status           TEXT    NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    closed_ts        INTEGER,
    exit_price       REAL,
    exit_fx          REAL,
    exit_reason      TEXT,                         -- 'manual' | 'stop' | 'target' | 'time_stop' | 'session_close' | 'reset'
    fees_base        REAL    NOT NULL DEFAULT 0,   -- entry + exit fees
    realized_pnl     REAL                          -- base currency, after fees (set when closed)
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS positions_one_open_idx ON positions (portfolio_id, symbol) WHERE status = 'open';
CREATE INDEX IF NOT EXISTS positions_portfolio_idx ON positions (portfolio_id, status, opened_ts DESC);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    portfolio_id    INTEGER NOT NULL REFERENCES portfolios (id),
    ts              INTEGER NOT NULL,
    equity          REAL    NOT NULL,
    cash            REAL    NOT NULL,
    long_exposure   REAL    NOT NULL DEFAULT 0,    -- base currency
    short_exposure  REAL    NOT NULL DEFAULT 0,    -- base currency, positive number
    open_positions  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (portfolio_id, ts)
) STRICT, WITHOUT ROWID;

-- ---------------------------------------------------------------------------------------------
-- Operations
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT    NOT NULL,                  -- JSON
    updated_ts  INTEGER NOT NULL
) STRICT, WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS mail_log (
    id         INTEGER PRIMARY KEY,
    ts         INTEGER NOT NULL,
    kind       TEXT    NOT NULL,                   -- 'signal' | 'exit' | 'news' | 'test' | 'digest'
    signal_id  INTEGER,
    recipient  TEXT    NOT NULL,
    subject    TEXT    NOT NULL,
    status     TEXT    NOT NULL CHECK (status IN ('sent', 'file', 'skipped', 'failed')),
    detail     TEXT    NOT NULL DEFAULT ''
) STRICT;
CREATE INDEX IF NOT EXISTS mail_log_ts_idx ON mail_log (ts DESC);
CREATE INDEX IF NOT EXISTS mail_log_signal_idx ON mail_log (signal_id) WHERE signal_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS llm_calls (
    id             INTEGER PRIMARY KEY,
    ts             INTEGER NOT NULL,
    kind           TEXT    NOT NULL,               -- 'news' | 'trade_review' | 'chart'
    ok             INTEGER NOT NULL CHECK (ok IN (0, 1)),
    latency_ms     INTEGER NOT NULL,
    queue_wait_ms  INTEGER NOT NULL DEFAULT 0,
    prompt_tokens  INTEGER,
    output_tokens  INTEGER,
    error          TEXT    NOT NULL DEFAULT ''
) STRICT;
CREATE INDEX IF NOT EXISTS llm_calls_ts_idx ON llm_calls (ts DESC);

CREATE TABLE IF NOT EXISTS job_runs (
    id           INTEGER PRIMARY KEY,
    job          TEXT    NOT NULL,
    started_ts   INTEGER NOT NULL,
    duration_ms  INTEGER NOT NULL,
    ok           INTEGER NOT NULL CHECK (ok IN (0, 1)),
    message      TEXT    NOT NULL DEFAULT ''
) STRICT;
CREATE INDEX IF NOT EXISTS job_runs_job_idx ON job_runs (job, started_ts DESC);

CREATE TABLE IF NOT EXISTS activity_log (
    id        INTEGER PRIMARY KEY,
    ts        INTEGER NOT NULL,
    level     TEXT    NOT NULL CHECK (level IN ('info', 'warning', 'error')),
    category  TEXT    NOT NULL,                    -- 'data' | 'news' | 'llm' | 'signal' | 'sim' | 'mail' | 'system'
    message   TEXT    NOT NULL,                    -- German, shown in the GUI
    ref_type  TEXT,
    ref_id    INTEGER
) STRICT;
CREATE INDEX IF NOT EXISTS activity_log_ts_idx ON activity_log (ts DESC);

CREATE TABLE IF NOT EXISTS validation_runs (
    id              INTEGER PRIMARY KEY,
    ts              INTEGER NOT NULL,
    status          TEXT    NOT NULL CHECK (status IN ('running', 'done', 'failed')),
    params_json     TEXT    NOT NULL DEFAULT '{}',
    result_json     TEXT    NOT NULL DEFAULT '{}',
    engine_version  TEXT    NOT NULL,
    message         TEXT    NOT NULL DEFAULT ''
) STRICT;
