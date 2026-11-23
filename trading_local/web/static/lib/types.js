// JSDoc types of the API payloads (mirror of trading_local/dto.py and config.RuntimeSettings).
// Types only: this module is referenced from JSDoc comments and never loaded by the browser.
// All *_ts fields are epoch seconds (UTC); prices are in the source's units (see priceUnit()).

/**
 * @typedef {'LONG' | 'SHORT'} Direction
 * @typedef {'US' | 'EU' | 'ASIA' | 'GLOBAL'} Region
 * @typedef {'open' | 'break' | 'pre' | 'post' | 'closed'} SessionPhase
 * @typedef {'review' | 'pending_entry' | 'active' | 'target' | 'stopped' | 'expired' | 'cancelled'} SignalStatus
 * @typedef {'confirm' | 'reduce' | 'veto' | 'unavailable'} LlmVerdict
 * @typedef {[number, number, number, number, number, number]} BarTuple [ts, open, high, low, close, volume]
 * @typedef {[number, number]} Point [ts, value]
 */

/**
 * @typedef {object} QuoteOut
 * @property {string} symbol
 * @property {number} price
 * @property {number} ts
 * @property {number | null} change_pct
 * @property {number | null} prev_close
 * @property {number | null} day_open
 * @property {number | null} day_high
 * @property {number | null} day_low
 * @property {number | null} day_volume
 * @property {number | null} bid
 * @property {number | null} ask
 * @property {string | null} market_state
 * @property {string} source
 */

/**
 * @typedef {object} SignalBrief
 * @property {number} id
 * @property {Direction} direction
 * @property {SignalStatus} status
 * @property {number} confidence
 * @property {number} created_ts
 */

/**
 * @typedef {object} InstrumentOut
 * @property {string} symbol
 * @property {string} name
 * @property {'equity' | 'index' | 'crypto' | 'fx' | 'future'} asset_class
 * @property {Region} region
 * @property {string} exchange
 * @property {string} exchange_name
 * @property {string} currency
 * @property {number} price_scale
 * @property {string} timezone
 * @property {number} delay_min
 * @property {boolean} tradable
 * @property {boolean} active
 * @property {string | null} benchmark
 * @property {number} sort_order
 * @property {SessionPhase} phase
 * @property {boolean} is_open
 * @property {QuoteOut | null} quote
 * @property {SignalBrief | null} open_signal
 */

/**
 * @typedef {object} BarsOut
 * @property {string} symbol
 * @property {'1m' | '5m' | '15m' | '1h' | '1d'} timeframe
 * @property {string} currency
 * @property {number} price_scale
 * @property {number} delay_min
 * @property {BarTuple[]} bars
 * @property {Record<string, Point[]>} overlays 'ema20' | 'ema50' | 'ema200' | 'vwap'
 */

/**
 * @typedef {object} ExchangeSessionOut
 * @property {string} exchange
 * @property {string} name
 * @property {Region} region
 * @property {string} timezone
 * @property {SessionPhase} phase
 * @property {boolean} is_open
 * @property {number | null} since
 * @property {number | null} until
 * @property {number | null} next_open
 * @property {number | null} next_close
 * @property {string} note
 *
 * @typedef {object} RegionSessionOut
 * @property {Region} region
 * @property {string} label
 * @property {boolean} is_open
 * @property {ExchangeSessionOut[]} exchanges
 *
 * @typedef {object} TimelineSegmentOut
 * @property {string} exchange
 * @property {Region} region
 * @property {number} start
 * @property {number} end
 * @property {'regular' | 'pre' | 'post'} kind
 *
 * @typedef {object} SessionsOut
 * @property {number} now
 * @property {RegionSessionOut[]} regions
 * @property {number} timeline_start
 * @property {number} timeline_end
 * @property {TimelineSegmentOut[]} timeline
 */

/**
 * @typedef {object} NewsAnalysisOut
 * @property {number} analyzed_ts
 * @property {number} relevance 1..5
 * @property {number} sentiment -1..1
 * @property {number} confidence
 * @property {string} category
 * @property {string} horizon
 * @property {string} summary_de
 * @property {Array<{symbol: string, sentiment: number}>} symbols
 *
 * @typedef {object} NewsOut
 * @property {number} id
 * @property {string} source
 * @property {string} source_name
 * @property {string} url
 * @property {string} title
 * @property {string} summary
 * @property {string} lang
 * @property {number | null} published_ts
 * @property {number} fetched_ts
 * @property {string} status
 * @property {string[]} hinted_symbols
 * @property {NewsAnalysisOut | null} analysis
 */

/**
 * @typedef {object} ReadingOut
 * @property {string} key
 * @property {string} label_de
 * @property {string} value_de
 * @property {'bullish' | 'bearish' | 'neutral'} stance
 *
 * @typedef {object} LevelOut
 * @property {number} price
 * @property {string} kind
 * @property {string} label_de
 *
 * @typedef {object} ChartAnalysisOut
 * @property {number} id
 * @property {string} symbol
 * @property {number} ts
 * @property {string} timeframe
 * @property {string} model
 * @property {'bullish' | 'bearish' | 'neutral'} bias
 * @property {number} strength
 * @property {string} pattern
 * @property {number | null} support
 * @property {number | null} resistance
 * @property {string} comment_de
 * @property {number | null} latency_ms
 *
 * @typedef {object} AnalysisOut
 * @property {string} symbol
 * @property {number | null} as_of
 * @property {number | null} data_ts
 * @property {boolean} available
 * @property {string} note_de
 * @property {Record<string, number | null>} features
 * @property {ReadingOut[]} readings
 * @property {LevelOut[]} levels
 * @property {number | null} news_score
 * @property {ChartAnalysisOut | null} chart_analysis
 */

/**
 * @typedef {object} SignalOut
 * @property {number} id
 * @property {number} created_ts
 * @property {number | null} issued_ts
 * @property {number} data_ts
 * @property {string} symbol
 * @property {string} name
 * @property {Region} region
 * @property {string} currency
 * @property {number} delay_min
 * @property {Direction} direction
 * @property {string} setup
 * @property {string} setup_label_de
 * @property {string} timeframe
 * @property {boolean} intraday
 * @property {number} horizon_min
 * @property {number} ref_price
 * @property {number} atr
 * @property {number} stop_dist
 * @property {number} target_dist
 * @property {number} rr
 * @property {number} score
 * @property {number} confidence
 * @property {number} expected_r
 * @property {LlmVerdict | null} llm_verdict
 * @property {number | null} llm_conviction
 * @property {string} rationale_de
 * @property {string} risks_de
 * @property {boolean} actionable
 * @property {string} shadow_reason
 * @property {SignalStatus} status
 * @property {string} status_reason
 * @property {number | null} entry_ts
 * @property {number | null} entry_price
 * @property {number | null} stop_price
 * @property {number | null} target_price
 * @property {number | null} expires_ts
 * @property {number | null} exit_ts
 * @property {number | null} exit_price
 * @property {number | null} outcome_r
 * @property {number | null} mfe_r
 * @property {number | null} mae_r
 * @property {number | null} mailed_ts
 * @property {string} engine_version
 * @property {number | null} last_price
 * @property {number | null} open_r
 *
 * @typedef {SignalOut & {components: Record<string, number>, features: Record<string, unknown>, news: NewsOut[]}} SignalDetailOut
 */

/**
 * @typedef {object} StatBlock
 * @property {number} n
 * @property {number} wins
 * @property {number | null} win_rate
 * @property {number | null} win_rate_lo
 * @property {number | null} win_rate_hi
 * @property {number | null} avg_r
 * @property {number | null} avg_r_lo
 * @property {number | null} avg_r_hi
 * @property {number} total_r
 * @property {number | null} profit_factor
 *
 * @typedef {{key: string, label_de: string, stats: StatBlock}} GroupStat
 * @typedef {{lo: number, hi: number, n: number, predicted: number, observed: number | null}} CalibrationBin
 *
 * @typedef {object} SignalStatsOut
 * @property {number | null} since_ts
 * @property {string} engine_version
 * @property {StatBlock} actionable
 * @property {StatBlock} shadow
 * @property {GroupStat[]} by_setup
 * @property {GroupStat[]} by_region
 * @property {GroupStat[]} by_direction
 * @property {GroupStat[]} by_llm_verdict
 * @property {CalibrationBin[]} calibration
 * @property {string} note_de
 */

/**
 * @typedef {object} PositionOut
 * @property {number} id
 * @property {string} portfolio_key
 * @property {string} symbol
 * @property {string} name
 * @property {string} currency
 * @property {Direction} side
 * @property {number} qty
 * @property {number} entry_price
 * @property {number} entry_fx
 * @property {number} opened_ts
 * @property {number | null} stop_price
 * @property {number | null} target_price
 * @property {number | null} time_stop_ts
 * @property {boolean} flatten_at_close
 * @property {number | null} signal_id
 * @property {'open' | 'closed'} status
 * @property {number | null} last_price
 * @property {number | null} last_ts
 * @property {number | null} exposure_base
 * @property {number | null} unrealized_pnl
 * @property {number | null} unrealized_pct
 * @property {number | null} closed_ts
 * @property {number | null} exit_price
 * @property {string | null} exit_reason
 * @property {number} fees_base
 * @property {number | null} realized_pnl
 * @property {number | null} realized_pct
 *
 * @typedef {object} OrderOut
 * @property {number} id
 * @property {string} portfolio_key
 * @property {number} created_ts
 * @property {string} symbol
 * @property {string} name
 * @property {'open' | 'close'} intent
 * @property {Direction} side
 * @property {number | null} qty
 * @property {number | null} notional_base
 * @property {number | null} ref_price
 * @property {number | null} stop_price
 * @property {number | null} target_price
 * @property {string} origin
 * @property {'pending' | 'filled' | 'cancelled' | 'rejected'} status
 * @property {string} status_reason
 * @property {number | null} expire_ts
 * @property {number | null} filled_ts
 * @property {number | null} filled_qty
 * @property {number | null} fill_price
 * @property {number | null} fill_fx
 * @property {number | null} fee_base
 * @property {number | null} signal_id
 * @property {number | null} position_id
 *
 * @typedef {object} PortfolioSummaryOut
 * @property {string} key
 * @property {string} name
 * @property {string} base_currency
 * @property {number} start_cash
 * @property {number} cash
 * @property {number} equity
 * @property {number} buying_power
 * @property {number} long_exposure
 * @property {number} short_exposure
 * @property {number} gross_leverage
 * @property {number} unrealized_pnl
 * @property {number} realized_pnl
 * @property {number} fees_paid
 * @property {number} total_return_pct
 * @property {number} day_pnl
 * @property {number} day_return_pct
 * @property {number} open_positions
 * @property {number} pending_orders
 * @property {number} closed_trades
 * @property {number} wins
 * @property {number | null} win_rate
 * @property {number | null} profit_factor
 * @property {number | null} avg_win
 * @property {number | null} avg_loss
 * @property {number | null} max_drawdown_pct
 * @property {number | null} sharpe
 * @property {number} reset_ts
 * @property {number} as_of_ts
 *
 * @typedef {{summary: PortfolioSummaryOut, positions: PositionOut[], pending_orders: OrderOut[]}} PortfolioDetailOut
 *
 * @typedef {object} EquityCurvesOut
 * @property {number} since_ts
 * @property {string} base_currency
 * @property {number} start_cash
 * @property {Point[]} ai
 * @property {Point[]} me
 * @property {Record<string, Point[]>} benchmarks
 */

/**
 * @typedef {object} ValidationResultOut
 * @property {number} period_start_ts
 * @property {number} period_end_ts
 * @property {number} trading_days
 * @property {number} symbols
 * @property {number} decisions
 * @property {StatBlock} overall
 * @property {GroupStat[]} by_setup
 * @property {GroupStat[]} by_region
 * @property {GroupStat[]} by_direction
 * @property {Array<{index: number, start_ts: number, end_ts: number, stats: StatBlock}>} folds
 * @property {CalibrationBin[]} calibration
 * @property {{n_sims: number, mean_avg_r: number, p95_avg_r: number, p_value: number | null} | null} baseline
 * @property {Point[]} equity_r
 * @property {string[]} warnings_de
 * @property {string} verdict_de
 *
 * @typedef {object} ValidationRunOut
 * @property {number} id
 * @property {number} ts
 * @property {'running' | 'done' | 'failed'} status
 * @property {string} engine_version
 * @property {string} message
 * @property {Record<string, unknown>} params
 * @property {ValidationResultOut | null} result
 */

/**
 * @typedef {{id: number, ts: number, level: 'info' | 'warning' | 'error', category: string, message: string, ref_type: string | null, ref_id: number | null}} ActivityOut
 * @typedef {{id: number, ts: number, kind: string, signal_id: number | null, recipient: string, subject: string, status: string, detail: string}} MailLogOut
 * @typedef {{ok: boolean, status: string, detail_de: string}} MailTestOut
 * @typedef {{feed_id: string, name: string, region: string, language: string, category: string, last_ok_ts: number | null, last_error: string, fail_count: number, items_total: number}} FeedStateOut
 * @typedef {{id: number, ts: number, kind: string, ok: boolean, latency_ms: number, queue_wait_ms: number, prompt_tokens: number | null, output_tokens: number | null, error: string}} LlmCallOut
 * @typedef {{job: string, label_de: string, last_started_ts: number | null, last_duration_ms: number | null, ok: boolean, message: string}} JobStatusOut
 */

/**
 * @typedef {object} SystemStatusOut
 * @property {number} server_time
 * @property {number} started_ts
 * @property {string} version
 * @property {string} engine_version
 * @property {string} base_currency
 * @property {boolean} offline
 * @property {{model: string, enabled: boolean, available: boolean, busy: boolean, queue_depth: number, last_latency_ms: number | null, calls_last_hour: number, errors_last_hour: number, last_error: string}} llm
 * @property {{stream_connected: boolean, last_tick_ts: number | null, last_rest_ok_ts: number | null, rate_limited_until: number | null, symbols_active: number, symbols_stale: number, backfill_progress: number | null}} data
 * @property {{feeds_total: number, feeds_failing: number, items_last_hour: number, analyzed_last_hour: number, backlog: number, last_poll_ts: number | null}} news
 * @property {{mode: string, configured: boolean, recipient_masked: string, sent_last_hour: number, last_error: string}} mail
 * @property {{size_mb: number, bars_1m: number, first_bar_ts: number | null, last_bar_ts: number | null, news: number, signals: number}} db
 * @property {JobStatusOut[]} jobs
 */

/**
 * @typedef {object} RuntimeSettings
 * @property {boolean} mail_enabled
 * @property {Region[]} mail_regions
 * @property {number} mail_max_per_hour 0..120
 * @property {number} mail_symbol_cooldown_min 0..1440
 * @property {boolean} mail_on_exit
 * @property {boolean} mail_attach_chart
 * @property {boolean} signals_enabled
 * @property {Region[]} signal_regions
 * @property {string[]} disabled_setups
 * @property {boolean} allow_long
 * @property {boolean} allow_short
 * @property {number} min_score_review 0..1
 * @property {number} min_score_actionable 0..1
 * @property {boolean} llm_review
 * @property {number} llm_review_timeout_s 5..300
 * @property {number} signal_cooldown_min 0..1440
 * @property {number} risk_per_trade_pct (0, 5]
 * @property {number} max_position_pct (0, 100]
 * @property {number} max_open_positions 1..100
 * @property {number} max_gross_leverage (0, 5]
 * @property {boolean} news_llm_enabled
 * @property {boolean} chart_scan_enabled
 * @property {number} chart_scan_interval_min 5..1440
 * @property {string[]} benchmarks
 */

export {};
