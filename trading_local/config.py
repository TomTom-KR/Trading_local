"""Configuration.

Three layers:

* ``Settings``         static configuration from environment variables / the ``.env`` file (prefix ``TL_``).
                       Contains secrets (SMTP password) - never log or return it through the API.
* ``CostParams``       trading costs. The slippage is used IDENTICALLY by the paper broker, the signal tracker
                       and the validation replay, so that simulated, tracked and validated results are
                       comparable. The flat order fee depends on the position size and is therefore charged
                       only by the paper broker; results in R (signal journal, validation) are after slippage
                       but before order fees.
* ``RuntimeSettings``  preferences the user can change in the GUI while the platform runs. Persisted as
                       JSON in the ``settings`` table (key ``runtime``).

Strategy parameters (indicator periods, setup rules, score weights) deliberately do NOT live here and are
not user-tunable: they are fixed textbook defaults inside ``trading_local.analysis`` and are versioned
(``Engine.version``). Tuning them on the stored history would be exactly the overfitting this platform
is designed to avoid.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from trading_local.models import Instrument, Region

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TL_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage -------------------------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"  # database, mail outbox, logs

    # --- web service ---------------------------------------------------------------------------
    host: str = "127.0.0.1"  # keep the service on loopback unless you add your own access protection
    port: int = 8000
    open_browser: bool = True

    # --- local LLM (Ollama) --------------------------------------------------------------------
    ollama_host: str = ""  # empty: library default (http://127.0.0.1:11434)
    ollama_model: str = "qwen3.5:9b"
    # ONE context size for ALL job types and equal to the server's OLLAMA_CONTEXT_LENGTH: any difference
    # between requests (or to other programs using the model) reloads the model (about 5 s each time).
    # 4096 holds every job (largest: trade review with chart image, ~1,500 prompt + ~210 output tokens).
    ollama_num_ctx: int = 4096
    # None lets Ollama place the layers itself (at num_ctx 4096 the whole model fits on the 8 GB GPU).
    # Set 99 to force full GPU offload when using a larger num_ctx.
    ollama_num_gpu: int | None = None
    ollama_temperature: float = 0.1  # used for every job type (retries of rejected answers add 0.1)
    ollama_keep_alive: str = "30m"

    # --- e-mail alerts -------------------------------------------------------------------------
    mail_mode: Literal["smtp", "file", "off"] = "file"  # 'file' writes .eml files to data/outbox
    smtp_host: str = "mail.gmx.net"
    smtp_port: int = 465
    smtp_security: Literal["ssl", "starttls"] = "ssl"
    smtp_user: str = ""
    smtp_password: SecretStr = SecretStr("")
    mail_from: str = ""
    mail_to: str = ""

    # --- simulation ----------------------------------------------------------------------------
    base_currency: str = "EUR"
    start_capital: float = 100_000.0

    # --- behaviour -----------------------------------------------------------------------------
    timezone: str = "Europe/Berlin"  # the user's time zone: day boundaries, times in mails
    offline: bool = False  # True: no background jobs, no network - serve the existing database only
    enable_stream: bool = True
    enable_news: bool = True
    enable_llm: bool = True
    log_level: str = "INFO"

    # --- derived paths -------------------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "trading.sqlite"

    @property
    def outbox_dir(self) -> Path:
        return self.data_dir / "outbox"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def mail_configured(self) -> bool:
        if self.mail_mode == "off":
            return False
        if self.mail_mode == "file":
            return bool(self.mail_to)
        return bool(self.smtp_host and self.smtp_user and self.smtp_password.get_secret_value() and self.mail_to)

    @property
    def sender(self) -> str:
        return self.mail_from or self.smtp_user

    @property
    def is_loopback(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (read once). Tests build their own ``Settings(...)`` instead."""
    return Settings()


# ------------------------------------------------------------------------------------------------
# Costs
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CostParams:
    """Trading-cost assumptions (defaults are deliberately a little pessimistic).

    ``slippage`` is the ONE-WAY cost of crossing half the spread plus market impact, in basis points of the
    price. It is applied against the trader on every fill: buys fill higher, sells fill lower.

    Exchange-traded crypto (instruments of the pseudo venue ``KRAKEN``) has no flat fee but a proportional
    taker fee. A proportional fee does not depend on the position size, so it is modelled as part of the
    one-way cost (``slippage_bps`` = half spread + taker fee) and therefore appears identically in the paper
    broker, the signal tracker, the validation replay and the expected value; ``order_fee`` is 0 for them.
    """

    slippage_bps_us_equity: float = 2.0
    slippage_bps_eu_equity: float = 4.0
    slippage_bps_asia_equity: float = 6.0
    slippage_bps_index: float = 1.5
    slippage_bps_crypto: float = 5.0
    slippage_bps_other: float = 5.0
    # half spread + impact of EUR pairs on Kraken: measured 2026-11-23 at 0.8-2.7 bp (BTC/ETH/SOL/XRP/LINK/DOGE) and
    # 2.5-4.2 bp (ADA/LTC/AVAX/DOT); a 10k EUR market order moves DOT by up to 18 bp - deliberately pessimistic
    slippage_bps_kraken: float = 4.0
    fee_bps_kraken: float = 40.0  # Kraken Pro taker fee of the lowest volume tier (0.40 %)
    fee_flat_base: float = 1.0  # per order, in base currency (typical neo-broker flat fee)
    fee_pct: float = 0.0  # per order, percent of the order value

    def slippage_bps(self, instrument: Instrument) -> float:
        if instrument.exchange == "KRAKEN":
            return self.slippage_bps_kraken + self.fee_bps_kraken
        if instrument.asset_class == "index":
            return self.slippage_bps_index
        if instrument.asset_class == "crypto":
            return self.slippage_bps_crypto
        if instrument.asset_class == "equity":
            if instrument.region == "US":
                return self.slippage_bps_us_equity
            if instrument.region == "EU":
                return self.slippage_bps_eu_equity
            if instrument.region == "ASIA":
                return self.slippage_bps_asia_equity
        return self.slippage_bps_other

    def fill_price(self, instrument: Instrument, price: float, *, buying: bool) -> float:
        """Price after slippage for a market order that buys (``buying=True``) or sells."""
        factor = self.slippage_bps(instrument) / 10_000.0
        return price * (1.0 + factor) if buying else price * (1.0 - factor)

    def round_trip_cost_r(self, instrument: Instrument, price: float, stop_dist: float) -> float:
        """Slippage of entry plus exit, expressed in R (multiples of the stop distance)."""
        if stop_dist <= 0:
            return 0.0
        return 2.0 * self.slippage_bps(instrument) / 10_000.0 * price / stop_dist

    def order_fee(self, order_value_base: float, instrument: Instrument | None = None) -> float:
        """Fee of one order in base currency (0 for KRAKEN instruments: their fee is in ``slippage_bps``)."""
        if instrument is not None and instrument.exchange == "KRAKEN":
            return 0.0
        return self.fee_flat_base + abs(order_value_base) * self.fee_pct / 100.0


DEFAULT_COSTS = CostParams()


# ------------------------------------------------------------------------------------------------
# Runtime settings (editable in the GUI)
# ------------------------------------------------------------------------------------------------
ALL_REGIONS: list[Region] = ["ASIA", "EU", "US", "GLOBAL"]


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # --- e-mail alerts -------------------------------------------------------------------------
    mail_enabled: bool = True
    mail_regions: list[Region] = Field(default_factory=lambda: list(ALL_REGIONS))
    mail_max_per_hour: int = Field(default=12, ge=0, le=120)
    mail_symbol_cooldown_min: int = Field(default=30, ge=0, le=1440)
    mail_on_exit: bool = True  # also mail when a recommended position should be closed
    mail_attach_chart: bool = True

    # --- signal engine -------------------------------------------------------------------------
    signals_enabled: bool = True
    signal_regions: list[Region] = Field(default_factory=lambda: list(ALL_REGIONS))
    disabled_setups: list[str] = Field(default_factory=list)
    allow_long: bool = True
    allow_short: bool = True
    min_score_review: float = Field(default=0.52, ge=0.0, le=1.0)  # candidates below are tracked as shadow only
    min_score_actionable: float = Field(default=0.60, ge=0.0, le=1.0)
    llm_review: bool = True  # let the LLM confirm / reduce / veto candidates
    llm_review_timeout_s: int = Field(default=45, ge=5, le=300)
    signal_cooldown_min: int = Field(default=30, ge=0, le=1440)  # per symbol after a signal closed

    # --- AI portfolio risk ---------------------------------------------------------------------
    risk_per_trade_pct: float = Field(default=0.5, gt=0.0, le=5.0)  # equity risked between entry and stop
    max_position_pct: float = Field(default=10.0, gt=0.0, le=100.0)  # position value cap, percent of equity
    max_open_positions: int = Field(default=10, ge=1, le=100)
    max_gross_leverage: float = Field(default=1.0, gt=0.0, le=5.0)  # (long + short exposure) / equity, both portfolios

    # --- LLM background work -------------------------------------------------------------------
    news_llm_enabled: bool = True
    chart_scan_enabled: bool = True
    chart_scan_interval_min: int = Field(default=30, ge=5, le=1440)  # per symbol, while its venue is open

    # --- comparison ----------------------------------------------------------------------------
    benchmarks: list[str] = Field(default_factory=lambda: ["^GDAXI", "^GSPC"])


RUNTIME_SETTINGS_KEY = "runtime"
