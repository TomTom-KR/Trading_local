"""Registry of the news feeds the platform polls.

The feeds are the tier-1 and tier-2 sources vetted by the news probe (147 feeds on 83 hosts, measured on this
machine): USA, Europe/Germany and Asia; companies, markets, macro, press releases and crypto.

* tier 1  fast lane, polled every 60-300 s: wire-speed sources whose small item windows (10-20 items) would
          overflow at slower intervals, the German ad-hoc disclosures (finanznachrichten.de, wallstreet-online:
          there is no EQS/DGAP feed) and the official macro sources (Fed, ECB, BLS, Destatis, Eurostat).
* tier 2  background, polled every 120-1800 s.

``quality`` (1-5) is the probe's editorial rating: items of feeds with quality >= 4 may go to the LLM even
without a matched instrument. ``poll_seconds`` are the probe's measured intervals; together they keep every
host below 5 requests per minute (the scheduler in ``service`` enforces that budget as well).

Not part of the registry (tier 3, deliberately): SEC EDGAR (requires contact details in the User-Agent),
Google News (feed terms: personal, non-commercial reading only) and Nikkei Asia (unreliable, no dates).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit

from trading_local.models import Region

FeedCategory = Literal["markets", "macro", "companies", "press_release", "crypto"]


@dataclass(frozen=True, slots=True)
class Feed:
    """One RSS/Atom feed.

    ``default_tz``     IANA zone for publication dates that carry no zone (e.g. RBI writes local time)
    ``title_strip``    regex removed from every title (publisher prefixes and suffixes)
    ``drop_title_re``  entries whose title matches are dropped (adverts, law-firm class-action spam)
    """

    id: str
    name: str
    url: str
    region: Region
    language: str
    category: FeedCategory
    quality: int  # 1..5
    poll_seconds: int
    enabled: bool = True
    tier: int = 1
    default_tz: str = "UTC"
    title_strip: str | None = None
    drop_title_re: str | None = None

    @property
    def host(self) -> str:
        """Lower-case host name of the feed URL (the unit of the per-host request budget)."""
        return urlsplit(self.url).netloc.lower()


_CNBC = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id="
_DOWJONES = "https://feeds.content.dowjones.io/public/rss/"
_INVESTING = "https://www.investing.com/rss/"
_FN = "https://www.finanznachrichten.de/rss-"
_WO = "https://www.wallstreet-online.de/rss/nachrichten-"
_BLOOMBERG = "https://www.bloomberg.com/feeds/"
_FED = "https://www.federalreserve.gov/feeds/"
_GLOBENEWSWIRE = "https://www.globenewswire.com/RssFeed/"

# Junk inside feeds (titles to drop), see the probe findings.
_SMARTBROKER_AD = r"(?i)^Der neue Online Broker|Smartbroker"  # advert item on top of every wallstreet-online feed
_CLASS_ACTION = (
    r"\b(?:investor (?:alert|deadline|notice|reminder)|deadline (?:alert|approaching)|class action|"
    r"shareholder (?:alert|notice)|encourages .{3,60} investors|reminds .{3,60} investors|securities fraud|"
    r"law offices? of|rosen law|faruqi|pomerantz|levi & korsinsky)\b"
)
_CLASS_ACTION_SPAM = r"(?i)" + _CLASS_ACTION  # US law-firm releases flooding the press-release wires
_WO_ALLE_DROP = r"(?i)(?:^Der neue Online Broker|Smartbroker|" + _CLASS_ACTION + ")"
_ET_QUOTE_PAGES = r"Share Price (?:Highlights|Live Updates)|Stock Price History"  # auto-generated quote pages

FEEDS: Final[tuple[Feed, ...]] = (
    # --- tier 1: fast lane, 60-300 s ------------------------------------------------------------
    Feed("bls_cpi", "BLS CPI", "https://www.bls.gov/feed/cpi.rss", "US", "en", "macro", 5, 300, tier=1),
    Feed(
        "bls_empsit", "BLS employment situation", "https://www.bls.gov/feed/empsit.rss",
        "US", "en", "macro", 5, 300, tier=1,
    ),
    Feed("cnbc_earnings", "CNBC Earnings", _CNBC + "15839135", "US", "en", "companies", 4, 120, tier=1),
    Feed("cnbc_economy", "CNBC Economy", _CNBC + "20910258", "US", "en", "macro", 4, 120, tier=1),
    Feed("cnbc_finance", "CNBC Finance", _CNBC + "10000664", "US", "en", "markets", 4, 120, tier=1),
    Feed("cnbc_technology", "CNBC Technology", _CNBC + "19854910", "US", "en", "companies", 4, 120, tier=1),
    Feed("cnbc_top", "CNBC Top News", _CNBC + "100003114", "US", "en", "markets", 4, 60, tier=1),
    Feed(
        "fed_press_all", "Federal Reserve press releases", _FED + "press_all.xml",
        "US", "en", "macro", 5, 60, tier=1,
    ),
    Feed(
        "fed_press_monetary", "Federal Reserve monetary policy press", _FED + "press_monetary.xml",
        "US", "en", "macro", 5, 60, tier=1,
    ),
    Feed("mw_bulletins", "MarketWatch Bulletins", _DOWJONES + "mw_bulletins", "US", "en", "markets", 4, 60, tier=1),
    Feed(
        "mw_topstories", "MarketWatch Top Stories", _DOWJONES + "mw_topstories",
        "US", "en", "markets", 3, 120, tier=1,
    ),
    Feed(
        "nasdaq_halts", "Nasdaq Trader trade halts", "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts",
        "US", "en", "markets", 4, 60, tier=1,
    ),
    Feed(
        "sa_market_currents", "Seeking Alpha Market Currents", "https://seekingalpha.com/market_currents.xml",
        "US", "en", "markets", 5, 60, tier=1,
    ),
    Feed("wsj_business", "WSJ US Business", _DOWJONES + "WSJcomUSBusiness", "US", "en", "companies", 4, 180, tier=1),
    Feed("wsj_markets", "WSJ Markets", _DOWJONES + "RSSMarketsMain", "US", "en", "markets", 4, 120, tier=1),
    Feed(
        "bbg_economics", "Bloomberg Economics", _BLOOMBERG + "economics/news.rss",
        "GLOBAL", "en", "macro", 5, 60, tier=1,
    ),
    Feed(
        "bbg_industries", "Bloomberg Industries", _BLOOMBERG + "industries/news.rss",
        "GLOBAL", "en", "companies", 4, 120, tier=1,
    ),
    Feed("bbg_markets", "Bloomberg Markets", _BLOOMBERG + "markets/news.rss", "GLOBAL", "en", "markets", 5, 60, tier=1),
    Feed(
        "bbg_technology", "Bloomberg Technology", _BLOOMBERG + "technology/news.rss",
        "GLOBAL", "en", "companies", 4, 120, tier=1,
    ),
    Feed(
        "coindesk", "CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss",
        "GLOBAL", "en", "crypto", 4, 120, tier=1,
    ),
    Feed(
        "financialjuice", "FinancialJuice", "https://www.financialjuice.com/feed.ashx?xy=rss",
        "GLOBAL", "en", "macro", 5, 60, tier=1, title_strip=r"^FinancialJuice:\s*",
    ),
    Feed(
        "forexlive", "investingLive (ForexLive)", "https://investinglive.com/feed/news",
        "GLOBAL", "en", "macro", 5, 60, tier=1,
    ),
    Feed(
        "ft_companies", "FT Companies", "https://www.ft.com/companies?format=rss",
        "GLOBAL", "en", "companies", 4, 180, tier=1,
    ),
    Feed("ft_home", "FT home", "https://www.ft.com/rss/home/international", "GLOBAL", "en", "markets", 4, 120, tier=1),
    Feed(
        "ft_markets", "FT Markets", "https://www.ft.com/markets?format=rss",
        "GLOBAL", "en", "markets", 4, 180, tier=1,
    ),
    Feed(
        "fxstreet_news", "FXStreet news", "https://www.fxstreet.com/rss/news",
        "GLOBAL", "en", "markets", 4, 120, tier=1,
    ),
    Feed(
        "investing_econ_ind", "Investing.com economic indicators", _INVESTING + "news_95.rss",
        "GLOBAL", "en", "macro", 4, 60, tier=1,
    ),
    Feed(
        "investing_economy", "Investing.com economy news", _INVESTING + "news_14.rss",
        "GLOBAL", "en", "macro", 4, 120, tier=1,
    ),
    Feed("investing_news", "Investing.com all news", _INVESTING + "news.rss", "GLOBAL", "en", "markets", 4, 60, tier=1),
    Feed(
        "investing_stock", "Investing.com stock market news", _INVESTING + "news_25.rss",
        "GLOBAL", "en", "markets", 4, 60, tier=1,
    ),
    Feed("theblock", "The Block", "https://www.theblock.co/rss.xml", "GLOBAL", "en", "crypto", 4, 180, tier=1),
    Feed("cash_ch", "cash.ch", "https://www.cash.ch/feeds/latest/news", "EU", "de", "markets", 4, 300, tier=1),
    Feed(
        "ecb_press", "ECB press releases", "https://www.ecb.europa.eu/rss/press.html",
        "EU", "en", "macro", 5, 60, tier=1,
    ),
    Feed(
        "eurostat_prerel", "Eurostat Euro-Indikatoren",
        "https://ec.europa.eu/eurostat/en/search?p_p_id=estatsearchportlet_WAR_estatsearchportlet&p_p_lifecyc"
        "le=2&p_p_state=maximized&p_p_mode=view&p_p_resource_id=atom&_estatsearchportlet_WAR_estatsearchportl"
        "et_collection=CAT_PREREL",
        "EU", "en", "macro", 4, 300, tier=1,
    ),
    Feed(
        "destatis_press", "Destatis Pressemitteilungen",
        "https://www.destatis.de/SiteGlobals/Functions/RSSFeed/DE/RSSNewsfeed/Aktuell.xml",
        "EU", "de", "macro", 4, 300, tier=1,
    ),
    Feed(
        "finanzen_net_news", "finanzen.net News", "https://www.finanzen.net/rss/news",
        "EU", "de", "markets", 5, 60, tier=1,
    ),
    Feed("fn_adhoc", "finanznachrichten.de Ad-hoc", _FN + "aktien-adhoc", "EU", "de", "press_release", 5, 60, tier=1),
    Feed(
        "fn_aktien", "finanznachrichten.de Aktien-Nachrichten", _FN + "aktien-nachrichten",
        "EU", "de", "companies", 4, 60, tier=1,
    ),
    Feed(
        "fn_dax", "finanznachrichten.de DAX (Germany 40)", _FN + "germany-40-nachrichten-1",
        "EU", "de", "companies", 4, 120, tier=1,
    ),
    Feed(
        "hb_finanzen", "Handelsblatt Finanzen", "https://feeds.cms.handelsblatt.com/finanzen",
        "EU", "de", "markets", 4, 300, tier=1,
    ),
    Feed(
        "hb_schlagzeilen", "Handelsblatt Schlagzeilen", "https://feeds.cms.handelsblatt.com/schlagzeilen",
        "EU", "de", "markets", 4, 180, tier=1,
    ),
    Feed(
        "hb_unternehmen", "Handelsblatt Unternehmen", "https://feeds.cms.handelsblatt.com/unternehmen",
        "EU", "de", "companies", 4, 300, tier=1,
    ),
    Feed(
        "onvista_news", "onvista News", "https://www.onvista.de/news/feed/rss.xml",
        "EU", "de", "markets", 4, 300, tier=1,
    ),
    Feed(
        "wo_adhoc", "wallstreet-online Ad-hocs", _WO + "ad-hocs.xml",
        "EU", "de", "press_release", 5, 60, tier=1, drop_title_re=_SMARTBROKER_AD,
    ),
    Feed(
        "wo_unternehmen", "wallstreet-online Unternehmen", _WO + "unternehmen.xml",
        "EU", "de", "companies", 4, 60, tier=1, drop_title_re=_SMARTBROKER_AD,
    ),
    Feed(
        "bt_sg_markets", "Business Times SG companies & markets",
        "https://www.businesstimes.com.sg/rss/companies-markets",
        "ASIA", "en", "markets", 4, 120, tier=1,
    ),
    Feed(
        "cna_business", "Channel NewsAsia Business",
        "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6936",
        "ASIA", "en", "markets", 4, 120, tier=1,
    ),
    Feed(
        "et_markets", "Economic Times Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "ASIA", "en", "markets", 4, 300, tier=1, drop_title_re=_ET_QUOTE_PAGES,
    ),
    Feed(
        "scmp_business", "SCMP Business", "https://www.scmp.com/rss/92/feed/",
        "ASIA", "en", "markets", 4, 300, tier=1,
    ),
    Feed(
        "scmp_companies", "SCMP Companies", "https://www.scmp.com/rss/10/feed/",
        "ASIA", "en", "companies", 4, 300, tier=1,
    ),
    # --- tier 2: background, 120-1800 s ------------------------------------------------------------
    Feed("bea_rss", "BEA news releases", "https://apps.bea.gov/rss/rss.xml", "US", "en", "macro", 4, 300, tier=2),
    Feed("benzinga_feed", "Benzinga", "https://www.benzinga.com/feed", "US", "en", "markets", 3, 300, tier=2),
    Feed(
        "census_indicators", "US Census economic indicators",
        "https://www.census.gov/economic-indicators/indicator.xml",
        "US", "en", "macro", 4, 300, tier=2,
    ),
    Feed("cnbc_investing", "CNBC Investing", _CNBC + "15839069", "US", "en", "markets", 3, 300, tier=2),
    Feed(
        "eia_press", "EIA press releases", "https://www.eia.gov/rss/press_rss.xml",
        "US", "en", "macro", 3, 1800, tier=2,
    ),
    Feed(
        "eia_today", "EIA Today in Energy", "https://www.eia.gov/rss/todayinenergy.xml",
        "US", "en", "macro", 2, 1800, tier=2,
    ),
    Feed("fed_speeches", "Federal Reserve speeches", _FED + "speeches.xml", "US", "en", "macro", 4, 300, tier=2),
    Feed(
        "nyt_business", "NYT Business", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "US", "en", "markets", 3, 300, tier=2,
    ),
    Feed(
        "nyt_economy", "NYT Economy", "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
        "US", "en", "macro", 3, 600, tier=2,
    ),
    Feed(
        "prn_all", "PR Newswire", "https://www.prnewswire.com/rss/news-releases-list.rss",
        "US", "en", "press_release", 2, 120, tier=2, drop_title_re=_CLASS_ACTION_SPAM,
    ),
    Feed(
        "sec_press", "SEC press releases", "https://www.sec.gov/news/pressreleases.rss",
        "US", "en", "macro", 3, 900, tier=2,
    ),
    Feed(
        "stocktitan", "StockTitan news", "https://www.stocktitan.net/rss",
        "US", "en", "press_release", 3, 120, tier=2,
        title_strip=r"\s*\|\s*[A-Z0-9.]{1,8} Stock News$", drop_title_re=_CLASS_ACTION_SPAM,
    ),
    Feed("techcrunch", "TechCrunch", "https://techcrunch.com/feed/", "US", "en", "companies", 3, 300, tier=2),
    Feed("wsj_economy", "WSJ Economy", _DOWJONES + "socialeconomyfeed", "US", "en", "macro", 4, 180, tier=2),
    Feed("wsj_tech", "WSJ Technology", _DOWJONES + "RSSWSJD", "US", "en", "companies", 4, 180, tier=2),
    Feed("bbg_crypto", "Bloomberg Crypto", _BLOOMBERG + "crypto/news.rss", "GLOBAL", "en", "crypto", 3, 300, tier=2),
    Feed(
        "bbg_politics", "Bloomberg Politics", _BLOOMBERG + "politics/news.rss",
        "GLOBAL", "en", "macro", 3, 300, tier=2,
    ),
    Feed("cnbc_world", "CNBC World News", _CNBC + "100727362", "GLOBAL", "en", "markets", 3, 300, tier=2),
    Feed("cointelegraph", "Cointelegraph", "https://cointelegraph.com/rss", "GLOBAL", "en", "crypto", 4, 300, tier=2),
    Feed("cryptoslate", "CryptoSlate", "https://cryptoslate.com/feed/", "GLOBAL", "en", "crypto", 3, 600, tier=2),
    Feed("decrypt", "Decrypt", "https://decrypt.co/feed", "GLOBAL", "en", "crypto", 3, 300, tier=2),
    Feed(
        "ft_global_economy", "FT Global Economy", "https://www.ft.com/global-economy?format=rss",
        "GLOBAL", "en", "macro", 4, 300, tier=2,
    ),
    Feed(
        "gnw_earnings", "GlobeNewswire earnings releases",
        "https://www.globenewswire.com/RssFeed/subjectcode/13-Earnings%20Releases%20And%20Operating%20Results"
        "/feedTitle/GlobeNewswire%20-%20Earnings%20Releases%20And%20Operating%20Results",
        "GLOBAL", "en", "press_release", 3, 300, tier=2,
    ),
    Feed(
        "gnw_public", "GlobeNewswire public companies",
        _GLOBENEWSWIRE + "orgclass/1/feedTitle/GlobeNewswire%20-%20News%20about%20Public%20Companies",
        "GLOBAL", "en", "press_release", 3, 120, tier=2, drop_title_re=_CLASS_ACTION_SPAM,
    ),
    Feed(
        "investing_commod", "Investing.com commodities news", _INVESTING + "news_11.rss",
        "GLOBAL", "en", "markets", 3, 120, tier=2,
    ),
    Feed(
        "investing_crypto", "Investing.com crypto news", _INVESTING + "news_301.rss",
        "GLOBAL", "en", "crypto", 3, 180, tier=2,
    ),
    Feed(
        "investing_forex", "Investing.com forex news", _INVESTING + "news_1.rss",
        "GLOBAL", "en", "markets", 3, 120, tier=2,
    ),
    Feed("mining_com", "Mining.com", "https://www.mining.com/feed/", "GLOBAL", "en", "companies", 3, 900, tier=2),
    Feed("oilprice", "OilPrice.com", "https://oilprice.com/rss/main", "GLOBAL", "en", "markets", 3, 600, tier=2),
    Feed("wsj_world", "WSJ World News", _DOWJONES + "RSSWorldNews", "GLOBAL", "en", "macro", 3, 300, tier=2),
    Feed(
        "bbc_business", "BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml",
        "EU", "en", "macro", 3, 300, tier=2,
    ),
    Feed(
        "boe_news", "Bank of England news", "https://www.bankofengland.co.uk/rss/news",
        "EU", "en", "macro", 4, 300, tier=2,
    ),
    Feed("cityam", "City A.M.", "https://www.cityam.com/feed/", "EU", "en", "markets", 3, 600, tier=2),
    Feed("cnbc_europe", "CNBC Europe News", _CNBC + "19794221", "EU", "en", "markets", 3, 300, tier=2),
    Feed(
        "ecb_statpress", "ECB statistical press releases", "https://www.ecb.europa.eu/rss/statpress.html",
        "EU", "en", "macro", 3, 600, tier=2,
    ),
    Feed("finanzen_ch", "finanzen.ch News", "https://www.finanzen.ch/rss/news", "EU", "de", "markets", 3, 300, tier=2),
    Feed(
        "guardian_business", "Guardian Business", "https://www.theguardian.com/uk/business/rss",
        "EU", "en", "macro", 3, 300, tier=2,
    ),
    Feed(
        "guardian_markets", "Guardian Stock markets", "https://www.theguardian.com/business/stock-markets/rss",
        "EU", "en", "markets", 3, 600, tier=2,
    ),
    Feed(
        "lesechos_marches", "Les Echos Finance & Marchés",
        "https://services.lesechos.fr/rss/les-echos-finance-marches.xml",
        "EU", "fr", "markets", 3, 600, tier=2,
    ),
    Feed("nzz_finanzen", "NZZ Finanzen", "https://www.nzz.ch/finanzen.rss", "EU", "de", "markets", 3, 900, tier=2),
    Feed("nzz_wirtschaft", "NZZ Wirtschaft", "https://www.nzz.ch/wirtschaft.rss", "EU", "de", "macro", 3, 600, tier=2),
    Feed(
        "sky_business", "Sky News business", "https://feeds.skynews.com/feeds/rss/business.xml",
        "EU", "en", "macro", 3, 300, tier=2,
    ),
    Feed(
        "snb_press", "SNB press releases", "https://www.snb.ch/public/rss/en/pressrel",
        "EU", "en", "macro", 4, 600, tier=2,
    ),
    Feed(
        "aktionaer", "Der Aktionär", "https://www.deraktionaer.de/aktionaer-news.rss",
        "EU", "de", "companies", 2, 600, tier=2,
    ),
    Feed(
        "boerse_frankfurt_news", "Börse Frankfurt",
        "https://api.boerse-frankfurt.de/v1/feeds/news.rss",
        "EU", "de", "markets", 3, 600, tier=2,
    ),
    Feed("btc_echo", "BTC-ECHO", "https://www.btc-echo.de/feed/", "EU", "de", "crypto", 3, 300, tier=2),
    Feed(
        "bundesbank_presse", "Bundesbank Presse", "https://www.bundesbank.de/service/rss/de/633286/feed.rss",
        "EU", "de", "macro", 3, 900, tier=2,
    ),
    Feed(
        "bundesbank_reden", "Bundesbank Reden/Interviews", "https://www.bundesbank.de/service/rss/de/633282/feed.rss",
        "EU", "de", "macro", 3, 1800, tier=2,
    ),
    Feed(
        "faz_finanzen", "FAZ Finanzen", "https://www.faz.net/rss/aktuell/finanzen/",
        "EU", "de", "markets", 3, 300, tier=2,
    ),
    Feed(
        "faz_unternehmen", "FAZ Unternehmen", "https://www.faz.net/rss/aktuell/wirtschaft/unternehmen/",
        "EU", "de", "companies", 3, 600, tier=2,
    ),
    Feed(
        "faz_wirtschaft", "FAZ Wirtschaft", "https://www.faz.net/rss/aktuell/wirtschaft/",
        "EU", "de", "macro", 3, 300, tier=2,
    ),
    Feed(
        "finanzen_net_analysen", "finanzen.net Analysen", "https://www.finanzen.net/rss/analysen",
        "EU", "de", "companies", 3, 300, tier=2,
    ),
    Feed(
        "fn_analysen", "finanznachrichten.de Analysen", _FN + "aktien-analysen",
        "EU", "de", "companies", 3, 300, tier=2,
    ),
    Feed(
        "fn_konjunktur", "finanznachrichten.de Wirtschaft/Konjunktur", _FN + "nachrichten-wirtschaft-konjunktur",
        "EU", "de", "macro", 3, 300, tier=2,
    ),
    Feed(
        "fn_marktberichte", "finanznachrichten.de Marktberichte", _FN + "marktberichte",
        "EU", "de", "markets", 3, 300, tier=2,
    ),
    Feed(
        "gnw_germany", "GlobeNewswire country Germany",
        _GLOBENEWSWIRE + "country/Germany/feedTitle/GlobeNewswire%20-%20News%20from%20Germany",
        "EU", "en", "press_release", 3, 300, tier=2,
    ),
    Feed("ifo", "ifo Institut", "https://www.ifo.de/pressemitteilungen/rss.xml", "EU", "de", "macro", 4, 900, tier=2),
    Feed(
        "investing_de_news", "Investing.com DE all news", "https://de.investing.com/rss/news.rss",
        "EU", "de", "markets", 3, 120, tier=2,
    ),
    Feed(
        "investing_de_stock", "Investing.com DE Aktienmarkt", "https://de.investing.com/rss/news_25.rss",
        "EU", "de", "markets", 3, 180, tier=2,
    ),
    Feed(
        "mm_finanzen", "manager magazin Finanzen", "https://www.manager-magazin.de/finanzen/index.rss",
        "EU", "de", "markets", 3, 600, tier=2,
    ),
    Feed(
        "mm_news", "manager magazin News", "https://www.manager-magazin.de/news/index.rss",
        "EU", "de", "companies", 3, 300, tier=2,
    ),
    Feed(
        "ntv_wirtschaft", "n-tv Wirtschaft", "https://www.n-tv.de/wirtschaft/rss",
        "EU", "de", "markets", 3, 180, tier=2,
    ),
    Feed(
        "presseportal_wirtschaft", "Presseportal (ots) Wirtschaft", "https://www.presseportal.de/rss/wirtschaft.rss2",
        "EU", "de", "press_release", 2, 600, tier=2,
    ),
    Feed(
        "spiegel_unternehmen", "Spiegel Unternehmen", "https://www.spiegel.de/wirtschaft/unternehmen/index.rss",
        "EU", "de", "companies", 3, 600, tier=2,
    ),
    Feed(
        "spiegel_wirtschaft", "Spiegel Wirtschaft", "https://www.spiegel.de/wirtschaft/index.rss",
        "EU", "de", "macro", 3, 300, tier=2,
    ),
    Feed(
        "tagesschau_finanzen", "tagesschau Finanzen", "https://www.tagesschau.de/wirtschaft/finanzen/index~rss2.xml",
        "EU", "de", "markets", 3, 600, tier=2,
    ),
    Feed(
        "tagesschau_konjunktur", "tagesschau Konjunktur",
        "https://www.tagesschau.de/wirtschaft/konjunktur/index~rss2.xml",
        "EU", "de", "macro", 3, 900, tier=2,
    ),
    Feed(
        "tagesschau_wirtschaft", "tagesschau Wirtschaft", "https://www.tagesschau.de/wirtschaft/index~rss2.xml",
        "EU", "de", "macro", 3, 180, tier=2,
    ),
    Feed(
        "wiwo_finanzen", "WirtschaftsWoche Finanzen", "https://feeds.cms.wiwo.de/rss/finanzen",
        "EU", "de", "markets", 3, 600, tier=2,
    ),
    Feed(
        "wiwo_schlagzeilen", "WirtschaftsWoche Schlagzeilen", "https://feeds.cms.wiwo.de/rss/schlagzeilen",
        "EU", "de", "markets", 3, 300, tier=2,
    ),
    Feed(
        "wiwo_unternehmen", "WirtschaftsWoche Unternehmen", "https://feeds.cms.wiwo.de/rss/unternehmen",
        "EU", "de", "companies", 3, 600, tier=2,
    ),
    Feed(
        "wo_alle", "wallstreet-online alle Nachrichten", _WO + "alle.xml",
        "EU", "de", "markets", 3, 300, tier=2, drop_title_re=_WO_ALLE_DROP,
    ),
    Feed(
        "wo_dd", "wallstreet-online Directors Dealings", _WO + "directors-dealings.xml",
        "EU", "de", "press_release", 3, 600, tier=2, drop_title_re=_SMARTBROKER_AD,
    ),
    Feed(
        "wo_konjunktur", "wallstreet-online Wirtschaft/Konjunktur", _WO + "wirtschaft-konjunktur.xml",
        "EU", "de", "macro", 3, 180, tier=2, drop_title_re=_SMARTBROKER_AD,
    ),
    Feed(
        "abc_au_business", "ABC Australia business", "https://www.abc.net.au/news/feed/51892/rss.xml",
        "ASIA", "en", "markets", 3, 600, tier=2,
    ),
    Feed(
        "boj_whatsnew", "Bank of Japan", "https://www.boj.or.jp/en/rss/whatsnew.xml",
        "ASIA", "en", "macro", 4, 300, tier=2,
    ),
    Feed(
        "business_standard_mkts", "Business Standard markets", "https://www.business-standard.com/rss/markets-106.rss",
        "ASIA", "en", "markets", 3, 600, tier=2,
    ),
    Feed("cnbc_asia", "CNBC Asia News", _CNBC + "19832390", "ASIA", "en", "markets", 3, 300, tier=2),
    Feed(
        "digitimes", "DigiTimes daily", "https://www.digitimes.com/rss/daily.xml",
        "ASIA", "en", "companies", 4, 600, tier=2,
    ),
    Feed(
        "et_stocks", "Economic Times Stocks",
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "ASIA", "en", "companies", 3, 600, tier=2, drop_title_re=_ET_QUOTE_PAGES,
    ),
    Feed(
        "focustaiwan", "Focus Taiwan (CNA)", "https://feeds.feedburner.com/rsscna/engnews/",
        "ASIA", "en", "markets", 3, 600, tier=2,
    ),
    Feed(
        "hbl_markets", "Hindu BusinessLine markets", "https://www.thehindubusinessline.com/markets/feeder/default.rss",
        "ASIA", "en", "markets", 3, 600, tier=2,
    ),
    Feed(
        "hkex_news", "HKEX news releases", "https://www.hkex.com.hk/Services/RSS-Feeds/News-Releases?sc_lang=en",
        "ASIA", "en", "press_release", 3, 1800, tier=2,
    ),
    Feed(
        "japantimes_business", "Japan Times business", "https://www.japantimes.co.jp/business/feed/",
        "ASIA", "en", "markets", 3, 600, tier=2,
    ),
    Feed(
        "koreaherald_biz", "Korea Herald business", "https://www.koreaherald.com/rss/kh_Business",
        "ASIA", "en", "markets", 3, 600, tier=2,
    ),
    Feed(
        "kyodo_feed", "Kyodo News English", "https://english.kyodonews.net/list/feed/rss4kyodonews-fzone",
        "ASIA", "en", "macro", 3, 300, tier=2,
    ),
    Feed(
        "livemint_companies", "Livemint companies", "https://www.livemint.com/rss/companies",
        "ASIA", "en", "companies", 3, 600, tier=2,
    ),
    Feed(
        "livemint_markets", "Livemint markets", "https://www.livemint.com/rss/markets",
        "ASIA", "en", "markets", 3, 300, tier=2,
    ),
    Feed(
        "nhk_new_cat5", "NHK Business", "https://news.web.nhk/n-data/conf/na/rss/cat5.xml",
        "ASIA", "ja", "markets", 3, 300, tier=2,
    ),
    Feed(
        "rba_media", "RBA media releases", "https://www.rba.gov.au/rss/rss-cb-media-releases.xml",
        "ASIA", "en", "macro", 3, 1800, tier=2,
    ),
    Feed(
        "rbi_press", "RBI press releases", "https://www.rbi.org.in/pressreleases_rss.xml",
        "ASIA", "en", "macro", 3, 1800, tier=2, default_tz="Asia/Kolkata",
    ),
    Feed(
        "rthk_finance", "RTHK finance", "https://rthk.hk/rthk/news/rss/e_expressnews_efinance.xml",
        "ASIA", "en", "markets", 3, 600, tier=2,
    ),
    Feed(
        "scmp_economy", "SCMP Global Economy", "https://www.scmp.com/rss/12/feed/",
        "ASIA", "en", "macro", 3, 600, tier=2,
    ),
    Feed(
        "smh_business", "Sydney Morning Herald business", "https://www.smh.com.au/rss/business.xml",
        "ASIA", "en", "markets", 3, 600, tier=2,
    ),
    Feed(
        "straitstimes_business", "Straits Times business", "https://www.straitstimes.com/news/business/rss.xml",
        "ASIA", "en", "markets", 3, 600, tier=2,
    ),
    Feed(
        "taipeitimes_all", "Taipei Times all", "https://www.taipeitimes.com/xml/index.rss",
        "ASIA", "en", "markets", 2, 1800, tier=2,
    ),
    Feed(
        "yonhap_news", "Yonhap English all news", "https://en.yna.co.kr/RSS/news.xml",
        "ASIA", "en", "macro", 3, 600, tier=2,
    ),
)

FEEDS_BY_ID: Final[Mapping[str, Feed]] = MappingProxyType({feed.id: feed for feed in FEEDS})
if len(FEEDS_BY_ID) != len(FEEDS):
    raise RuntimeError("feed ids must be unique")
