"""Publication dates: the probe's constructed cases (feedparser mis-reads several of them)."""

import pytest

from trading_local.news.fetcher import parse_published

CASES = [
    # raw, default zone, expected UTC (None = unreadable), note
    ("Sun, 22 Nov 2026 16:36:15 +0100", "UTC", "2026-11-22T15:36:15", "RFC 822 numeric offset"),
    ("Sun, 22 Nov 2026 15:00:00 GMT", "UTC", "2026-11-22T15:00:00", "GMT"),
    ("Thu, 1 Oct 2026 19:00:00 GMT", "UTC", "2026-10-01T19:00:00", "one-digit day (Fed)"),
    ("Mon, 16 Nov 2026 15:30 GMT", "UTC", "2026-11-16T15:30:00", "no seconds (GlobeNewswire)"),
    ("Wed, 28 Oct 2026 14:16:49 Z", "UTC", "2026-10-28T14:16:49", "RFC 822 with Z (BoE)"),
    ("Wed, 18 Nov 2026 16:27:09 +09:00", "UTC", "2026-11-18T07:27:09", "colon offset (Korea Herald)"),
    ("Tue, 13 Jan 2026 12:00:00 EST", "UTC", "2026-01-13T17:00:00", "EST in winter"),
    ("Fri, 04 Sep 2026  09:00:00 EST", "UTC", "2026-09-04T13:00:00", "'EST' in summer (EIA): New York wall clock"),
    ("Thu, 18 Apr 2013 08:30:00 EDT", "UTC", "2013-04-18T12:30:00", "EDT"),
    ("Mon, 21 Sep 26 07:30:00 +0100", "UTC", "2026-09-21T06:30:00", "two-digit year (ifo)"),
    ('Wed, 30 Sep 26 <time datetime="2026-09-30T07:20:00+02:00">07:20:00 +0200</time>', "UTC",
     "2026-09-30T05:20:00", "HTML inside pubDate (ifo)"),
    ("Fri, 20 Nov 2026 12:35:00", "Asia/Kolkata", "2026-11-20T07:05:00", "no zone: feed default (RBI)"),
    ("Fri, 09 Oct  09:00:00 EST", "UTC", None, "year missing (EIA)"),
    ("2026-11-22 16:42:20", "UTC", "2026-11-22T16:42:20", "naive ISO (investing.com = UTC)"),
    ("2026-11-22 21:18:20", "Asia/Seoul", "2026-11-22T12:18:20", "naive ISO with feed default zone"),
    ("2026-11-16 11:00:00.0", "Europe/Luxembourg", "2026-11-16T10:00:00", "fraction + naive"),
    ("2026-11-19T19:46:51+01:00", "UTC", "2026-11-19T18:46:51", "ISO offset"),
    ("2026-11-19T12:43:00Z", "UTC", "2026-11-19T12:43:00", "ISO Z"),
    ("2025-12-02T15:01:15.000Z", "UTC", "2025-12-02T15:01:15", "ISO millis Z"),
    ("2026-07-14T07:50:27.969-04:00", "UTC", "2026-07-14T11:50:27", "ISO millis offset (BLS)"),
    ("2026-10-14T07:50:31.2-04:00", "UTC", "2026-10-14T11:50:31", "ISO one-digit fraction (BLS)"),
    ("2026-11-06T10:45+11:00", "UTC", "2026-11-05T23:45:00", "ISO without seconds (RBA)"),
    ("2017-03-25", "Asia/Shanghai", "2017-03-24T16:00:00", "date only"),
    ("Sun, 22 Nov 2026 13:17:25 -0000", "UTC", "2026-11-22T13:17:25", "-0000"),
    ("Mon, 23 Nov 2026 09:00:00 CET", "UTC", "2026-11-23T08:00:00", "CET (feedparser: UTC)"),
    ("Mon, 20 Jul 2026 09:00:00 MESZ", "UTC", "2026-07-20T07:00:00", "MESZ"),
    ("Mon, 23 Nov 2026 09:00:00 CST", "UTC", "2026-11-23T15:00:00", "CST = US Central"),
    ("November 20, 2026 4:15 PM ET", "UTC", "2026-11-20T21:15:00", "US long form"),
    ("20. Nov. 2026 14:30", "Europe/Berlin", "2026-11-20T13:30:00", "German style, feed zone"),
    ("Mo, 23 Dez 2026 09:00:00 +0100", "UTC", "2026-12-23T08:00:00", "German month name"),
    ("Sun Nov 22 16:00:00 2026", "UTC", "2026-11-22T16:00:00", "asctime (feedparser fallback)"),
    ("", "UTC", None, "empty"),
    ("gestern", "UTC", None, "garbage"),
    ("Mon, 23 Nov 1990 09:00:00 GMT", "UTC", None, "implausible year"),
]


@pytest.mark.parametrize(("raw", "zone", "expected", "note"), CASES, ids=[case[3] for case in CASES])
def test_parse_published(raw: str, zone: str, expected: str | None, note: str) -> None:
    parsed = parse_published(raw, zone)
    assert (parsed.strftime("%Y-%m-%dT%H:%M:%S") if parsed else None) == expected
    if parsed is not None:
        assert parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0


def test_unknown_default_zone_gives_none() -> None:
    assert parse_published("2026-11-22 16:42:20", "Mars/Olympus") is None
