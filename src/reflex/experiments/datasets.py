"""Opt-in, row-oriented public dataset adapters with explicit temporal caveats.

No downloader is included: large archives, terms, and coverage change over time.
Callers provide parsed rows and retain original source metadata in provenance.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from html import unescape
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import re
from typing import Any, Iterable
from urllib.request import urlopen
from zipfile import ZipFile

from reflex.core.models.events import Event, stable_id


GDELT_EVENT_COLUMNS = (
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode",
    "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2EthnicCode",
    "Actor2Religion1Code", "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass", "GoldsteinScale",
    "NumMentions", "NumSources", "NumArticles", "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName",
    "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_Lat",
    "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName",
    "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_Lat",
    "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type", "ActionGeo_FullName",
    "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_Lat",
    "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL",
)


def download_gdelt_day(date: str, *, limit: int = 10_000, timeout_seconds: int = 60) -> tuple[tuple[Event, ...], dict[str, Any]]:
    """Download one official GDELT 2.0 daily event archive and return a sorted sample."""
    if not re.fullmatch(r"\d{8}", date) or limit < 1:
        raise ValueError("date must be YYYYMMDD and limit must be positive")
    source_url = f"https://data.gdeltproject.org/events/{date}.export.CSV.zip"
    with urlopen(source_url, timeout=timeout_seconds) as response:
        archive = response.read()
    digest = sha256(archive).hexdigest()
    with ZipFile(BytesIO(archive)) as zipped:
        members = [name for name in zipped.namelist() if name.lower().endswith((".csv", ".tsv"))]
        if not members:
            raise ValueError("GDELT archive contains no tabular event file")
        with zipped.open(members[0]) as binary:
            reader = csv.reader(TextIOWrapper(binary, encoding="utf-8", errors="replace"), delimiter="\t")
            rows = []
            for row_number, values in enumerate(reader, start=1):
                if not values or not any(values):
                    continue
                if values[0] == "GLOBALEVENTID":
                    continue
                if len(values) != len(GDELT_EVENT_COLUMNS):
                    raise ValueError(
                        f"GDELT row {row_number} has {len(values)} columns; "
                        f"expected {len(GDELT_EVENT_COLUMNS)}"
                    )
                rows.append(dict(zip(GDELT_EVENT_COLUMNS, values, strict=True)))
    events = adapt_gdelt_rows(rows)
    selected = events[:limit]
    return selected, {"dataset": "GDELT Events 2.0", "date": date, "source_url": source_url,
                      "archive_sha256": digest, "archive_bytes": len(archive),
                      "parsed_row_count": len(rows), "selected_event_count": len(selected),
                      "downloaded_at": datetime.now(timezone.utc).isoformat(),
                      "attribution": "Data from the GDELT Project (www.gdeltproject.org)."}


def _time(value: Any, *, date_only: bool = False) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        raw = str(value).strip()
        if raw.endswith(" UTC"):
            raw = raw[:-4] + "+00:00"
        if date_only and re.fullmatch(r"\d{8}", raw):
            result = datetime.strptime(raw, "%Y%m%d").replace(tzinfo=timezone.utc)
        elif re.fullmatch(r"\d{14}", raw):
            result = datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        elif re.fullmatch(r"\d{8}", raw):
            result = datetime.strptime(raw, "%Y%m%d").replace(tzinfo=timezone.utc)
        else:
            try:
                result = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                result = datetime.strptime(raw, "%d-%b-%Y").replace(tzinfo=timezone.utc)
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def adapt_gdelt_rows(rows: Iterable[dict[str, Any]]) -> tuple[Event, ...]:
    """Map GDELT Events 2.0 records; not full article text or market labels."""
    events = []
    for row in rows:
        timestamp = _time(row.get("DATEADDED") or row.get("date") or row.get("SQLDATE"),
                          date_only=not row.get("DATEADDED") and not row.get("date"))
        event_id = str(row.get("GLOBALEVENTID") or stable_id("gdelt", str(timestamp), str(row)))
        events.append(Event(
            event_id=event_id, event_type="news.gdelt_event", event_time=timestamp,
            source="GDELT-Events-2.0", entity_ids=tuple(filter(None, [str(row.get("Actor1Name", "")),
                       str(row.get("Actor2Name", "")), str(row.get("ActionGeo_FullName", ""))])),
            attributes={"event_code": row.get("EventCode"), "event_root_code": row.get("EventRootCode"),
                        "goldstein_scale": _number(row.get("GoldsteinScale")),
                        "average_tone": _number(row.get("AvgTone")),
                        "mention_count": _number(row.get("NumMentions")),
                        "source_url": row.get("SOURCEURL")},
            provenance={"dataset": "GDELT Events 2.0", "global_event_id": event_id},
        ))
    return tuple(sorted(events, key=lambda item: (item.event_time, item.event_id)))


def adapt_fnspid_rows(rows: Iterable[dict[str, Any]]) -> tuple[Event, ...]:
    """Map FNSPID-like news rows; available dates and symbols do not imply causality."""
    result = []
    for index, row in enumerate(rows):
        date = row.get("Date") or row.get("date") or row.get("Time")
        symbol = row.get("Stock_symbol") or row.get("Ticker") or row.get("ticker") or "UNKNOWN"
        title = row.get("Article_title") or row.get("title") or row.get("Headline") or ""
        link = row.get("Url") or row.get("url") or ""
        timestamp = _time(date)
        result.append(Event(event_id=stable_id("fnspid", str(symbol), str(date), str(title), str(index)),
                            event_type="news.financial_article", event_time=timestamp,
                            source="FNSPID", entity_ids=(f"equity:{symbol}",),
                            attributes={"title": str(title), "url": str(link),
                                        "publisher": row.get("Publisher") or row.get("publisher"),
                                        "text": row.get("Article_text") or row.get("text")},
                            provenance={"dataset": "FNSPID", "row_index": index}))
    return tuple(sorted(result, key=lambda item: (item.event_time, item.event_id)))


def adapt_market_price_rows(rows: Iterable[dict[str, Any]], *, symbol_field: str = "symbol",
                            date_field: str = "date", close_field: str = "close",
                            dataset_name: str = "user-supplied-historical-prices") -> tuple[Event, ...]:
    events = []
    for index, row in enumerate(rows):
        symbol = str(row.get(symbol_field, "UNKNOWN"))
        timestamp = _time(row[date_field], date_only=True)
        close = _number(row[close_field])
        events.append(Event(event_id=stable_id("market-price", dataset_name, symbol, str(timestamp), str(index)),
                            event_type="market.close_price", event_time=timestamp,
                            source=dataset_name, entity_ids=(f"instrument:{symbol}",),
                            attributes={"close": close, "adjusted_close": _number(row.get("adjusted_close")),
                                        "volume": _number(row.get("volume"))},
                            provenance={"dataset": dataset_name, "row_index": index}))
    return tuple(sorted(events, key=lambda item: (item.event_time, item.event_id)))


def parse_reuters21578(text: str) -> tuple[dict[str, Any], ...]:
    """Extract basic dated documents from Reuters-21578 SGML for topic tasks."""
    docs: list[dict[str, Any]] = []
    for index, match in enumerate(re.finditer(r"<REUTERS\b([^>]*)>(.*?)</REUTERS>", text, re.I | re.S)):
        attrs, body = match.group(1), match.group(2)
        date_match = re.search(r"<DATE>(.*?)</DATE>", body, re.I | re.S)
        title_match = re.search(r"<TITLE>(.*?)</TITLE>", body, re.I | re.S)
        text_match = re.search(r"<BODY>(.*?)</BODY>", body, re.I | re.S)
        doc_id = re.search(r'NEWID="(\d+)"', attrs, re.I)
        raw_date = _clean(date_match.group(1)) if date_match else ""
        docs.append({"id": (doc_id.group(1) if doc_id else str(index)), "date": raw_date,
                     "title": _clean(title_match.group(1)) if title_match else "",
                     "text": _clean(text_match.group(1)) if text_match else ""})
    return tuple(docs)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["GDELT_EVENT_COLUMNS", "download_gdelt_day", "adapt_gdelt_rows", "adapt_fnspid_rows",
           "adapt_market_price_rows", "parse_reuters21578"]
