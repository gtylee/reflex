"""Local-file adapters that normalize selected BPI and Loghub formats to Events.

Adapters preserve source fields as attributes and attach file/row provenance.
They do not fetch source data or make domain judgments about event meaning.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models.events import Event


def _parse_time(value: str) -> datetime:
    """Parse ISO-8601 text and normalize it to an aware UTC timestamp."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _pop_first(row: dict, names: tuple[str, ...], default: str = ""):
    """Remove and return the first present alias from a mutable source row."""
    for name in names:
        if name in row:
            return row.pop(name)
    return default


def read_bpi2017(path: str | Path, *, source: str = "bpi-challenge-2017") -> Iterator[Event]:
    """Stream BPI CSV or XES/XES.GZ events as immutable REFLEX records.

    The case identifier becomes the entity ID, the activity name becomes the
    event type, and other event fields remain in ``attributes``. XES trace and
    event positions, or CSV row positions, are retained in provenance. IDs are
    based on those positions, so stable source-native IDs are preferable when
    repeated exports can reorder rows.
    """
    input_path = Path(path)
    if input_path.name.lower().endswith((".xes", ".xes.gz")):
        opener = gzip.open if input_path.name.lower().endswith(".gz") else open
        case_index = event_index = 0
        with opener(input_path, "rb") as file:
            for _action, element in ET.iterparse(file, events=("end",)):
                if not element.tag.endswith("trace"):
                    continue
                trace_attributes = {child.attrib.get("key", ""): _xes_value(child)
                                    for child in list(element) if "key" in child.attrib and not child.tag.endswith("event")}
                entity = str(trace_attributes.get("case:concept:name", trace_attributes.get("concept:name", f"trace-{case_index}")))
                for trace_event in (child for child in list(element) if child.tag.endswith("event")):
                    values = {child.attrib.get("key", ""): _xes_value(child) for child in list(trace_event) if "key" in child.attrib}
                    raw_time = values.pop("time:timestamp", None)
                    if raw_time is None:
                        raise ValueError(f"XES event in trace {entity!r} has no time:timestamp")
                    kind = str(values.pop("concept:name", "unknown"))
                    timestamp = _parse_time(str(raw_time))
                    yield Event(event_id=f"bpi2017:{case_index}:{event_index}", event_type=kind,
                                event_time=timestamp, entity_ids=(entity,), attributes=values,
                                source=source, provenance={"file": input_path.name, "trace": case_index, "event": event_index})
                    event_index += 1
                case_index += 1
                element.clear()
        return
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        case_field = next((name for name in ("case:concept:name", "case_id", "case id", "Case ID") if name in fields), None)
        type_field = next((name for name in ("concept:name", "event_type", "Activity", "activity") if name in fields), None)
        time_field = next((name for name in ("time:timestamp", "timestamp", "Timestamp") if name in fields), None)
        if not all((case_field, type_field, time_field)):
            raise ValueError("BPI CSV requires case, activity, and timestamp columns")
        for index, row in enumerate(reader):
            entity = str(row.pop(case_field, ""))
            kind = str(row.pop(type_field, "unknown"))
            timestamp = _parse_time(row.pop(time_field, ""))
            yield Event(event_id=f"bpi2017:{index}", event_type=kind, event_time=timestamp,
                        entity_ids=(entity,), attributes=row, source=source,
                        provenance={"file": Path(path).name, "row": index + 2})


def _xes_value(element: ET.Element):
    """Decode common XES primitive values while preserving unknowns as text."""
    value = element.attrib.get("value", "")
    tag = element.tag.rsplit("}", 1)[-1]
    try:
        if tag == "int":
            return int(value)
        if tag == "float":
            return float(value)
        if tag == "boolean":
            return value.lower() == "true"
    except ValueError:
        return value
    return value


_HDFS_LINE = re.compile(r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>\w+)\s+(?P<component>[^:]+):\s*(?P<message>.*)$")


def read_loghub(path: str | Path, *, source: str = "loghub") -> Iterator[Event]:
    """Read local JSONL, CSV, or common HDFS text as REFLEX events.

    JSONL and CSV fields outside timestamp/entity/type become attributes. The
    HDFS text reader recognizes the standard date/time/PID/level/component
    prefix and skips lines that do not match it. No network request is made.
    """
    input_path = Path(path)
    if input_path.suffix.lower() in {".jsonl", ".ndjson"}:
        with input_path.open(encoding="utf-8") as file:
            for index, line in enumerate(file):
                if not line.strip():
                    continue
                row = json.loads(line)
                timestamp = _parse_time(str(_pop_first(row, ("timestamp", "time"))))
                entity = str(_pop_first(row, ("entity_id", "entity", "pid"), "log"))
                kind = str(_pop_first(row, ("event_type", "level"), "log"))
                yield Event(event_id=f"loghub:{input_path.name}:{index}", event_type=kind,
                            event_time=timestamp, entity_ids=(entity,), attributes=row,
                            source=source, provenance={"file": input_path.name, "line": index + 1})
        return
    if input_path.suffix.lower() == ".csv":
        with input_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames or []
            time_field = next((name for name in ("timestamp", "time", "date") if name in fields), None)
            entity_field = next((name for name in ("entity_id", "entity", "pid", "block_id") if name in fields), None)
            type_field = next((name for name in ("event_type", "level", "component") if name in fields), None)
            if not time_field:
                raise ValueError("Loghub CSV requires a timestamp, time, or date column")
            for index, row in enumerate(reader):
                timestamp = _parse_time(str(row.pop(time_field, "")))
                entity = str(row.pop(entity_field, "log") if entity_field else "log")
                kind = str(row.pop(type_field, "log") if type_field else "log")
                yield Event(event_id=f"loghub:{input_path.name}:{index}", event_type=kind,
                            event_time=timestamp, entity_ids=(entity,), attributes=row,
                            source=source, provenance={"file": input_path.name, "row": index + 2})
        return
    with input_path.open(encoding="utf-8", errors="replace") as file:
        for index, line in enumerate(file):
            match = _HDFS_LINE.match(line.strip())
            if not match:
                continue
            values = match.groupdict()
            timestamp = datetime.strptime(values["date"] + values["time"], "%y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            yield Event(event_id=f"loghub:{input_path.name}:{index}", event_type=values["level"],
                        event_time=timestamp, entity_ids=(values["pid"],),
                        attributes={"level": values["level"], "component": values["component"], "message": values["message"]},
                        source=source, provenance={"file": input_path.name, "line": index + 1})
