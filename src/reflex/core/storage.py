"""Local durable stores for events, observations, and versioned state.

``EventStore`` is the DuckDB ledger. ``StateStore`` is the SQLite store for
current values, transition history, candidate lifecycle, and run budgets. They
are separate databases, so a write spanning both stores is not one atomic
transaction.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pyarrow.parquet as parquet

from .models.events import ActionEvent, ActionResultEvent, Event, NativeEvent, ObservationEvent
from .models.lenses import LensCandidate, Observation


def _json(value: Any) -> str:
    """Serialize model and datetime values into deterministic JSON text."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, default=lambda item: item.isoformat() if isinstance(item, datetime) else str(item), sort_keys=True)


class EventStore:
    """DuckDB ledger for native facts and REFLEX-derived event records.

    Opening the store creates its tables and adds metadata columns missing
    from earlier local schemas. External DuckDB file access is disabled for
    this connection. Call :meth:`close` when the owner is done using it.
    """

    def __init__(self, path: str | Path):
        """Open or create the ledger at ``path`` and prepare its local schema."""
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(self.path, config={"enable_external_access": "false"})
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id VARCHAR PRIMARY KEY,
                event_type VARCHAR NOT NULL,
                event_time TIMESTAMPTZ NOT NULL,
                entity_ids JSON NOT NULL,
                attributes JSON NOT NULL,
                source VARCHAR NOT NULL,
                provenance JSON NOT NULL,
                relationships JSON NOT NULL
            )
        """)
        self.connection.execute("CREATE INDEX IF NOT EXISTS events_time_idx ON events(event_time)")
        # Add missing columns when opening an earlier REFLEX database.
        event_columns = {row[1] for row in self.connection.execute("PRAGMA table_info('events')").fetchall()}
        metadata_columns = {
            "event_kind": "VARCHAR DEFAULT 'native'",
            "payload": "JSON DEFAULT '{}'",
            "parent_event_ids": "JSON DEFAULT '[]'",
            "correlation_id": "VARCHAR",
            "causation_id": "VARCHAR",
            "schema_ref": "VARCHAR",
            "idempotency_key": "VARCHAR",
            "access_scope": "JSON DEFAULT '[]'",
            "lens_id": "VARCHAR",
            "lens_version": "VARCHAR",
            "subject": "VARCHAR",
            "action_function": "VARCHAR",
            "arguments": "JSON DEFAULT '{}'",
            "required_permission": "VARCHAR",
            "retry_limit": "INTEGER DEFAULT 0",
            "action_event_id": "VARCHAR",
            "status": "VARCHAR",
            "attempt": "INTEGER",
        }
        for column, sql_type in metadata_columns.items():
            if column not in event_columns:
                self.connection.execute(f"ALTER TABLE events ADD COLUMN {column} {sql_type}")
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                observation_id VARCHAR PRIMARY KEY,
                lens_id VARCHAR NOT NULL,
                lens_version VARCHAR NOT NULL,
                subject VARCHAR NOT NULL,
                window_start TIMESTAMPTZ NOT NULL,
                window_end TIMESTAMPTZ NOT NULL,
                observation_type VARCHAR NOT NULL,
                record JSON NOT NULL
            )
        """)

    def add_events(self, events: Iterable[Event]) -> int:
        """Insert a batch and return the number of new IDs.

        A repeated ID with identical normalized contents is treated as a
        replay and is not inserted again. Reusing an ID with different content
        fails the batch. New rows are committed together in one transaction.
        """
        inserted = 0
        self.connection.execute("BEGIN TRANSACTION")
        try:
            for event in events:
                row = self.connection.execute("SELECT event_type, event_time, entity_ids, attributes, source, provenance, relationships, event_kind, payload, parent_event_ids, correlation_id, causation_id, schema_ref, idempotency_key, access_scope, lens_id, lens_version, subject, action_function, arguments, required_permission, retry_limit, action_event_id, status, attempt FROM events WHERE event_id = ?", [event.event_id]).fetchone()
                normalized = (
                    event.event_type,
                    event.event_time,
                    _json(event.entity_ids),
                    _json(event.attributes),
                    event.source,
                    _json(event.provenance),
                    _json(event.relationships),
                    event.event_kind,
                    _json(event.payload),
                    _json(event.parent_event_ids),
                    event.correlation_id,
                    event.causation_id,
                    event.schema_ref,
                    event.idempotency_key,
                    _json(event.access_scope),
                    getattr(event, "lens_id", None),
                    getattr(event, "lens_version", None),
                    getattr(event, "subject", None),
                    getattr(event, "action_function", None),
                    _json(getattr(event, "arguments", {})),
                    getattr(event, "required_permission", None),
                    getattr(event, "retry_limit", 0),
                    getattr(event, "action_event_id", None),
                    getattr(event, "status", None),
                    getattr(event, "attempt", None),
                )
                if row:
                    prior = tuple(row)
                    current = normalized
                    # DuckDB may return JSON with different whitespace or key
                    # ordering, so compare normalized content rather than text.
                    for index in (2, 3, 5, 6, 8, 9, 14, 19):
                        prior = (*prior[:index], _json(json.loads(str(prior[index]))), *prior[index + 1:])
                    if prior != current:
                        raise ValueError(f"event_id {event.event_id!r} already exists with different content")
                    continue
                columns = "event_id,event_type,event_time,entity_ids,attributes,source,provenance,relationships,event_kind,payload,parent_event_ids,correlation_id,causation_id,schema_ref,idempotency_key,access_scope,lens_id,lens_version,subject,action_function,arguments,required_permission,retry_limit,action_event_id,status,attempt"
                placeholders = ",".join("?" for _ in range(26))
                # Preserve the original event columns, then append optional REFLEX metadata.
                values = [event.event_id, *normalized[:7], *normalized[7:]]
                self.connection.execute(f"INSERT INTO events ({columns}) VALUES ({placeholders})", values)
                inserted += 1
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return inserted

    def events(self, start: datetime | None = None, end: datetime | None = None, limit: int | None = None,
               *, include_derived: bool = True) -> list[Event]:
        """Return events ordered by time then ID in ``[start, end)``.

        By default the result contains native input plus derived REFLEX events.
        Set ``include_derived=False`` when constructing model inputs. A missing
        bound is unbounded; ``limit`` is passed to DuckDB as a row limit.
        """
        clauses, params = [], []
        if not include_derived:
            clauses.append("event_kind = 'native'")
        if start is not None:
            clauses.append("event_time >= ?")
            params.append(start)
        if end is not None:
            clauses.append("event_time < ?")
            params.append(end)
        query = "SELECT event_id, event_type, event_time, entity_ids, attributes, source, provenance, relationships, event_kind, payload, parent_event_ids, correlation_id, causation_id, schema_ref, idempotency_key, access_scope, lens_id, lens_version, subject, action_function, arguments, required_permission, retry_limit, action_event_id, status, attempt FROM events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY event_time, event_id"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        records = []
        for row in self.connection.execute(query, params).fetchall():
            shared = dict(
                event_id=row[0], event_type=row[1], event_time=row[2],
                entity_ids=tuple(json.loads(str(row[3]))), attributes=json.loads(str(row[4])),
                source=row[5], provenance=json.loads(str(row[6])), relationships=tuple(json.loads(str(row[7]))),
                event_kind=row[8] or "native", payload=json.loads(str(row[9] or "{}")),
                parent_event_ids=tuple(json.loads(str(row[10] or "[]"))), correlation_id=row[11],
                causation_id=row[12], schema_ref=row[13], idempotency_key=row[14],
                access_scope=tuple(json.loads(str(row[15] or "[]"))),
            )
            category = row[8] or "native"
            if category == "observation" and row[16]:
                event = ObservationEvent(**shared, lens_id=row[16], lens_version=row[17], subject=row[18])
            elif category == "action" and row[19]:
                event = ActionEvent(**shared, action_function=row[19], arguments=json.loads(str(row[20] or "{}")),
                                    required_permission=row[21], retry_limit=row[22] or 0)
            elif category == "action" and row[23]:
                event = ActionResultEvent(**shared, action_event_id=row[23], status=row[24], attempt=row[25])
            else:
                event = NativeEvent(**shared)
            records.append(event)
        return records

    def native_events(self, start: datetime | None = None, end: datetime | None = None,
                      limit: int | None = None) -> list[Event]:
        """Return only source events, commonly used as model input."""
        return self.events(start, end, limit, include_derived=False)

    def all_events(self, start: datetime | None = None, end: datetime | None = None,
                   limit: int | None = None) -> list[Event]:
        """Return source and derived records from the complete ledger history."""
        return self.events(start, end, limit, include_derived=True)

    def count_events(self) -> int:
        """Count all native and derived ledger rows."""
        return self.connection.execute("SELECT count(*) FROM events").fetchone()[0]

    def count_native_events(self) -> int:
        """Count source rows only, excluding observations and action records."""
        return self.connection.execute("SELECT count(*) FROM events WHERE event_kind = 'native'").fetchone()[0]

    def export_parquet(self, path: str | Path) -> int:
        """Write an event-time-sorted Parquet snapshot and return its row count.

        The export is a point-in-time copy for analysis. Changes later written
        to the DuckDB ledger are not reflected in the existing Parquet file.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        table = self.connection.execute("SELECT * FROM events ORDER BY event_time, event_id").fetch_arrow_table()
        parquet.write_table(table, target)
        return table.num_rows

    def batches(self, batch_days: int = 5, *, include_derived: bool = False) -> list[tuple[datetime, datetime, int]]:
        """Describe consecutive UTC-sized event-time batches and their counts."""
        if batch_days <= 0:
            raise ValueError("batch_days must be positive")
        kind_filter = "" if include_derived else " WHERE event_kind = 'native'"
        minimum, maximum = self.connection.execute(f"SELECT min(event_time), max(event_time) FROM events{kind_filter}").fetchone()
        if minimum is None:
            return []
        from datetime import timedelta
        result = []
        start = minimum.replace(hour=0, minute=0, second=0, microsecond=0)
        while start <= maximum:
            end = start + timedelta(days=batch_days)
            kind_filter = "" if include_derived else " AND event_kind = 'native'"
            count = self.connection.execute(f"SELECT count(*) FROM events WHERE event_time >= ? AND event_time < ?{kind_filter}", [start, end]).fetchone()[0]
            result.append((start, end, count))
            start = end
        return result

    def query_sql_lens(self, sql: str, *, limit: int = 1000, start: datetime | None = None,
                       end: datetime | None = None) -> list[dict[str, Any]]:
        """Run a single bounded read query for a deterministic SQL Lens.

        The query is wrapped with optional event-time predicates and a
        ``limit + 1`` sentinel. Returning the extra row raises an error instead
        of silently truncating candidate output. Common external-file and
        connection-changing functions are rejected before execution. This is
        an application guardrail for local trusted code, not an adversarial SQL
        isolation boundary.
        """
        if limit < 1 or limit > 10_000:
            raise ValueError("lens row limit must be between 1 and 10000")
        if (start is not None and start.tzinfo is None) or (end is not None and end.tzinfo is None):
            raise ValueError("query bounds must include a timezone")
        if start is not None and end is not None and end <= start:
            raise ValueError("query end must follow query start")
        statement = sql.strip().rstrip(";").strip()
        if not statement.lower().startswith(("select ", "with ")) or ";" in statement:
            raise ValueError("lens SQL must be one SELECT statement")
        lowered = statement.lower()
        forbidden = ("read_csv", "read_parquet", "read_json", "glob(", "attach ", "copy ", "install ", "load ", "pragma ", "httpfs", "sqlite_scan")
        if any(term in lowered for term in forbidden):
            raise ValueError("lens SQL may not read external files or relations outside registered events")
        # Apply bounds and a sentinel limit outside the submitted SELECT so a
        # lens cannot bypass the adapter's output-size contract.
        wrapped = f"SELECT * FROM ({statement}) AS lens_output"
        params: list[Any] = []
        clauses = []
        if start is not None:
            clauses.append("event_time >= ?")
            params.append(start)
        if end is not None:
            clauses.append("event_time < ?")
            params.append(end)
        if clauses:
            wrapped += " WHERE " + " AND ".join(clauses)
        wrapped += " LIMIT ?"
        params.append(limit + 1)
        cursor = self.connection.execute(wrapped, params)
        if len(cursor.description or ()) == 0:
            return []
        rows = cursor.fetchall()
        if len(rows) > limit:
            raise ValueError(f"lens output exceeds the {limit} row limit")
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def save_observations(self, observations: Iterable[Observation]) -> int:
        """Persist findings as both typed rows and ledger observation events."""
        items = list(observations)
        self.add_events(item.as_event() for item in items)
        count = 0
        for observation in items:
            self.connection.execute("""
                INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(observation_id) DO UPDATE SET record = excluded.record
            """, [observation.observation_id, observation.lens_id, observation.lens_version, observation.subject,
                  observation.window_start, observation.window_end, observation.observation_type, _json(observation)])
            count += 1
        return count

    def load_observations(self, lens_id: str | None = None) -> list[Observation]:
        """Load findings in deterministic window, subject, and ID order."""
        query = "SELECT record FROM observations"
        params: list[Any] = []
        if lens_id:
            query += " WHERE lens_id = ?"
            params.append(lens_id)
        query += " ORDER BY window_start, subject, observation_id"
        return [Observation.model_validate(json.loads(str(row[0]))) for row in self.connection.execute(query, params).fetchall()]

    def close(self) -> None:
        """Close the DuckDB connection owned by this store."""
        self.connection.close()


class StateStore:
    """SQLite store for current state, audit records, and bounded run budgets.

    Every value lives in a named namespace and key. Updates replace the current
    value while appending a versioned transition with a reason. Candidate
    lifecycle, hypotheses, investigations, and usage budgets have dedicated
    tables in the same local database.
    """

    NAMESPACES = {"operational", "derived", "knowledge", "lens", "discovery", "execution"}
    TRANSITIONS = {
        "proposed": {"testing", "rejected"}, "testing": {"validated", "rejected"},
        "validated": {"approved", "rejected"}, "approved": {"active", "retired"},
        "active": {"retired"}, "retired": set(), "rejected": {"proposed"},
    }

    def __init__(self, path: str | Path):
        """Open or create the state database and enable SQLite write-ahead logging."""
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS state_records (
                namespace TEXT NOT NULL, state_key TEXT NOT NULL, version INTEGER NOT NULL,
                value_json TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(namespace, state_key)
            );
            CREATE TABLE IF NOT EXISTS state_transitions (
                transition_id INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT NOT NULL,
                state_key TEXT NOT NULL, from_version INTEGER, to_version INTEGER NOT NULL,
                value_json TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lens_candidates (
                candidate_id TEXT PRIMARY KEY, status TEXT NOT NULL, record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY, status TEXT NOT NULL, record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS investigations (
                investigation_id TEXT PRIMARY KEY, record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS policy_budgets (
                run_id TEXT PRIMARY KEY, max_investigations INTEGER NOT NULL,
                max_queries INTEGER NOT NULL, investigations_used INTEGER NOT NULL DEFAULT 0,
                queries_used INTEGER NOT NULL DEFAULT 0
            );
        """)

    def set(self, namespace: str, key: str, value: Any, reason: str) -> int:
        """Store a new version and append an audit entry; return its version."""
        if namespace not in self.NAMESPACES:
            raise ValueError(f"unknown state namespace: {namespace}")
        now = datetime.now(timezone.utc).isoformat()
        record = _json(value)
        with self.connection:
            prior = self.connection.execute("SELECT version FROM state_records WHERE namespace=? AND state_key=?", (namespace, key)).fetchone()
            version = (prior["version"] if prior else 0) + 1
            self.connection.execute("INSERT INTO state_records VALUES (?, ?, ?, ?, ?) ON CONFLICT(namespace,state_key) DO UPDATE SET version=excluded.version,value_json=excluded.value_json,updated_at=excluded.updated_at", (namespace, key, version, record, now))
            self.connection.execute("INSERT INTO state_transitions(namespace,state_key,from_version,to_version,value_json,reason,created_at) VALUES (?,?,?,?,?,?,?)", (namespace, key, prior["version"] if prior else None, version, record, reason, now))
        return version

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Return the current decoded value, or ``default`` when absent."""
        row = self.connection.execute("SELECT value_json FROM state_records WHERE namespace=? AND state_key=?", (namespace, key)).fetchone()
        return json.loads(row[0]) if row else default

    def snapshot(self, namespace: str | None = None) -> dict[str, dict[str, Any]]:
        """Return current values, optionally restricted to one namespace.

        Pass a namespace when keys may be reused in multiple namespaces.
        """
        query = "SELECT namespace,state_key,value_json FROM state_records"
        params: tuple[Any, ...] = ()
        if namespace:
            query += " WHERE namespace=?"
            params = (namespace,)
        return {row["state_key"]: json.loads(row["value_json"]) for row in self.connection.execute(query, params)}

    def snapshot_at(self, namespace: str, as_of: datetime) -> dict[str, dict[str, Any]]:
        """Return a namespace view reconstructed at an aware timestamp.

        For values with an ``as_of`` field, the latest transition whose value
        timestamp is no later than ``as_of`` is returned. Values without that
        field are treated as static state and returned as currently stored.
        """
        if as_of.tzinfo is None:
            raise ValueError("state snapshot time must include a timezone")
        result: dict[str, dict[str, Any]] = {}
        rows = self.connection.execute("SELECT state_key,value_json FROM state_records WHERE namespace=?", (namespace,)).fetchall()
        for row in rows:
            value = json.loads(row["value_json"])
            if isinstance(value, dict) and value.get("as_of"):
                history = self.connection.execute("SELECT transition_id,value_json FROM state_transitions WHERE namespace=? AND state_key=?", (namespace, row["state_key"])).fetchall()
                eligible = []
                for transition in history:
                    candidate = json.loads(transition["value_json"])
                    candidate_time = candidate.get("as_of") if isinstance(candidate, dict) else None
                    if candidate_time and datetime.fromisoformat(candidate_time) <= as_of:
                        eligible.append((datetime.fromisoformat(candidate_time), transition["transition_id"], candidate))
                if not eligible:
                    continue
                value = max(eligible, key=lambda item: (item[0], item[1]))[2]
            result[row["state_key"]] = value
        return result

    def audit(self, namespace: str | None = None) -> list[dict[str, Any]]:
        """List state transitions in insertion order, optionally by namespace."""
        query = "SELECT * FROM state_transitions"
        params: tuple[Any, ...] = ()
        if namespace:
            query += " WHERE namespace=?"
            params = (namespace,)
        query += " ORDER BY transition_id"
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

    def put_hypothesis(self, hypothesis: Any) -> None:
        """Persist a hypothesis record and its versioned knowledge-state value."""
        with self.connection:
            self.connection.execute("INSERT INTO hypotheses VALUES (?,?,?) ON CONFLICT(hypothesis_id) DO UPDATE SET status=excluded.status,record_json=excluded.record_json", (hypothesis.hypothesis_id, hypothesis.status, _json(hypothesis)))
        self.set("knowledge", f"hypothesis:{hypothesis.hypothesis_id}", hypothesis, "hypothesis update")

    def put_candidate(self, candidate: LensCandidate, reason: str = "candidate recorded") -> None:
        """Persist a candidate and append its status to the state audit."""
        with self.connection:
            self.connection.execute("INSERT INTO lens_candidates VALUES (?,?,?) ON CONFLICT(candidate_id) DO UPDATE SET status=excluded.status,record_json=excluded.record_json", (candidate.candidate_id, candidate.status.value, _json(candidate)))
        self.set("lens", f"candidate:{candidate.candidate_id}", candidate, reason)

    def transition_candidate(self, candidate_id: str, status: str, *, validation: dict[str, Any] | None = None) -> LensCandidate:
        """Apply one allowed lifecycle transition and return the updated record."""
        row = self.connection.execute("SELECT record_json FROM lens_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown lens candidate {candidate_id}")
        candidate = LensCandidate.model_validate(json.loads(row[0]))
        allowed = self.TRANSITIONS[candidate.status.value]
        if status not in allowed:
            raise ValueError(f"invalid candidate transition: {candidate.status.value} -> {status}")
        changes: dict[str, Any] = {"status": status}
        if validation is not None:
            changes["validation"] = validation
        if "status" in changes:
            from .models.lenses import CandidateStatus
            changes["status"] = CandidateStatus(changes["status"])
        updated = candidate.model_copy(update=changes)
        self.put_candidate(updated, f"candidate {candidate.status.value} -> {status}")
        return updated

    def initialize_budget(self, run_id: str, max_investigations: int, max_queries: int) -> None:
        """Create a per-run allowance once; later calls keep the original limits."""
        if max_investigations < 0 or max_queries < 0:
            raise ValueError("budgets cannot be negative")
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO policy_budgets(run_id,max_investigations,max_queries) VALUES (?,?,?)", (run_id, max_investigations, max_queries))

    def consume_budget(self, run_id: str, kind: str, amount: int = 1) -> bool:
        """Atomically consume allowance, returning false when the cap is reached."""
        if kind not in {"investigation", "query"} or amount <= 0:
            raise ValueError("invalid budget consumption")
        used_col, max_col = ("investigations_used", "max_investigations") if kind == "investigation" else ("queries_used", "max_queries")
        with self.connection:
            row = self.connection.execute(f"SELECT {used_col},{max_col} FROM policy_budgets WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"budget {run_id!r} is not initialized")
            if row[used_col] + amount > row[max_col]:
                return False
            self.connection.execute(f"UPDATE policy_budgets SET {used_col}={used_col}+? WHERE run_id=?", (amount, run_id))
            return True

    def budget(self, run_id: str) -> dict[str, int]:
        """Return the configured limits and consumed investigation/query counts."""
        row = self.connection.execute("SELECT * FROM policy_budgets WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError(run_id)
        return dict(row)

    def close(self) -> None:
        """Close the SQLite connection owned by this store."""
        self.connection.close()

