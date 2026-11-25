"""Example deterministic lenses for transaction-like event attributes."""

TRANSACTION_HIGH_AMOUNT_SQL = """
SELECT event_id, event_type, event_time, entity_ids, attributes
FROM events
WHERE TRY_CAST(json_extract_string(attributes, '$.amount') AS DOUBLE) >= 4000
  AND event_kind = 'native'
""".strip()

__all__ = ["TRANSACTION_HIGH_AMOUNT_SQL"]
