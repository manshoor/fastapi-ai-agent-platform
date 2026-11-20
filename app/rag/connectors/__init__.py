"""Operator-configured pull connectors: PostgreSQL queries and JSON APIs.

connectors.yaml (read-only mount) -> connector-sync: one runner per connector
  fetch (no transaction open) -> per record: create / new version / unchanged
  complete run only: records missing from the source are deleted
"""
