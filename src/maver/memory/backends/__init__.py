"""Storage backends behind the memory ports.

Embedded (SQLite, optionally ChromaDB) for local use, server (PostgreSQL, optionally
pgvector) for durable multi-run state. Both satisfy the same contract test.
"""
