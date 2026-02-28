"""Initialize the DuckDB database and create all tables. Safe to re-run."""

from src.db import connect

con = connect()
print("Database initialized successfully")
