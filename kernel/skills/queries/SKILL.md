---
name: queries
description: "Running SQL straight over files on this machine with duckdb — a directory of CSVs, Parquet, JSON or a SQLite database, queried in place without loading any of it into the cell. Read this when the data is larger than the answer, or spread across more files than you want to open one at a time."
---

# queries

Instructions only — there is nothing to import here. `duckdb` is installed and you use it
directly. Editing a single spreadsheet as a document is the `spreadsheets` skill.

The reason to reach for this rather than pandas: the query runs over the files, and only
the result comes back. A directory of monthly exports totalling a gigabyte answers a
`GROUP BY` without a gigabyte ever entering the cell — which matters here, because what
enters the cell has to fit in the machine and what you print has to fit in your context.

## Files are tables

A path, or a glob, is a table name:

```python
import duckdb

duckdb.sql("SELECT region, sum(amount) FROM 'sales/*.csv' GROUP BY 1 ORDER BY 2 DESC")
duckdb.sql("SELECT count(*) FROM 'events/*.parquet'")
duckdb.sql("SELECT * FROM 'log.json' LIMIT 5")
```

The glob is read as one table, so files must share a schema; where they do not, duckdb
raises rather than guessing. Column types are inferred from a sample, which is the same
hazard `spreadsheets` describes for CSV — an identifier of digits becomes a number. Force
it when it matters:

```python
duckdb.sql("SELECT * FROM read_csv('sales/*.csv', types={'code': 'VARCHAR'})")
```

`read_csv(..., filename=true)` adds the source path as a column, which is how you find out
which file the odd rows came from.

## Getting the answer out

`duckdb.sql(...)` returns a relation, and printing it shows a preview rather than
everything — the query has not necessarily finished feeding you. Take what you mean:

```python
duckdb.sql(q).df()          # pandas DataFrame
duckdb.sql(q).fetchone()    # one row, for a scalar
duckdb.sql(q).fetchall()    # every row, so bound it with LIMIT first
```

Aggregate in SQL and print the aggregate. Pulling rows into Python to count them there
spends the context the aggregate was supposed to save.

## Dataframes are tables too

A pandas DataFrame already bound in the namespace is queryable by its variable name, with
no registration step:

```python
df = pd.read_excel("book.xlsx")
duckdb.sql("SELECT region, sum(amount) FROM df GROUP BY 1")
```

This works because the namespace persists between cells, so a frame built three cells ago
is still a table now. It also means renaming the variable renames the table.

## SQLite, and the one thing that needs the network

```python
duckdb.sql("INSTALL sqlite; LOAD sqlite; ATTACH 'app.db' AS app (TYPE sqlite)")
duckdb.sql("SELECT * FROM app.users LIMIT 5")
```

`INSTALL` downloads the extension the first time it is used. On a machine without network
access that fails, and it fails at the point of use rather than at import — so on an
isolated box, read SQLite with the standard library's `sqlite3` instead and hand duckdb the
resulting frame. Untyped SQLite columns come back as `bytes`; cast them in the query.

## Writing results

```python
duckdb.sql("COPY (SELECT ...) TO 'out.parquet'")
duckdb.sql("COPY (SELECT ...) TO 'out.csv' (HEADER, DELIMITER ',')")
```

A result that is going into a report should go through `documents` as a table, not through
a CSV the reader has to open separately.
