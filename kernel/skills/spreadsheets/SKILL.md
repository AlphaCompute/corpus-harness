---
name: spreadsheets
description: "Reading, editing and writing .xlsx and CSV with openpyxl and pandas. Read this before you write a formula into a sheet or read one back, because a formula this interpreter writes has no computed value stored beside it: the cell reads back as None here, as an empty cell to pandas, and only becomes a number when a person opens the file."
---

# spreadsheets

Instructions only — there is nothing to import here. `openpyxl` and `pandas` are installed
and you use them directly. Querying a pile of data files rather than editing one is the
`queries` skill.

Two libraries, two jobs. `pandas` moves bulk data in and out — `read_excel`, `to_excel`,
`read_csv` — and knows nothing about formatting. `openpyxl` addresses individual cells and
owns formulas, fonts, fills, widths and merges. Reach for pandas when the sheet is a table,
openpyxl when it is a document.

## A formula written here has no value in it

This is the trap that governs everything else. openpyxl writes a formula as the string
`"=SUM(A1:A2)"` and stores no computed result beside it. Excel stores both, and everything
that reads a spreadsheet without evaluating it reads the stored result.

So, on a file this interpreter just wrote:

```python
load_workbook(path, data_only=True)["Sheet"]["A3"].value   # -> None
load_workbook(path)["Sheet"]["A3"].value                   # -> '=SUM(A1:A2)'
pd.read_excel(path, header=None).iloc[2, 0]                # -> nan
```

Nothing raises at any point. The file is valid, opens correctly in Excel, and shows the
right number the moment a person opens it — because Excel computes it then. Until then the
cell is empty to every reader.

Normally you would recalculate with LibreOffice and bake the values in. There is no
LibreOffice here, so the decision falls to you, and it turns on **who opens the file next**:

- **A person, in Excel.** Write formulas. The sheet stays live, recalculates against its
  own inputs, and shows correct numbers on open. Say in your answer that the totals appear
  when the file is opened.
- **Anything programmatic** — your own next cell, pandas, another tool, a check you are
  about to run. Write computed values. A formula here is a hole that reads as blank.
- **Both.** Compute in Python, write the value, and put the formula's intent in a note or
  an adjacent cell. Do not write a formula and then read your own file back to verify it:
  you will read `None` and conclude, wrongly, that the write failed.

## Reading a workbook somebody else made

Their formulas *do* have cached values, so both halves exist — but one load cannot give you
both:

```python
values   = load_workbook(path, data_only=True)   # numbers, formulas gone
formulas = load_workbook(path)                   # formula strings, no numbers
```

**Never save a workbook you loaded with `data_only=True`.** That object has no formulas in
it, so saving replaces every one with the literal it happened to be showing — permanently,
across the whole file, and the sheet stops being a model.

## Cells that are not where you think

- **openpyxl is 1-indexed.** `ws.cell(row=1, column=1)` is A1, and row or column `0` raises
  `ValueError` rather than wrapping.
- **A merged range holds its value in the top-left cell only.** Every other cell in the
  range is a `MergedCell` reading `None`, and assigning to one raises. Reading a merged
  header column-by-column silently yields one label and a run of blanks.
- **`read_only=True`** streams rather than building the whole workbook, which is what makes
  a large file tractable; the cells it hands back cannot be written to.

## CSV, and the digits that vanish

`pd.read_csv` infers a type per column, and an identifier made of digits is a number to it:
a `code` column of `00123` comes back as the integer `123`. Leading zeros are gone, and
nothing warns. Postcodes, account numbers, part numbers and phone numbers all lose the same
way, and a join against them then silently matches nothing.

```python
pd.read_csv(path, dtype=str)                    # everything as written
pd.read_csv(path, dtype={"code": str})          # or name the columns that are identifiers
```

Read as strings first and convert the columns you actually want as numbers. The reverse —
inferring and then repairing — cannot recover what was dropped.
