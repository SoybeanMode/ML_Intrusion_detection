#!/usr/bin/env python3
"""Inspect CIC-IDS2017 CSVs without changing them.

Install: python -m pip install pandas numpy
Run: python inspect_data.py --data-dir MachineLearningCSV
CSV discovery is recursive. Reports default to reports/data_inspection.json.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def inspect_file(path, chunk_size, seen):
    # Read as strings to preserve labels and expose bad numeric values explicitly.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        original = next(csv.reader(handle), [])
    columns = [name.strip() for name in original]
    if not columns:
        raise ValueError("No header found")
    used = set()
    renamed = []

    for name in columns:
        candidate = name
        suffix = 2
        while candidate in used:
            candidate = f"{name}__{suffix}"
            suffix += 1
        used.add(candidate)
        renamed.append(candidate)

    if renamed != columns:
        print("  Repeated headers renamed:",
              [(old, new) for old, new in zip(columns, renamed)
               if old != new])

    columns = renamed
    result = {
        "file": str(path), "size_mb": round(path.stat().st_size / 1_000_000, 2),
        "rows": 0, "columns": columns,
        "headers_changed_by_stripping": sum(a != b for a, b in zip(original, columns)),
        "label_counts": Counter(), "missing_by_column": Counter(),
        "infinite_by_column": Counter(), "invalid_numeric_by_column": Counter(),
        "constant_columns": [], "duplicate_rows_within_file_hash_based": 0,
        "rows_matching_earlier_files_hash_based": 0,
    }
    local_seen = set()
    schema = tuple(sorted(columns))
    prior_seen = seen.setdefault(schema, set())
    unique_values = {col: set() for col in columns}
    for chunk in pd.read_csv(path, dtype=str, chunksize=chunk_size,
                             encoding="utf-8-sig", keep_default_na=False):
        chunk.columns = columns
        chunk = chunk.apply(lambda col: col.str.strip())
        # Treat standard missing markers as missing, including blank cells.
        missing = chunk.apply(lambda col: col.str.lower().isin(
            ["", "nan", "na", "n/a", "null", "none", "<na>"]))
        result["rows"] += len(chunk)
        for col in columns:
            result["missing_by_column"][col] += int(missing[col].sum())
            if len(unique_values[col]) < 2:
                unique_values[col].update(chunk.loc[~missing[col], col].unique()[:2])
            if col == "Label":
                labels = chunk[col].mask(missing[col], "<MISSING>")
                result["label_counts"].update(labels.value_counts().to_dict())
                continue
            numbers = pd.to_numeric(chunk[col], errors="coerce")
            result["infinite_by_column"][col] += int(np.isinf(numbers).sum())
            result["invalid_numeric_by_column"][col] += int(
                (numbers.isna() & ~missing[col]).sum())
        # Order columns consistently; source filenames are deliberately excluded.
        # 64-bit hashes keep this check manageable, but are not an exact proof.
        hashes = pd.util.hash_pandas_object(chunk.loc[:, list(schema)], index=False)
        for value in hashes:
            value = int(value)
            if value in local_seen:
                result["duplicate_rows_within_file_hash_based"] += 1
            if value in prior_seen:
                result["rows_matching_earlier_files_hash_based"] += 1
            local_seen.add(value)
    prior_seen.update(local_seen)
    result["constant_columns"] = [col for col, values in unique_values.items()
                                  if len(values) == 1]
    result["all_missing_columns"] = [col for col in columns
                                     if result["missing_by_column"][col] == result["rows"]]
    for key in ("missing_by_column", "infinite_by_column", "invalid_numeric_by_column"):
        result[key] = {k: v for k, v in result[key].items() if v}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_inspection.json"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if not args.data_dir.is_dir():
        parser.error(f"Data directory does not exist: {args.data_dir}")
    files = sorted(p for p in args.data_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() == ".csv")
    if not files:
        parser.error(f"No CSV files found under {args.data_dir}")
    print(f"Found {len(files)} CSV files. Data will not be modified.", flush=True)
    print("Duplicate checks retain row hashes in memory across chunks.", flush=True)
    results, errors, seen = [], [], {}
    for path in files:
        print(f"\nInspecting {path.name} ...", flush=True)
        try:
            result = inspect_file(path, args.chunk_size, seen)
        except (ValueError, OSError, UnicodeError, pd.errors.ParserError) as exc:
            errors.append({"file": str(path), "error": str(exc)})
            print(f"  ERROR: {exc}", flush=True)
            continue
        results.append(result)
        print(f"  Rows: {result['rows']:,}; columns: {len(result['columns'])}")
        print(f"  Labels: {dict(result['label_counts'])}")
        for key in ("missing_by_column", "infinite_by_column", "invalid_numeric_by_column"):
            print(f"  {key}: {result[key] or 'none'}")
        print(f"  Constant columns: {result['constant_columns']}")
        print(f"  Duplicate rows within file (hash-based): "
              f"{result['duplicate_rows_within_file_hash_based']:,}")
        print(f"  Rows matching earlier files (hash-based): "
              f"{result['rows_matching_earlier_files_hash_based']:,}", flush=True)
    labels = Counter()
    for result in results:
        labels.update(result["label_counts"])
    schemas = {tuple(sorted(result["columns"])) for result in results}
    report = {
        "total_rows": sum(r["rows"] for r in results),
        "files_found": len(files), "files_inspected": len(results),
        "label_counts": dict(labels),
        "same_column_names_across_files": len(schemas) == 1 if results else None,
        "files_missing_Label": [r["file"] for r in results if "Label" not in r["columns"]],
        "duplicate_check_note": "Uses 64-bit hashes of stripped string values, grouped by schema. "
                                "Hash collisions are possible; differently formatted numbers may not match. "
                                "Within-file and earlier-file counts can overlap; do not add them.",
        "files": results, "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nTOTAL: {report['total_rows']:,} rows across {len(results)} inspected files")
    print(f"Matching column names: {report['same_column_names_across_files']}")
    print(f"Combined label counts: {dict(labels)}")
    print(f"Full report: {args.output.resolve()}")
    if errors:
        print("Report is incomplete: some files could not be inspected.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
