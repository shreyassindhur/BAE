"""
data_engine.py

Handles:
- Loading uploaded CSV/XLSX into pandas
- Auto-detecting schema (column types, sample values, likely date columns)
- Executing SAFE, STRUCTURED operations against the data

IMPORTANT SAFETY DESIGN:
The LLM never writes or executes raw Python/pandas/SQL code against user data.
Instead, it must output a small, validated JSON "operation spec" (see OPERATION
SCHEMA below). This backend interprets that spec using a fixed set of pandas
calls. This means a malicious or hallucinated LLM response can, at worst,
request an operation that doesn't exist (which we reject) - it can never
execute arbitrary code.
"""

import pandas as pd
import numpy as np
from typing import Any


# ---------- Schema detection ----------

def load_dataframe(file_path: str) -> pd.DataFrame:
    if file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_schema(df: pd.DataFrame) -> dict:
    """Produce a compact schema description to feed the LLM."""
    schema = {"columns": [], "row_count": int(len(df))}

    for col in df.columns:
        series = df[col]
        col_info: dict[str, Any] = {"name": col}

        # try to detect dates
        is_date = False
        if series.dtype == object:
            try:
                parsed = pd.to_datetime(series.dropna().head(20), errors="raise")
                is_date = True
            except Exception:
                is_date = False

        if is_date:
            col_info["type"] = "date"
        elif pd.api.types.is_numeric_dtype(series):
            col_info["type"] = "numeric"
            col_info["min"] = float(series.min()) if series.notna().any() else None
            col_info["max"] = float(series.max()) if series.notna().any() else None
        elif pd.api.types.is_bool_dtype(series):
            col_info["type"] = "boolean"
        else:
            col_info["type"] = "categorical"
            col_info["sample_values"] = [
                str(v) for v in series.dropna().unique()[:8]
            ]

        col_info["null_count"] = int(series.isna().sum())
        schema["columns"].append(col_info)

    return schema


# ---------- Operation execution ----------

ALLOWED_AGG_FUNCS = {"sum", "mean", "count", "min", "max", "median", "nunique"}
ALLOWED_OPERATIONS = {
    "groupby_agg",
    "filter_sort_top_n",
    "value_counts",
    "describe",
    "trend_over_time",
    "correlation",
    "raw_preview",
    "missing_values",
}


class OperationError(Exception):
    pass


def _validate_column(df: pd.DataFrame, col: str, context: str):
    if col not in df.columns:
        raise OperationError(f"Unknown column '{col}' requested in {context}.")


def _clean_table_for_json(table: list[dict]) -> list[dict]:
    """Replace nan/inf with None to be JSON compliant."""
    for row in table:
        for k, v in row.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                row[k] = None
    return table


def execute_operation(df: pd.DataFrame, spec: dict) -> dict:
    """
    spec looks like:
    {
      "operation": "groupby_agg",
      "group_by": "region",
      "agg_column": "sales",
      "agg_func": "sum",
      "sort_desc": true,
      "limit": 10
    }
    Returns: {"table": [...], "chart_hint": "bar"|"line"|"none", "summary": str}
    """
    op = spec.get("operation")
    if op not in ALLOWED_OPERATIONS:
        raise OperationError(f"Operation '{op}' is not permitted.")

    if op == "groupby_agg":
        group_by = spec["group_by"]
        agg_col = spec["agg_column"]
        agg_func = spec.get("agg_func", "sum")
        limit = int(spec.get("limit", 20))
        sort_desc = bool(spec.get("sort_desc", True))

        _validate_column(df, group_by, "group_by")
        _validate_column(df, agg_col, "agg_column")
        if agg_func not in ALLOWED_AGG_FUNCS:
            raise OperationError(f"Aggregation '{agg_func}' not permitted.")

        result = (
            df.groupby(group_by)[agg_col]
            .agg(agg_func)
            .reset_index()
            .rename(columns={agg_col: f"{agg_func}_{agg_col}"})
        )
        value_col = f"{agg_func}_{agg_col}"
        result = result.sort_values(value_col, ascending=not sort_desc).head(limit)
        return {
            "table": _clean_table_for_json(result.to_dict(orient="records")),
            "chart_hint": "bar",
            "summary": f"{agg_func} of {agg_col} grouped by {group_by} (top {limit}).",
        }

    if op == "filter_sort_top_n":
        sort_col = spec["sort_column"]
        limit = int(spec.get("limit", 10))
        sort_desc = bool(spec.get("sort_desc", True))
        filters = spec.get("filters", [])  # list of {column, operator, value}

        _validate_column(df, sort_col, "sort_column")
        filtered = df.copy()
        for f in filters:
            col, operator, value = f["column"], f["operator"], f["value"]
            _validate_column(df, col, "filter")
            if operator == "==":
                filtered = filtered[filtered[col] == value]
            elif operator == ">":
                filtered = filtered[filtered[col] > value]
            elif operator == "<":
                filtered = filtered[filtered[col] < value]
            elif operator == ">=":
                filtered = filtered[filtered[col] >= value]
            elif operator == "<=":
                filtered = filtered[filtered[col] <= value]
            elif operator == "contains":
                filtered = filtered[filtered[col].astype(str).str.contains(str(value), case=False, na=False)]
            else:
                raise OperationError(f"Filter operator '{operator}' not permitted.")

        filtered = filtered.sort_values(sort_col, ascending=not sort_desc).head(limit)
        return {
            "table": _clean_table_for_json(filtered.to_dict(orient="records")),
            "chart_hint": "table",
            "summary": f"Top {limit} rows sorted by {sort_col}.",
        }

    if op == "value_counts":
        col = spec["column"]
        limit = int(spec.get("limit", 20))
        _validate_column(df, col, "column")
        result = df[col].value_counts().head(limit).reset_index()
        result.columns = [col, "count"]
        return {
            "table": _clean_table_for_json(result.to_dict(orient="records")),
            "chart_hint": "bar",
            "summary": f"Value counts for {col}.",
        }

    if op == "describe":
        col = spec.get("column")
        if col:
            _validate_column(df, col, "column")
            desc = df[col].describe().to_dict()
            table = [{"stat": k, "value": v} for k, v in desc.items()]
        else:
            desc = df.describe().to_dict()
            # Normalize {column: {stat: value}} into a flat list of rows,
            # so this always matches the list-of-records shape every other
            # operation returns.
            table = []
            for column_name, stats in desc.items():
                row = {"column": column_name}
                row.update(stats)
                table.append(row)
        return {
            "table": _clean_table_for_json(table),
            "chart_hint": "none",
            "summary": "Summary statistics.",
        }

    if op == "missing_values":
        limit = int(spec.get("limit", 50))
        null_counts = df.isna().sum()
        null_pct = (df.isna().mean() * 100).round(2)
        result = pd.DataFrame({
            "column": null_counts.index,
            "missing_count": null_counts.values,
            "missing_percent": null_pct.values,
        }).sort_values("missing_count", ascending=False).head(limit)
        return {
            "table": _clean_table_for_json(result.to_dict(orient="records")),
            "chart_hint": "bar",
            "summary": "Missing (blank) values per column, sorted highest first.",
        }

    if op == "trend_over_time":
        date_col = spec["date_column"]
        value_col = spec["value_column"]
        agg_func = spec.get("agg_func", "sum")
        freq = spec.get("freq", "M")  # D, W, M, Y

        _validate_column(df, date_col, "date_column")
        _validate_column(df, value_col, "value_column")
        if agg_func not in ALLOWED_AGG_FUNCS:
            raise OperationError(f"Aggregation '{agg_func}' not permitted.")

        temp = df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col])
        grouped = (
            temp.set_index(date_col)[value_col]
            .resample(freq)
            .agg(agg_func)
            .reset_index()
        )
        grouped.columns = [date_col, f"{agg_func}_{value_col}"]
        grouped[date_col] = grouped[date_col].astype(str)
        return {
            "table": _clean_table_for_json(grouped.to_dict(orient="records")),
            "chart_hint": "line",
            "summary": f"{agg_func} of {value_col} over time ({freq}).",
        }

    if op == "correlation":
        cols = spec.get("columns")
        numeric_df = df.select_dtypes(include=[np.number])
        if cols:
            non_numeric = [c for c in cols if c not in numeric_df.columns]
            if non_numeric:
                raise OperationError(
                    f"Correlation only works on numeric columns. "
                    f"{non_numeric} are not numeric, so they can't be correlated directly."
                )
            numeric_df = numeric_df[cols]
        if numeric_df.shape[1] < 2:
            raise OperationError(
                "Not enough numeric columns available to compute a correlation."
            )
        corr = numeric_df.corr().round(3)
        corr_reset = corr.reset_index().rename(columns={"index": "column"})
        return {
            "table": _clean_table_for_json(corr_reset.to_dict(orient="records")),
            "chart_hint": "none",
            "summary": "Correlation matrix of numeric columns.",
        }

    if op == "raw_preview":
        limit = int(spec.get("limit", 10))
        return {
            "table": _clean_table_for_json(df.head(limit).to_dict(orient="records")),
            "chart_hint": "table",
            "summary": f"First {limit} rows of the dataset.",
        }

    raise OperationError("Unhandled operation.")
