"""
services.ai_tasks
High-level AI helpers for DataBotti.

Responsibility:
- Build prompts for the LLM based on dataset context
- Keep app-specific prompt phrasing in one place
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


# ---- Minimal, decoupled types so we don't import app internals here ---------

@dataclass
class ColumnBrief:
    name: str
    dtype: str
    nullable: Optional[bool] = None


@dataclass
class DatasetBrief:
    filename: str
    rows: Optional[int] = None
    cols: Optional[int] = None
    delimiter: Optional[str] = None
    encoding: Optional[str] = None
    columns: Optional[Iterable[ColumnBrief]] = None


# ---- Public API -------------------------------------------------------------

def build_dataset_summary_prompt(summary: Optional[Mapping[str, Any]]) -> str:
    """
    Build a short, system-like instruction to summarize a dataset.
    Accepts a loose dict (e.g., what you pass to templates) to avoid tight coupling.
    """
    brief = _to_brief(summary)
    meta_lines = [
        f"Filename: {brief.filename}",
        f"Shape: {brief.rows} rows x {brief.cols} cols" if brief.rows is not None and brief.cols is not None else "Shape: unknown",
        f"Delimiter: {brief.delimiter or 'unknown'}",
        f"Encoding: {brief.encoding or 'unknown'}",
    ]

    cols_line = ""
    if brief.columns:
        names = [c.name for c in list(brief.columns)[:12]]
        more = " (…)" if brief.columns and len(list(brief.columns)) > 12 else ""
        cols_line = "Columns: " + ", ".join(names) + more
        meta_lines.append(cols_line)

    meta = "\n".join(meta_lines)

    return (
        "Fasse den Datensatz in einem prägnanten Absatz zusammen: Größe (Zeilen x Spalten), auffällige Spalten, "
        "offensichtliche Datenqualitätsprobleme (nur wenn erkennbar) sowie 1–2 sinnvolle nächste Analyseschritte."
    )


# ---- Stage-1 column selection prompt builder --------------------------------

def build_relevant_columns_prompt(user_task: str, rows: int, cols: int, column_summaries: list[str]) -> str:
    """
    Build the Stage-1 prompt for column relevance selection.
    """
    header = (
        f"User task: {user_task}\n\n"
        f"Dataset has {rows} rows and {cols} columns.\n\n"
        "Below is one line per column with type info, cardinality and flags:\n"
    )
    cols_block = "\n".join(column_summaries)

    instructions = (
        "\n\nYour job:\n"
        "- Select which columns are relevant to the user task.\n"
        "- Answer ONLY with a comma-separated list of column names, or 'ALL' if all are relevant.\n"
        "- Never invent names; use only the listed ones.\n"
        "- Do not drop sparse columns unless clearly irrelevant to the task.\n"
        "- Constant columns are usually not relevant, unless explicitly asked.\n"
        "- If the user task explicitly mentions a column, always include it.\n"
    )

    return header + cols_block + instructions


def build_chat_prompt(user_prompt: str, summary: Optional[Mapping[str, Any]]) -> str:
    """
    Build a user-facing prompt that includes dataset context plus the user's request.
    """
    base = build_dataset_summary_prompt(summary)
    up = (user_prompt or "").strip()
    if not up:
        up = "Provide useful insights about this dataset."
    return f"{base}\n\nUser request:\n{up}"


# ---- Helpers ----------------------------------------------------------------

def _to_brief(summary: Optional[Mapping[str, Any]]) -> DatasetBrief:
    """
    Convert the loose 'summary' dict (as used in templates) into a DatasetBrief.
    Handles missing fields gracefully.
    """
    if not summary:
        return DatasetBrief(filename="unknown")

    filename = _safe_str(summary.get("filename", "unknown"))
    rows, cols = None, None
    shape = summary.get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        try:
            rows = int(shape[0])
            cols = int(shape[1])
        except Exception:
            rows, cols = None, None

    delimiter = _normalize_delimiter(summary.get("delimiter_used"))
    encoding = _safe_str(summary.get("encoding_used"))

    # columns might be a list of names or richer dicts from your analysis
    columns_iter = []
    cols_val = summary.get("columns")
    if isinstance(cols_val, Iterable):
        for c in cols_val:
            if isinstance(c, Mapping):
                columns_iter.append(
                    ColumnBrief(
                        name=_safe_str(c.get("name", "")) or _safe_str(c.get("column", "")),
                        dtype=_safe_str(c.get("dtype", "")),
                        nullable=_to_bool_or_none(c.get("is_nullable")),
                    )
                )
            else:
                columns_iter.append(ColumnBrief(name=_safe_str(c), dtype=""))
    else:
        columns_iter = None

    return DatasetBrief(
        filename=filename,
        rows=rows,
        cols=cols,
        delimiter=delimiter,
        encoding=encoding,
        columns=columns_iter,
    )


def _normalize_delimiter(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val)
    if s == "\t":
        return "TAB (\\t)"
    if s == ",":
        return "COMMA (,)"
    if s == ";":
        return "SEMICOLON (;)"
    if s == "|":
        return "PIPE (|)"
    return s


def _safe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        s = str(v)
        return s if s.strip() else None
    except Exception:
        return None


def _to_bool_or_none(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v in ("True", "true", "1"):
        return True
    if v in ("False", "false", "0"):
        return False
    return None