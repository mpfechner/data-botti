from __future__ import annotations

import gzip
import csv
import re
import pandas as pd


def load_csv_resilient(file_path: str, preferred_encoding: str | None = None, preferred_delimiter: str | None = None):
    """Robustes Einlesen einer gz-gepackten CSV-Datei mit Fallbacks:
    - Encoding: preferred → utf-8 → latin-1
    - Delimiter: preferred → ',', ';', '\t', '|' → csv.Sniffer
    - Header-Erkennung: vergleicht die ersten ZWEI Zeilen; wenn 1. Zeile wie Daten aussieht → header=None + generische Spaltennamen
    - Cleanup: NA-Tokens, Whitespace, Dezimal-Kommas, duplizierte Headerzeilen
    Returns:
        df (pd.DataFrame), used_encoding (str), used_delimiter (str)
    """

    encodings = [e for e in [preferred_encoding, "utf-8", "latin-1"] if e]
    delim_candidates = [d for d in [preferred_delimiter, ",", ";", "\t", "|"] if d]

    last_err = None
    for enc in encodings:
        # Sample lesen (für Sniffer & Heuristiken)
        try:
            with gzip.open(file_path, "rt", encoding=enc, newline="") as _f:
                sample = _f.read(8192)
        except Exception as e:
            last_err = e
            continue

        # --- Delimiter bestimmen ---
        used_delim = None
        for delim in delim_candidates:
            try:
                with gzip.open(file_path, "rt", encoding=enc, newline="") as f_test:
                    test_df = pd.read_csv(f_test, delimiter=delim, engine="python", on_bad_lines="warn", nrows=50, header=0)
                if test_df.shape[1] > 1:
                    used_delim = delim
                    break
            except Exception as e:
                last_err = e
                continue
        if used_delim is None:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
                used_delim = dialect.delimiter
            except Exception:
                used_delim = ","

        # ---- Erste ZWEI Zeilen tokenisieren für Header-Heuristik
        def _tokenize_first_lines(delim: str):
            try:
                with gzip.open(file_path, "rt", encoding=enc, newline="") as f:
                    l1 = f.readline().rstrip("\n")
                    l2 = f.readline().rstrip("\n")
                r1 = next(csv.reader([l1], delimiter=delim))
                r2 = next(csv.reader([l2], delimiter=delim)) if l2 else []
                return r1, r2
            except Exception:
                return [], []

        first_tokens, second_tokens = _tokenize_first_lines(used_delim)

        num_re = re.compile(r"^\s*[-+]?\d+(?:[.,]\d+)?\s*$")
        time_re = re.compile(r"\d{1,2}:\d{2}")
        dateish_re = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")

        def _tok_type(s: str) -> str:
            s = (s or "").strip()
            if not s:
                return "empty"
            if num_re.fullmatch(s):
                return "num"
            if time_re.search(s) or dateish_re.search(s):
                return "dt"
            return "txt"

        def _looks_like_header(toks1: list[str], toks2: list[str]) -> bool:
            """Stärkerer Test:
            - Wenn >50% der Tokens in Zeile 1 num/datetime → eher DATEN.
            - Wenn 1. und 2. Zeile im Token-Typ-Muster stark übereinstimmen → eher DATEN.
            - Sonst Header.
            """
            if not toks1:
                return True  # lieber Header annehmen, falls wir nichts wissen

            types1 = [_tok_type(t) for t in toks1]
            frac_data = sum(t in ("num", "dt") for t in types1) / max(1, len(types1))
            if frac_data > 0.5:
                return False  # eher Daten

            if toks2 and len(toks2) == len(toks1):
                types2 = [_tok_type(t) for t in toks2]
                equal_types = sum(a == b for a, b in zip(types1, types2)) / max(1, len(types1))
                if equal_types >= 0.6:
                    return False  # zwei inhaltlich ähnliche Zeilen → Daten

            # Header-Indiz: mehrere „namenähnliche“ Tokens ohne Zahlen
            headerish = sum(t == "txt" for t in types1) / max(1, len(types1)) >= 0.6
            return headerish

        header_is_ok = _looks_like_header(first_tokens, second_tokens)

        # --- Datei einlesen (mit/ohne Header)
        try:
            if header_is_ok:
                with gzip.open(file_path, "rt", encoding=enc, newline="") as f_full:
                    df = pd.read_csv(f_full, delimiter=used_delim, engine="python", on_bad_lines="warn", header="infer")
                header_detected = True
            else:
                with gzip.open(file_path, "rt", encoding=enc, newline="") as f_full:
                    df = pd.read_csv(f_full, delimiter=used_delim, engine="python", on_bad_lines="warn", header=None)
                df.columns = [f"col_{i}" for i in range(df.shape[1])]
                header_detected = False
            used_enc = enc
        except Exception as e:
            last_err = e
            continue

        # ======= Cleanup-Phase (generisch) =======
        header_as_str = [str(c) for c in df.columns]
        try:
            mask_dupe_header = df.apply(lambda r: list(map(str, r.values.tolist())) == header_as_str, axis=1)
            if mask_dupe_header.any():
                df = df.loc[~mask_dupe_header].copy()
        except Exception:
            pass

        for col in df.select_dtypes(include=["object", "string"]).columns:
            try:
                df[col] = df[col].astype("string").str.strip()
            except Exception:
                pass

        try:
            df.replace({"NA": pd.NA, "NaN": pd.NA, "null": pd.NA, "": pd.NA}, inplace=True)
        except Exception:
            pass

        # Dezimalkommas → Punkte (spaltenweise Heuristik)
        decimal_pattern = re.compile(r"^\s*[-+]?\d{1,3}(?:[\d\.]*\d)?\,\d+\s*$")
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == "string":
                try:
                    frac = df[col].astype("string").str.fullmatch(decimal_pattern).mean()
                    if pd.notna(frac) and frac > 0.05:
                        df[col] = df[col].astype("string").str.replace(",", ".", regex=False)
                        # sanft numerisch konvertieren (coerce), aber nur übernehmen, wenn überwiegend numerisch
                        conv = pd.to_numeric(df[col], errors="coerce")
                        if conv.notna().mean() > 0.5:
                            df[col] = conv
                except Exception:
                    pass

        try:
            df.dropna(how="all", inplace=True)
        except Exception:
            pass

        # Flag für nachgelagerte UI
        try:
            df.attrs["header_detected"] = bool(header_detected)
        except Exception:
            pass

        return df, used_enc, used_delim

    raise RuntimeError(f"CSV konnte nicht gelesen werden. Letzter Fehler: {last_err}")