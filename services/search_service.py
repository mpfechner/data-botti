from __future__ import annotations
from typing import Optional
import numpy as np
from services.embeddings import embed_query
from services.models import QARecord, QueryRequest
from repo import repo_qa_find_by_hash, repo_qa_semantic_candidates, get_engine
from sqlalchemy import text

SEMANTIC_THRESHOLD = 0.90
SUGGEST_MIN_SCORE = 0.80  # stricter floor for suggestions
SUGGEST_WINDOW = 0.06     # slightly wider band to reduce false negatives

INTENT_PROTOTYPES = {
    "analysis": [
        "Zeige mir die 7 schwächsten Monate",
        "Top 5 Quartale",
        "die stärksten Monate",
        "niedrigste Werte",
        "höchste Umsätze",
        "sortiere nach Rang",
        "Top 5 months",
        "7 weakest months",
        "lowest values",
        "rank by",
        # English chart/trend phrases
        "Show a chart of sales",
        "Plot sales over time",
        "Visualize sales by month",
        "Trend of sales in 2023",
        "Time series of revenue",
        "Group by month and sort ascending",
        "Monthly trend chart",
        "Line chart over time",
        "Bar chart by month",
        "Top N by month",
        "Bottom N by month",
        "Top 5 months by revenue",
        "Top 5 Monate nach Umsatz",
        "Rank by profit",
        "Rang nach Gewinn",
        "Diagramm der Umsätze",
        "Verlauf der Verkäufe",
        "Trenddiagramm",
        "Zeitreihe",
        "Zeitreihen-Diagramm",
    ],
    "qa": [
        "Wie viele Zeilen hat das Dataset?",
        "Was bedeutet diese Spalte?",
        "Erkläre den Inhalt",
        "Wie hoch ist der Durchschnitt?",
        "What is the capital of France?",
        "Define column",
        "Explain dataset",
        "How many rows are in the dataset?",
        # Additional German QA prototypes
        "Was ist die Hauptstadt von Frankreich?",
        "Was bedeutet diese Kennzahl?",
        "Definiere die Spalte Umsatz",
        "Welche Werte kommen in dieser Spalte vor?",
        "Welche Spalten gibt es im Dataset?",
        "Wie viele Zeilen hat die Tabelle?",
        "Wie ist der Durchschnittswert von Umsatz?",
        # Additional English QA prototypes
        "What is the capital of Germany?",
        "What does this metric mean?",
        "Define the revenue column",
        "Which values appear in this column?",
        "Which columns exist in the dataset?",
        "How many rows does the table have?",
        "What is the average revenue?",
    ],
}

# --- simple lexical gate helpers for suggestion quality ------------------------------------
STOPWORDS = {
    # de
    "der", "die", "das", "und", "oder", "ein", "eine", "ist", "sind", "mit", "im", "in", "an", "am", "auf", "zu", "von", "für", "den", "dem", "des", "auch", "wie", "was", "wer", "wo", "wann", "warum",
    # en
    "the", "and", "or", "a", "an", "is", "are", "with", "in", "on", "to", "of", "for", "as", "by", "at", "be", "this", "that", "which",
}

def _sig_tokens(text: str) -> set[str]:
    import re
    if not text:
        return set()
    toks = re.findall(r"[\wÄÖÜäöüß]+", text.lower())
    return {t for t in toks if len(t) >= 4 and t not in STOPWORDS}

class SearchService:
    @staticmethod
    def _serialize_qa_record(rec):
        """Return a JSON-serializable dict for a QARecord-like object."""
        if hasattr(rec, "model_dump"):
            try:
                return rec.model_dump()
            except Exception:
                pass
        if hasattr(rec, "to_dict"):
            try:
                return rec.to_dict()
            except Exception:
                pass
        if isinstance(rec, dict):
            return rec
        return {
            "id": getattr(rec, "id", None),
            "question": getattr(rec, "question", None),
            "answer": getattr(rec, "answer", None),
        }

    @staticmethod
    def suggest_similar_questions(prompt: str, dataset_id: int, user_id: int, top_k: int = 3) -> dict:
        # Schritt 1: Berechtigung prüfen
        engine = get_engine()
        with engine.begin() as conn:
            result = conn.execute(text("""
                SELECT d.id FROM datasets d
                LEFT JOIN datasets_groups dg ON dg.dataset_id = d.id
                LEFT JOIN user_groups ug ON ug.group_id = dg.group_id
                WHERE d.id = :ds AND (d.user_id = :uid OR ug.user_id = :uid)
            """), {"uid": user_id, "ds": dataset_id}).fetchone()
            if not result:
                return {"found": False, "decision": "unauthorized", "records": []}

        from services.qa_service import make_query_request  # local import to avoid circular dependency
        # Schritt 2: Exact Match per Hash
        request = make_query_request(prompt)
        if not getattr(request, "file_hash", None):
            with engine.connect() as conn:
                fh = conn.execute(text("SELECT file_hash FROM dataset_files WHERE dataset_id = :ds ORDER BY id DESC LIMIT 1"), {"ds": dataset_id}).scalar()
                if fh:
                    request.file_hash = fh
        exact = None
        if getattr(request, "file_hash", None) and getattr(request, "question_hash", None):
            exact = repo_qa_find_by_hash(file_hash=request.file_hash, question_hash=request.question_hash)
        if exact:
            return {"found": True, "decision": "exact", "records": [SearchService._serialize_qa_record(exact)]}

        # Schritt 3: Embedding & semantische Suche
        embedding = embed_query(prompt)
        if not getattr(request, "file_hash", None):
            return {"found": False, "decision": "semantic", "records": []}
        candidates = repo_qa_semantic_candidates(file_hash=request.file_hash, model="intfloat/multilingual-e5-base")
        # --- Score candidates via cosine similarity and apply a soft threshold for suggestions ---
        query_vec = np.asarray(embedding, dtype=np.float32)
        if query_vec.size == 0:
            return {"found": False, "decision": "semantic", "records": []}

        # Pre-compute significant tokens from the query for a lexical fallback
        q_tokens = _sig_tokens(prompt)

        def _db_lexical_fallback() -> list[dict]:
            """Fallback: query recent QA pairs for this file_hash and rank by token overlap."""
            if not getattr(request, "file_hash", None):
                return []
            try:
                with engine.connect() as conn:
                    rows = conn.execute(text("""
                        SELECT id, question_original AS question, answer
                        FROM qa_pairs
                        WHERE file_hash = :fh
                        ORDER BY created_at DESC
                        LIMIT 200
                    """), {"fh": request.file_hash}).fetchall()
            except Exception:
                return []
            results: list[tuple[dict, float]] = []
            for rid, qtext, ans in rows:
                qtext = qtext or ""
                ans = ans or ""
                r_tokens = _sig_tokens(qtext + " " + ans)
                overlap = q_tokens & r_tokens
                has_long = any(len(t) >= 6 for t in overlap)
                if overlap and (has_long or len(overlap) >= 2):
                    # pseudo score by lexical overlap size normalized by query token count
                    pseudo = len(overlap) / max(1, len(q_tokens))
                    results.append(({"id": rid, "question": qtext, "answer": ans}, pseudo))
            results.sort(key=lambda it: it[1], reverse=True)
            top_hits = results[:top_k]
            cleaned: list[dict] = []
            for rec_dict, s in top_hits:
                rec_out = dict(rec_dict)
                try:
                    rec_out["score"] = round(float(s), 4)
                except Exception:
                    pass
                cleaned.append(rec_out)
            return cleaned

        scored: list[tuple[object, float]] = []
        lexical_hits: list[tuple[object, float, int]] = []  # (rec, score, overlap_size)
        for item in (candidates or []):
            try:
                rec, dim, vec_bytes = item  # expected shape from repo
            except Exception:
                # Fallback if repo returns only the record or a different tuple
                rec, dim, vec_bytes = (item, None, None)
            try:
                cand_vec = np.frombuffer(vec_bytes, dtype=np.float32) if vec_bytes is not None else None
                if cand_vec is None or cand_vec.size == 0:
                    continue
                denom = (np.linalg.norm(query_vec) * np.linalg.norm(cand_vec))
                if denom == 0:
                    continue
                score = float(np.dot(query_vec, cand_vec) / denom)
                scored.append((rec, score))

                # Lexical overlap tokens between query and candidate (question+answer)
                q_text = getattr(rec, "question", None) or ""
                a_text = getattr(rec, "answer", None) or ""
                r_tokens = _sig_tokens(q_text + " " + a_text)
                overlap_size = len(q_tokens & r_tokens)
                lexical_hits.append((rec, score, overlap_size))
            except Exception:
                continue

        if not scored:
            # No embedding candidates available → try DB lexical fallback
            db_hits = _db_lexical_fallback()
            if db_hits:
                return {"found": True, "decision": "semantic", "records": db_hits}
            return {"found": False, "decision": "semantic", "records": []}

        best_score = max(s for _r, s in scored)
        soft_cut = max(SUGGEST_MIN_SCORE, best_score - SUGGEST_WINDOW)
        filtered = [(r, s) for (r, s) in scored if s >= soft_cut]

        # Lexical gate: require at least one significant token overlap between query and candidate
        filtered2 = []
        for rec, s in filtered:
            q_text = getattr(rec, "question", None) or ""
            a_text = getattr(rec, "answer", None) or ""
            r_tokens = _sig_tokens(q_text + " " + a_text)
            overlap = q_tokens & r_tokens
            # require (i) at least one shared significant token AND (ii) either a long-token match or >=2 overlaps
            has_long = any(len(t) >= 6 for t in overlap)
            if overlap and (has_long or len(overlap) >= 2):
                filtered2.append((rec, s))
        filtered = filtered2
        if not filtered:
            # --- Fallback: lexical-only shortlist when semantic thresholding drops all ---
            # Pick candidates with any overlap (>=1) and rank by overlap size, then by cosine score
            lexical_only = [(rec, sc, ov) for (rec, sc, ov) in lexical_hits if ov >= 1]
            if not lexical_only:
                # No embedding-based lexical matches → try DB lexical fallback
                db_hits = _db_lexical_fallback()
                if db_hits:
                    return {"found": True, "decision": "semantic", "records": db_hits}
                # --- NEW: Top-1 cosine fallback when everything else yields no hit ---
                if scored:
                    best_rec, best_s = max(scored, key=lambda rs: rs[1])
                    item = SearchService._serialize_qa_record(best_rec)
                    try:
                        item["score"] = round(float(best_s), 4)
                    except Exception:
                        pass
                    return {"found": True, "decision": "semantic", "records": [item]}
                return {"found": False, "decision": "semantic", "records": []}
            lexical_only.sort(key=lambda rso: (rso[2], rso[1]), reverse=True)
            picked = [(rec, sc) for (rec, sc, _ov) in lexical_only[:top_k]]

            cleaned = []
            for rec, s in picked:
                item = SearchService._serialize_qa_record(rec)
                try:
                    item["score"] = round(float(s), 4)
                except Exception:
                    pass
                cleaned.append(item)
            return {"found": bool(cleaned), "decision": "semantic", "records": cleaned}

        # Deduplicate by record id (fallback to question text if id missing)
        seen = set()
        deduped = []
        for rec, s in filtered:
            key = getattr(rec, "id", None)
            if key is None:
                key = (getattr(rec, "question", None), getattr(rec, "answer", None))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((rec, s))

        # Sort by score descending and cut to top_k
        deduped.sort(key=lambda rs: rs[1], reverse=True)
        top = deduped[:top_k]

        cleaned = []
        for rec, s in top:
            item = SearchService._serialize_qa_record(rec)
            # Optionally include score for UI transparency (ignored by UI if not used)
            try:
                item["score"] = round(float(s), 4)
            except Exception:
                pass
            cleaned.append(item)

        return {"found": bool(cleaned), "decision": "semantic", "records": cleaned}

    """Main search orchestrator handling exact, intent-based analysis, semantic search, and fallback."""

    def search_exact(self, request: QueryRequest) -> Optional[QARecord]:
        """Perform exact search using file_hash + question_hash."""
        if not request.file_hash:
            raise ValueError("QueryRequest.file_hash is required for exact search")
        return repo_qa_find_by_hash(file_hash=request.file_hash, question_hash=request.question_hash)

    def search_fuzzy(self, request: QueryRequest):
        """Placeholder for fuzzy search (e.g., edit distance or trigram similarity)."""
        raise NotImplementedError("Fuzzy search not yet implemented")

    def search_semantic(self, request: QueryRequest):
        """Perform semantic search using embeddings and cosine similarity."""
        if not request.file_hash:
            raise ValueError("QueryRequest.file_hash is required for semantic search")
        query_vec = np.asarray(embed_query(request.question_norm), dtype=np.float32)
        candidates = repo_qa_semantic_candidates(file_hash=request.file_hash, model="intfloat/multilingual-e5-base")
        best_score = -1.0
        best_rec = None
        for rec, _dim, vec_bytes in candidates:
            candidate_vec = np.frombuffer(vec_bytes, dtype=np.float32)
            if candidate_vec.size == 0:
                continue
            sim = np.dot(query_vec, candidate_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(candidate_vec))
            if sim > best_score:
                best_score = sim
                best_rec = rec
        if best_score >= SEMANTIC_THRESHOLD and best_rec is not None:
            request.decision = "semantic"
            if hasattr(request, "badges") and isinstance(request.badges, list):
                request.badges.append("🔍 semantisch")
            return best_rec
        return None

    def detect_intent(self, request: QueryRequest) -> str:
        """Detect intent (qa vs analysis) using embedding similarity to prototypes."""
        query_vec = np.asarray(embed_query(request.question_norm), dtype=np.float32)
        scores = {}
        for intent, examples in INTENT_PROTOTYPES.items():
            example_vecs = [np.asarray(embed_query(ex), dtype=np.float32) for ex in examples]
            sims = [np.dot(query_vec, ev) / (np.linalg.norm(query_vec) * np.linalg.norm(ev)) for ev in example_vecs]
            scores[intent] = max(sims)
        tau_high = 0.68
        margin = 0.05
        analysis_score = scores.get("analysis", 0.0)
        qa_score = scores.get("qa", 0.0)
        # Apply small keyword bias to analysis_score if keywords present in raw question
        keywords = ["top", "bottom", "weakest", "rank", "diagramm", "chart", "plot", "trend", "verlauf", "zeitreihe", "visualize", "group by"]
        question_lower = request.question_raw.lower() if request.question_raw else ""
        if any(keyword in question_lower for keyword in keywords):
            analysis_score += 0.03
        if analysis_score >= tau_high and (analysis_score - qa_score) >= margin:
            best_intent = "analysis"
            best_score = analysis_score
        else:
            best_intent = "qa"
            best_score = qa_score
        request.intent = best_intent
        request.intent_score = best_score
        # In future, store full scores dict in request.intent_scores if supported; keyword bias applied
        return request.intent

    def search_orchestrated(self, request: QueryRequest) -> Optional[QARecord]:
        """Single entrypoint for search: exact → analysis intent → semantic → none (LLM fallback outside).
        Returns a QARecord if an exact or semantic match was found; otherwise returns None.
        Side effects: sets request.decision and may append badges.
        """
        # 1) Exact match first
        rec = self.search_exact(request)
        if rec is not None:
            request.decision = "exact"
            # Optional: mark badge for UI
            if hasattr(request, "badges") and isinstance(request.badges, list):
                request.badges.append("💾 aus Verlauf")
            return rec

        # 2) Intent detection (qa vs analysis)
        intent = self.detect_intent(request)
        if intent == "analysis":
            # Placeholder: analysis pipeline would run here (group-by, charts, etc.)
            request.decision = "analysis"
            if hasattr(request, "badges") and isinstance(request.badges, list):
                request.badges.append("📊 Analyse")
            # For now, we return None and let the caller handle analysis rendering.
            return None

        # 3) Otherwise, try semantic search; if none, mark for none decision
        rec = self.search_semantic(request)
        if rec is not None:
            # search_semantic already set decision and badge
            return rec
        # No semantic hit → next stage would be LLM (outside this method)
        request.decision = "none"
        return None