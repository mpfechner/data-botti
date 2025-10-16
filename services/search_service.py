from __future__ import annotations
from typing import Optional
import numpy as np
import unicodedata
import re
# Optional cross-encoder for semantic re-ranking (multilingual)
import logging
# Optional cross-encoder for semantic re-ranking (multilingual)
try:
    from sentence_transformers import CrossEncoder  # type: ignore
    _CE_AVAILABLE = True
except Exception:
    CrossEncoder = None  # type: ignore
    _CE_AVAILABLE = False

_ce_model = None
logger = logging.getLogger(__name__)

def _get_cross_encoder():
    """
    Lazy-load a multilingual cross-encoder for better ranking of short QA prompts.
    Tries a list of well-performing multilingual CE models. If HF is offline but
    the repo is cached locally, loading via repo-id still works.
    """
    import os
    global _ce_model
    if not _CE_AVAILABLE:
        return None
    if _ce_model is not None:
        return _ce_model

    HF_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0") == "1"
    mode = "OFFLINE (cache only)" if HF_OFFLINE else "ONLINE (may contact HF Hub)"
    logger.info("[rerank] CrossEncoder load mode: %s", mode)

    # Preferred → fallback list
    candidates = [
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",     # multilingual mMiniLMv2 (small, fast)
    ]

    for repo_id in candidates:
        try:
            if HF_OFFLINE:
                # Strictly offline: only load from local HF cache
                logger.info("[rerank] loading CrossEncoder %r with local_files_only=True", repo_id)
                _ce_model = CrossEncoder(
                    repo_id,
                    local_files_only=True,           # pass only here (do not duplicate in kwargs)
                    trust_remote_code=False,
                    model_kwargs={},                 # replacement for deprecated automodel_args
                    tokenizer_kwargs={}              # replacement for deprecated tokenizer_args
                )
            else:
                # Online allowed: will download on first run, then use cache
                logger.info("[rerank] loading CrossEncoder %r (online allowed)", repo_id)
                _ce_model = CrossEncoder(
                    repo_id,
                    trust_remote_code=False,
                    model_kwargs={},
                    tokenizer_kwargs={}
                )
            logger.info("[rerank] CrossEncoder ready: %s", repo_id)
            return _ce_model
        except OSError as e:
            # Typical when offline and model not in cache
            logger.warning("[rerank] OSError loading %s: %s", repo_id, e)
            continue
        except Exception as e:
            logger.warning("[rerank] failed to load %s: %s", repo_id, e)
            continue

    logger.warning("[rerank] no CrossEncoder available; continuing without CE")
    return None
from services.embeddings import embed_query
from services.models import QARecord, QueryRequest
from types import SimpleNamespace
from repo import repo_qa_find_by_hash, repo_embedding_candidates_for_file, get_engine
from sqlalchemy import text

SEMANTIC_THRESHOLD = 0.90
SUGGEST_MIN_SCORE = 0.75  # more forgiving floor for suggestions
SUGGEST_WINDOW = 0.12     # wider band to reduce false negatives

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


# --- Levenshtein similarity for typo-tolerant matching ---
def _lev_ratio(a: str, b: str) -> float:
    """
    Normalized Levenshtein similarity in [0,1].
    1.0 = identical, 0.0 = completely different.
    Uses a standard dynamic-programming edit distance and normalizes by max length.
    Suitable for short UI queries.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    # ensure a is the shorter for a tiny speed boost
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    # initialize previous row
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        curr = [j] + [0] * la
        bj = b[j - 1]
        for i in range(1, la + 1):
            cost = 0 if a[i - 1] == bj else 1
            curr[i] = min(
                prev[i] + 1,       # deletion
                curr[i - 1] + 1,   # insertion
                prev[i - 1] + cost # substitution
            )
        prev = curr
    dist = prev[la]
    denom = max(la, lb)
    return 1.0 - (dist / denom if denom else 0.0)


# --- Aggressive Normalisierung für Near-Exact-Matching ---
def _normalize_text(s: str) -> str:
    """
    Aggressive Normalisierung für Near-Exact-Matching:
    - Unicode NFKC
    - typografische Striche/Quotes → ASCII
    - lowercase
    - Satzzeichen (.,;:!?()[]{}"' ) entfernen, Bindestrich behalten
    - Whitespaces kollabieren
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-").replace("‚", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.lower()
    s = re.sub(r"[.,;:!?\(\)\[\]\{\}\"']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
    def suggest_similar_questions(prompt: str, dataset_id: int, user_id: int, top_k: int = 8, include_seeds: bool = True) -> dict:
        logger.info("[suggest] q=%r ds=%s top_k=%s include_seeds=%s", prompt, dataset_id, top_k, include_seeds)
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
                logger.info("[suggest] unauthorized user_id=%s for dataset_id=%s", user_id, dataset_id)
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
            logger.info("[suggest] exact hit id=%s", getattr(exact, "id", None))
            return {"found": True, "decision": "exact", "records": [SearchService._serialize_qa_record(exact)]}

        # NEW: Exact/Prefix text match against normalized question text before embeddings
        try:
            qnorm = getattr(request, "question_norm", None)
            if qnorm and getattr(request, "file_hash", None):
                with engine.connect() as conn:
                    # Prefer exact normalized match, then prefix-like; order exact first, then shortest
                    rows = conn.execute(text(
                        """
                        SELECT id, question_original AS question, answer
                        FROM qa_pairs
                        WHERE file_hash = :fh
                          AND (
                              question_norm = :qnorm
                              OR question_norm LIKE :qprefix
                          )
                        ORDER BY (question_norm = :qnorm) DESC, CHAR_LENGTH(question_norm) ASC
                        LIMIT :lim
                        """
                    ), {"fh": request.file_hash, "qnorm": qnorm, "qprefix": qnorm + "%", "lim": top_k}).fetchall()
                logger.info("[suggest] text-match rows=%d", len(rows))
                if rows:
                    logger.info("[suggest] returning text-match top_ids=%s", [r.id for r in rows[:3]])
                    recs = [{"id": r.id, "question": r.question, "answer": r.answer} for r in rows]
                    return {"found": True, "decision": "text", "records": recs}
        except Exception:
            pass

        # Schritt 3: Embedding & semantische Suche
        embedding = embed_query(f"query: {prompt}")
        if not getattr(request, "file_hash", None):
            return {"found": False, "decision": "semantic", "records": []}
        candidates = repo_embedding_candidates_for_file(
            file_hash=request.file_hash,
            model="intfloat/multilingual-e5-base",
            include_seeds=include_seeds,
            limit=200
        )
        logger.info("[suggest] embedding candidates=%d for file_hash=%s include_seeds=%s", len(candidates) if candidates else 0, request.file_hash, include_seeds)
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
                # Accept dict-shaped candidates from repo_embedding_candidates_for_file
                rec = None
                vec_bytes = None
                if isinstance(item, dict):
                    rid = item.get("id") or item.get("qa_id")
                    qtext = item.get("question") or item.get("question_norm") or ""
                    vec_bytes = item.get("vec") or item.get("embedding") or item.get("vector")
                    rec = SimpleNamespace(id=rid, question=qtext, answer=None)
                    if rid is None:
                        logger.warning("[suggest] candidate with missing id/qa_id: %r", item)
                else:
                    # Backward compatibility with tuple payloads: (rec, dim, vec_bytes)
                    try:
                        rec, _dim, vec_bytes = item
                    except Exception:
                        # Last resort: treat item as record-like and try vec attributes
                        rec = item
                        vec_bytes = getattr(item, "vec", None) or getattr(item, "embedding", None)

                if vec_bytes is None:
                    continue
                cand_vec = np.frombuffer(vec_bytes, dtype=np.float32)
                if cand_vec.size == 0:
                    continue

                denom = (np.linalg.norm(query_vec) * np.linalg.norm(cand_vec))
                if denom == 0:
                    continue

                score = float(np.dot(query_vec, cand_vec) / denom)
                adjusted = max(0.0, min(0.9999, score))
                scored.append((rec, adjusted))

                # Lexical overlap tokens between query and candidate (QUESTION ONLY)
                q_text = getattr(rec, "question", None) or ""
                r_tokens = _sig_tokens(q_text)
                overlap_size = len(q_tokens & r_tokens)
                lexical_hits.append((rec, adjusted, overlap_size))
            except Exception:
                continue
        logger.info("[suggest] cosine_scored=%d", len(scored))

        # Log top-5 cosine before filters
        try:
            top5_cos = sorted(scored, key=lambda rs: rs[1], reverse=True)[:5]
            logger.info("[suggest] cosine_top5=%s", [(
                getattr(r, "id", None) or getattr(r, "qa_id", None) or getattr(r, "question", None)[:24],
                round(float(s), 4)
            ) for r, s in top5_cos])
        except Exception:
            pass

        if not scored:
            # No embedding candidates available → try DB lexical fallback
            db_hits = _db_lexical_fallback()
            if db_hits:
                return {"found": True, "decision": "semantic", "records": db_hits}
            return {"found": False, "decision": "semantic", "records": []}

        best_score = max(s for _r, s in scored)
        soft_cut = max(SUGGEST_MIN_SCORE, best_score - SUGGEST_WINDOW)
        filtered = [(r, s) for (r, s) in scored if s >= soft_cut]
        logger.info("[suggest] filtered_by_soft_cut=%d (soft_cut=%.4f best=%.4f)", len(filtered), soft_cut, best_score)

        # Lexical gate: allow if there's *any* significant token overlap OR the cosine score is very close to the best
        filtered2 = []
        for rec, s in filtered:
            q_text = getattr(rec, "question", None) or ""
            r_tokens = _sig_tokens(q_text)
            overlap = q_tokens & r_tokens
            if overlap or s >= (best_score - 0.02):
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
                # --- NEW: Top-k cosine fallback when everything else yields no hit ---
                if scored:
                    top_scored = sorted(scored, key=lambda rs: rs[1], reverse=True)[:top_k]
                    cleaned = []
                    for best_rec, best_s in top_scored:
                        item = SearchService._serialize_qa_record(best_rec)
                        try:
                            item["score"] = round(float(best_s), 4)
                        except Exception:
                            pass
                        cleaned.append(item)
                    return {"found": bool(cleaned), "decision": "semantic", "records": cleaned}
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
        logger.info("[suggest] deduped=%d", len(deduped))

        # ----- Cross-encoder re-ranking (semantic pairwise scoring) -----
        ce_used = False
        try:
            ce = _get_cross_encoder()
            if ce is not None and deduped:
                # Prepare up to N best cosine candidates for CE scoring
                N = min(20, len(deduped))
                # Sort by cosine score first to pick the top-N
                deduped_sorted = sorted(deduped, key=lambda rs: rs[1], reverse=True)
                top_for_ce = deduped_sorted[:N]
                logger.info("[rerank] input_pairs=%d", len(top_for_ce))

                pairs = []
                ce_keys = []
                for rec, s in top_for_ce:
                    q_text = getattr(rec, "question", "") or ""
                    pairs.append((prompt, q_text))
                    # Prefer stable id if available, else tuple key
                    key = getattr(rec, "id", None)
                    if key is None:
                        key = (getattr(rec, "question", None), getattr(rec, "answer", None))
                    ce_keys.append(key)

                # Predict similarity scores with CE (higher is better)
                ce_scores = ce.predict(pairs)
                # Debug: log CE raw scores (top-5)
                try:
                    preview = list(zip(ce_keys, [float(x) for x in ce_scores]))[:5]
                    logger.info("[rerank] ce_raw_top5=%s", preview)
                except Exception:
                    pass
                # Normalize CE scores to [0,1] for blending
                import math
                ce_min = float(np.min(ce_scores)) if len(ce_scores) else 0.0
                ce_max = float(np.max(ce_scores)) if len(ce_scores) else 1.0
                denom = (ce_max - ce_min) if (ce_max - ce_min) > 1e-9 else 1.0
                ce_norm = [(float(s) - ce_min) / denom for s in ce_scores]

                # Build lookup and blend with cosine similarity
                ce_map = {k: v for k, v in zip(ce_keys, ce_norm)}
                BLEND_CE = 0.65  # weight for cross-encoder
                BLEND_COS = 0.35  # retain some cosine influence

                blended = []
                for rec, cos_s in deduped:
                    key = getattr(rec, "id", None)
                    if key is None:
                        key = (getattr(rec, "question", None), getattr(rec, "answer", None))
                    ce_s = ce_map.get(key, None)
                    if ce_s is None:
                        # If not in CE top-N, keep original cosine score
                        final_s = float(cos_s)
                    else:
                        final_s = BLEND_CE * float(ce_s) + BLEND_COS * float(cos_s)
                    blended.append((rec, final_s))
                ce_used = True
                logger.info("[rerank] blended=%d", len(blended))
                # Log top-5 after blending (by score)
                try:
                    top5_blend = sorted(blended, key=lambda rs: rs[1], reverse=True)[:5]
                    logger.info("[rerank] blended_top5=%s", [(getattr(r, "id", None), round(float(s), 4)) for r, s in top5_blend])
                except Exception:
                    pass
                # Replace deduped with blended scores (keep ordering flexible for next boosts)
                deduped = blended
        except Exception:
            # If anything goes wrong with CE, keep cosine-only path
            pass

        # ----- Hard-cut on final similarity score (cosine-only or CE-blended) -----
        try:
            import os
            before_hard = len(deduped)
            if before_hard:
                if ce_used:
                    # More permissive when CE was applied; allow small lists through
                    hard_min = float(os.getenv("SEARCH_HARD_MIN_CE", "0.25"))
                    if before_hard <= 3:
                        logger.info("[suggest] hard_cut skipped (ce_used=True, small list=%d)", before_hard)
                        pass
                    else:
                        deduped_after = [(r, s) for (r, s) in deduped if float(s) >= hard_min]
                        logger.info("[suggest] hard_cut(CE) filtered=%d (>= %.3f) kept=%d", before_hard - len(deduped_after), hard_min, len(deduped_after))
                        if deduped_after:
                            deduped = deduped_after
                else:
                    # Cosine-only path: use the stricter threshold
                    hard_min = float(os.getenv("SEARCH_HARD_MIN", "0.58"))
                    deduped_after = [(r, s) for (r, s) in deduped if float(s) >= hard_min]
                    logger.info("[suggest] hard_cut filtered=%d (>= %.3f) kept=%d", before_hard - len(deduped_after), hard_min, len(deduped_after))
                    if deduped_after:
                        deduped = deduped_after
                    else:
                        # If everything got cut, keep a best-effort single candidate if close
                        best_rec, best_s = max(deduped, key=lambda rs: rs[1])
                        if float(best_s) >= (hard_min - 0.03):
                            deduped = [(best_rec, best_s)]
                        else:
                            deduped = []
        except Exception as e:
            logger.warning("[suggest] hard_cut skipped due to error: %s", e)

        # ----- Deterministic re-ranking: boost near-exact matches of the typed text -----
        try:
            q_norm = _normalize_text(prompt)
            boosted = []
            for rec, s in deduped:
                rq = getattr(rec, "question", "") or ""
                rq_norm = _normalize_text(rq)

                boost = 0.0
                hardpin = False
                if rq_norm == q_norm:
                    boost += 2.0  # exact text match wins
                    hardpin = True
                else:
                    # strong prefix/containment boost (either direction)
                    if rq_norm.startswith(q_norm) or q_norm.startswith(rq_norm):
                        boost += 1.2
                        hardpin = True
                    # substring containment of a meaningful span
                    if len(q_norm) >= 12 and q_norm in rq_norm:
                        boost += 0.8
                    # Jaccard token overlap
                    qtoks = set(q_norm.split())
                    rtoks = set(rq_norm.split())
                    if qtoks and rtoks:
                        jacc = len(qtoks & rtoks) / max(1, len(qtoks | rtoks))
                        if jacc >= 0.75:
                            boost += 0.5
                    # high character similarity (typos) using Levenshtein similarity
                    lev = _lev_ratio(rq_norm, q_norm)
                    # Stronger boost for near-identical strings (better than difflib for typo handling)
                    if lev >= 0.95:
                        boost += 0.5
                    elif lev >= 0.90:
                        boost += 0.3

                boosted.append((rec, float(s) + boost, hardpin, abs(len(rq_norm) - len(q_norm)), getattr(rec, "id", 0)))

            # Sort: hardpinned first, then score desc, then smaller length gap, then id asc
            boosted.sort(key=lambda x: (not x[2], -x[1], x[3], x[4]))
            top_pairs = [(rec, sc) for (rec, sc, _hp, _len_gap, _id) in boosted[:top_k]]
        except Exception:
            deduped.sort(key=lambda rs: rs[1], reverse=True)
            top_pairs = deduped[:top_k]
        # Final top-k with scores for traceability
        try:
            logger.info("[suggest] final_top=%s", [(getattr(rec, "id", None), round(float(sc), 4)) for rec, sc in top_pairs])
        except Exception:
            pass
        logger.info("[suggest] top_pairs_ids=%s", [getattr(rec, "id", None) for rec, _ in top_pairs])
        cleaned = []
        for rec, s in top_pairs:
            item = SearchService._serialize_qa_record(rec)
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
        query_vec = np.asarray(embed_query(f"query: {request.question_norm}"), dtype=np.float32)
        candidates = repo_embedding_candidates_for_file(
            file_hash=request.file_hash,
            model="intfloat/multilingual-e5-base",
            include_seeds=True,
            limit=200
        )
        best_score = -1.0
        best_rec = None
        for item in (candidates or []):
            try:
                rec = None
                vec_bytes = None
                if isinstance(item, dict):
                    # dict payload from repo_embedding_candidates_for_file
                    rid = item.get("id") or item.get("qa_id")
                    qtext = item.get("question") or item.get("question_norm") or ""
                    vec_bytes = item.get("vec") or item.get("embedding") or item.get("vector")
                    from types import SimpleNamespace
                    rec = SimpleNamespace(id=rid, question=qtext, answer=None)
                    if rid is None:
                        logger.warning("[search_semantic] candidate with missing id/qa_id: %r", item)
                else:
                    # tuple payload: (rec, dim, vec_bytes)
                    try:
                        rec, _dim, vec_bytes = item
                    except Exception:
                        rec = item
                        vec_bytes = getattr(item, "vec", None) or getattr(item, "embedding", None)

                if vec_bytes is None:
                    continue
                candidate_vec = np.frombuffer(vec_bytes, dtype=np.float32)
                if candidate_vec.size == 0:
                    continue

                denom = (np.linalg.norm(query_vec) * np.linalg.norm(candidate_vec))
                if denom == 0:
                    continue
                sim = float(np.dot(query_vec, candidate_vec) / denom)

                if sim > best_score:
                    best_score = sim
                    best_rec = rec
            except Exception:
                continue
        if best_score >= SEMANTIC_THRESHOLD and best_rec is not None:
            request.decision = "semantic"
            if hasattr(request, "badges") and isinstance(request.badges, list):
                request.badges.append("🔍 semantisch")
            return best_rec
        return None

    def detect_intent(self, request: QueryRequest) -> str:
        """Detect intent (qa vs analysis) using embedding similarity to prototypes."""
        query_vec = np.asarray(embed_query(f"query: {request.question_norm}"), dtype=np.float32)
        scores = {}
        for intent, examples in INTENT_PROTOTYPES.items():
            example_vecs = [np.asarray(embed_query(f"passage: {ex}"), dtype=np.float32) for ex in examples]
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