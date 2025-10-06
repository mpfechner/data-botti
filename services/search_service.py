from __future__ import annotations
from typing import Optional
import numpy as np
from services.embeddings import embed_query
from services.models import QARecord, QueryRequest
from repo import repo_qa_find_by_hash, repo_qa_semantic_candidates, get_engine
from sqlalchemy import text

SEMANTIC_THRESHOLD = 0.90

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
                fh = conn.execute(text("SELECT file_hash FROM dataset_files WHERE dataset_id = :ds LIMIT 1"), {"ds": dataset_id}).scalar()
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
        # truncate to top_k if needed
        candidates = candidates[:top_k] if isinstance(candidates, list) else candidates
        if not candidates:
            return {"found": False, "decision": "semantic", "records": []}

        # candidates come as tuples: (QARecord, dim, vec_bytes); drop non-serializable parts
        cleaned = []
        if isinstance(candidates, list):
            for item in candidates:
                try:
                    rec, *_rest = item  # ignore dim and bytes
                except Exception:
                    rec = item
                cleaned.append(SearchService._serialize_qa_record(rec))
        return {"found": True, "decision": "semantic", "records": cleaned}

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