import os
import sys
# Add app root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.qa_service import make_query_request
from services.search_service import SearchService


def main():
    service = SearchService()

    tests = [
        # === Analysis-like (EN) ===
        ("analysis", "Show me a chart of sales trends for 2023"),
        ("analysis", "Plot sales over time"),
        ("analysis", "Visualize sales by month"),
        ("analysis", "Top 5 months by revenue"),
        ("analysis", "7 weakest months"),
        ("analysis", "Lowest values by month"),
        ("analysis", "Rank by profit, show top 3"),
        # === Analysis-like (DE) ===
        ("analysis", "Zeige mir die 7 schwächsten Monate"),
        ("analysis", "Erstelle ein Diagramm der Umsätze 2023"),
        ("analysis", "Verlauf der Verkäufe nach Monat"),
        ("analysis", "Gruppiere nach Monat und sortiere aufsteigend"),
        # === QA-like (EN) ===
        ("qa", "What is the capital of France?"),
        ("qa", "How many rows are in the dataset?"),
        ("qa", "Define the revenue column"),
        ("qa", "Explain the dataset structure"),
        # === QA-like (DE) ===
        ("qa", "Was ist die Hauptstadt von Frankreich?"),
        ("qa", "Wie viele Zeilen hat das Dataset?"),
        ("qa", "Definiere die Spalte Umsatz"),
        ("qa", "Erkläre den Aufbau des Datasets"),
    ]

    for expected, text in tests:
        req = make_query_request(text, file_hash="intent-demo")
        intent = service.detect_intent(req)
        mark = "✅" if intent == expected else "❌"
        analysis = getattr(req, 'intent_scores', {}).get('analysis', None)
        qa = getattr(req, 'intent_scores', {}).get('qa', None)
        if analysis is not None and qa is not None:
            print(f"[{mark}] expected={expected} got={intent} | analysis={analysis:.3f} | qa={qa:.3f} | q='{text}'")
        else:
            print(f"[{mark}] expected={expected} got={intent} | score={req.intent_score:.3f} | q='{text}'")


if __name__ == "__main__":
    main()