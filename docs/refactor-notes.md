# DataBotti Refactor Notes

## Ziel
Saubere Trennung von Schichten:
- `routes/` = HTTP-Endpoints (Flask Blueprints)
- `services/` = Fachlogik, AI-Logik, CSV-Handling
- `infra/` = Querschnitt (Config, Logging)
- `app.py` = nur App-Start & Blueprint-Registrierung

---

## Verantwortungen

### Routes
- `routes/datasets.py` → Upload, Analyse-Start, Summary-Ansicht
- `routes/assistant.py` → AI-Endpoints (Prompt, Result, Legacy-Redirect)

### Services
- `dataset_service.py` → CSV/Dataset-Handling (laden, validieren, summary)
- `insights.py` → generische Analysen & Warnungen
- `ai_tasks.py` → Prompt-Building
- `ai_router.py` → Modellwahl (choose_model)
- `ai_client.py` → API-Call zum Modell
- `storage.py` → File Handling (Upload/Save/Delete)

### Infra
- `config.py` → Settings/ENV
- `logging.py` → Logger-Setup

---

## Fehlerbehandlung & Logging
- Routes: Request validieren, Exceptions → HTTP-Codes (400/404/500)
- Services: fachliche Exceptions, Logging DEBUG/INFO
- AI-Client: Logging von Modell, Dauer, Status (kein Prompt im Klartext)

---

## Endpoints
- `/` → Upload-Form / Upload
- `/analyze/dataset/<id>` → Analyse-Ansicht
- `/ai/<id>` → AI-Prompt + Result
- `/ai-legacy/<id>` → Redirect zu `/ai/<id>`
- `/privacy` → statisch

---

## Migrationsfolge
1. AI-Flow nach `routes/assistant.py`
2. Analyse-Route nach `routes/datasets.py`
3. Upload nach `routes/datasets.py`
4. `dataset_service`: CSV-Handling aus helpers lösen
5. AI-Schicht (tasks → router → client) säubern
6. Alte Helpers deprecaten

---

## DoD pro Schritt
- App läuft, gleiche URLs funktionieren
- Logs schreiben in `logs/databotti.log`
- Keine direkten `os.getenv` in Routes/Services (außer `ai_client` für Key)
- Manuell: Upload → Analyze → AI durchspielbar

## Fortschritt (Checkliste)
- [x] Schritt 0: Branch erstellt
- [x] Schritt 1: Struktur + Platzhalter (inkl. dataset_service.py, ai_router.py)
- [x] Schritt 2: Blueprints registriert, App startet wie zuvor
- [x] Schritt 3: infra/config.py & infra/logging.py eingebunden
- [x] Schritt 4: AI-Layer Inventur dokumentiert


## AI-Layer Inventur (Ist-Zustand)
- `build_chat_prompt` wird **nur** an einer Stelle importiert und dort bei Importfehler lokal als Fallback definiert (Try/Except). → Aktuell **kein** zentraler Einsatz in Routen ersichtlich.
- `choose_model(expected_output, cache_ratio)` wird in `services/ai_client.ask_model` verwendet (Import: `from .ai_router import choose_model`).
- Der **eigentliche Modellaufruf** passiert in `services/ai_client.ask_model` via `OpenAI(...).chat.completions.create(...)`.
- API-Key-Quelle: `OPENAI_API_KEY` **oder** `OPENAI_API_KEY_BOTTI` (Env). → Konfig in `infra/config` optional, aber derzeit **direkter** Zugriff via `os.getenv` in `ai_client`.
- Fehlerszenarien: Kein Key → `AIClientNotConfigured`; fehlende `openai`-Lib → Textlicher Hinweis; leere Antwort → interner **Retry** mit angepasstem Modell/`max_tokens`.
- **Unklar/TBD:** Wo genau `ask_model(...)` und/oder `build_chat_prompt(...)` aktuell aufgerufen werden (Routen oder Hilfsfunktionen). → Nächster Schritt: Code-Stellen identifizieren und migrieren.

## Master-Checkliste (Refactoring gesamt)

### Etappe 1 – Struktur & Entkopplung
- [x] Schritt 0: Branch erstellt (`refactor/structure`)
- [x] Schritt 1: Struktur + Platzhalter (inkl. `dataset_service.py`, `ai_router.py`)
- [x] Schritt 2: Blueprints registriert, App startet wie zuvor
- [x] Schritt 3: `infra/config.py` & `infra/logging.py` eingebunden
- [ ] Schritt 4: AI-Layer Inventur dokumentiert (`ai_tasks`/`ai_router`/`ai_client` Usage Map)

### Etappe 2 – Migration Endpoints & Services
**Routes /assistant**
- [ ] `/ai/<id>` GET: Prompt-Form in `routes/assistant.py`
- [ ] `/ai/<id>` POST: Orchestrierung `ai_tasks` → `ai_router` → `ai_client`
- [ ] (Legacy) `/ai-legacy/<id>`: Redirect oder entfernen

**Routes /datasets**
- [ ] `/analyze/dataset/<id>` GET: nutzt `dataset_service` + `insights`, rendert `result.html`
- [ ] Upload POST (`/` oder `/upload`): Speichern via `storage.py`, Redirect auf Analyse

**Services**
- [ ] `dataset_service`: `load_csv()` (Encoding/Dialekt), `summary()` (shape/columns/describe)
- [ ] `insights`: generische Befunde (Ausreißer, Verteilungen, NaNs)
- [ ] `storage`: `save_file()`, `delete_file()`, Verzeichnis-Existenz sicherstellen
- [ ] `ai_client`: Modellaufruf kapseln (Key/Timeout/Errors/Logging) – Config nutzen
- [ ] `ai_tasks`: Prompt-Building konsolidieren (tote Pfade entfernen)
- [ ] `ai_router`: `choose_model()` Integration & Parametereingang prüfen

**App Cleanup**
- [ ] Alte AI/Analyse-Routen aus `app.py` entfernen
- [ ] `app.py` enthält nur App-Start + Blueprint-Registrierung

**Templates/Static**
- [ ] Templates an neue Endpoints angepasst (`index.html`, `result.html`)

**Config/Logging Konsistenz**
- [x] `infra` integriert
- [ ] Keine direkten `os.getenv` in Routes/Services (Ausnahme: `ai_client`, falls Key nicht via Config gereicht)

**Manuelle Tests**
- [ ] Upload → Analyse-Ansicht funktioniert (kleine CSV)
- [ ] AI-Prompt → Antwort wird angezeigt
- [ ] Logs landen in `logs/databotti.log` (Rotation aktiv)

### Etappe 3 – Feinschliff & Housekeeping
- [ ] Unbenutzte Helpers entfernen/deprecaten
- [ ] README aktualisieren (Struktur, ENV-Variablen, Start-Hinweise)
- [ ] `requirements.txt` prüfen/aufräumen
- [ ] Optionale Error-Pages (400/500) hinzufügen
- [ ] Docstrings & Type Hints an zentralen Stellen ergänzen
- [ ] Lint/PEP8-Pass (nur stilistisch, keine Logik-Änderung)