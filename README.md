# DataBotti

DataBotti ist ein Data-Analyzing-Tool mit AI-Unterstützung.  
Aktueller Status: **Pre-MVP / Work in Progress** – grundlegende Funktionen sind vorhanden, die KI-Funktionen werden aktuell vorbereitet.  

---

## Features (bisher)
- 📊 Datenbank-Anbindung (MariaDB via Docker oder lokal mit SQL-Skript)  
- 🔍 Erste Standardfunktionen zur Datenanalyse  
- 🤖 Vorbereitung für AI-gestützte Analysen (OpenAI API)  

---

## Setup

### 1. Repository klonen
```
git clone https://github.com/dein-user/data-botti.git
cd data-botti
```

### 2. Umgebungsvariablen konfigurieren
Kopiere die Beispieldatei:
```
cp .env-example .env
```
Bearbeite `.env` und trage deine eigenen Werte ein (DB-User, Passwörter, API-Key).

### 3. Datenbank vorbereiten

**a) Mit Docker (empfohlen)**  
- Stelle sicher, dass dein `create_database.sql` im Verzeichnis **`docker/init/`** liegt.  
- Beim ersten Start des Containers wird das Skript automatisch ausgeführt und die Datenbank erstellt.  
- Starte die Container:
```
docker compose -f docker/docker-compose.yml up --build
```

**b) Ohne Docker (lokal)**  
- Installiere MariaDB/MySQL lokal.  
- Führe das Skript manuell aus:  
```
mysql -u <user> -p < databotti < docker/init/create_database.sql
```

### 4. Backend starten
- Mit Docker läuft das Backend direkt als Service.  
- Alternativ lokal starten (z. B. in PyCharm):
```
python app.py
```

---

## PyCharm-Workflow (lokal entwickeln)
Für die Arbeit in PyCharm gibt es ein Override-File, das nur die DB im Container laufen lässt:
```
docker compose -f docker/docker-compose.yml -f docker/docker-compose.pycharm.yml up -d
```
Dann kannst du das Backend lokal in PyCharm starten, während die Datenbank im Container läuft.

---

## Status & Roadmap
- ✅ Basisfunktionen laufen  
- 🔄 Integration der KI-Funktionen (OpenAI API)  
- ⏳ Vorbereitung von Tests und erster Release als MVP  