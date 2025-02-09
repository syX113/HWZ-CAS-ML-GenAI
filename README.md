# HWZ-CAS-ML-GenAI: Setup der Python Umgebung

Diese Anleitung erklärt, wie du deine Entwicklungsumgebung einrichtest, um die Python‑Dateien in diesem Repository auf einem Laptop (ohne dedizierte GPU) auszuführen – sowohl mit einer klassischen virtuellen Umgebung (venv) als auch mit einer Conda‑Umgebung.

---

## Inhaltsverzeichnis

- [1. Voraussetzungen](#1-voraussetzungen)
- [2. Einrichtung einer virtuellen Umgebung mit Python (venv)](#2-einrichtung-einer-virtuellen-umgebung-mit-python-venv)
  - [2.1. Virtuelle Umgebung erstellen](#21-virtuelle-umgebung-erstellen)
  - [2.2. Virtuelle Umgebung aktivieren](#22-virtuelle-umgebung-aktivieren)
  - [2.3. Installation der benötigten Python-Pakete](#23-installation-der-benötigten-python-pakete)
  - [2.4. Ausführen der Python-Dateien](#24-ausführen-der-python-dateien)
- [3. Einrichtung einer Conda-Umgebung](#3-einrichtung-einer-conda-umgebung)
  - [3.1. Conda-Umgebung erstellen](#31-conda-umgebung-erstellen)
  - [3.2. Conda-Umgebung aktivieren](#32-conda-umgebung-aktivieren)
  - [3.3. Installation der benötigten Pakete](#33-installation-der-benötigten-pakete)
  - [3.4. Ausführen der Python-Dateien](#34-ausführen-der-python-dateien)
- [4. Installation von Ollama](#4-installation-von-ollama)
  - [4.1. Voraussetzungen und Hinweise](#41-voraussetzungen-und-hinweise)
  - [4.2. Installation auf macOS](#42-installation-auf-macos)
  - [4.3. Installation auf Windows/Linux](#43-installation-auf-windowslinux)
- [5. Zusätzliche Hinweise](#5-zusätzliche-hinweise)
- [6. Fehlerbehebung und weitere Ressourcen](#6-fehlerbehebung-und-weitere-ressourcen)

---

## 1. Voraussetzungen

- Python 3.11 oder höher (überprüfe mit: `python --version` oder `python3 --version`)
- Git (optional, falls das Repository geklont wird)
- Internetverbindung zum Herunterladen der Pakete  
- Keine dedizierte GPU erforderlich – die Ausführung erfolgt vollständig auf der CPU

---

## 2. Einrichtung einer virtuellen Umgebung mit Python (venv)

### 2.1. Virtuelle Umgebung erstellen

Navigiere im Terminal in das Repository-Verzeichnis und führe folgenden Befehl aus:

```bash
python3 -m venv venv-hwz-genai
```

### 2.2. Virtuelle Umgebung aktivieren

- **Auf Linux/macOS:**
  ```bash
  source venv-hwz-genai/bin/activate
  ```
- **Auf Windows:**
  ```cmd
  venv-hwz-genai\Scripts\activate
  ```

### 2.3. Installation der benötigten Python-Pakete

Mit aktivierter Umgebung installiere die Pakete (ohne GPU‑spezifische Optionen):

```bash
pip install numpy PyPDF2 sentence-transformers transformers torch scikit-learn openai ollama
```

### 2.4. Ausführen der Python-Dateien


- **Für die Sentimentanalyse (Aufgabe_4.py):**
  ```bash
  python Aufgabe_3.py
  ```

- **Für die OpenAI Integration (Aufgabe_5.py):**
  ```bash
  python Aufgabe_4.py
  ```
- **Für das Retrieval Augmented Generation System (Aufgabe_6.py):**
  ```bash
  python Aufgabe_5.py
  ```
---

## 3. Einrichtung einer Conda-Umgebung

### 3.1. Conda-Umgebung erstellen

Erstelle eine neue Conda‑Umgebung (passe ggf. den Namen an):

```bash
conda create -n venv-hwz-genai python=3.11
```

### 3.2. Conda-Umgebung aktivieren

Aktiviere die Umgebung mit:

```bash
conda activate venv-hwz-genai
```

### 3.3. Installation der benötigten Pakete

Installiere die Pakete entweder über Conda oder pip:

```bash
conda install numpy
conda install -c conda-forge pypdf2
pip install sentence-transformers transformers torch scikit-learn
```

### 3.4. Ausführen der Python-Dateien

- **Für die Sentimentanalyse (Aufgabe_4.py):**
  ```bash
  python Aufgabe_3.py
  ```
- **Für die OpenAI Integration / nutzen der OpenAI API (Aufgabe_5.py):**
  ```bash
  python Aufgabe_4.py
  ```
- **Für das Retrieval Augmented Generation System (Aufgabe_6.py):**
  ```bash
  python Aufgabe_5.py
  ```

---

## 4. Installation von Ollama

### 4.1. Voraussetzungen und Hinweise

Ollama ermöglicht den lokalen Aufruf eines LLM über die Kommandozeile. Für Laptops ohne GPU reicht die CPU‑Version aus. Für weiterführende Informationen besuche bitte die offizielle Webseite.

### 4.3. Installation von Ollama

- **Windows:** Lade den Installer von der offiziellen Webseite [Ollama](https://ollama.com/download/windows) herunter und folge den Installationsanweisungen.  
- **MacOS:** Lade den Installer von der offiziellen Webseite [Ollama](https://ollama.com/download/mac) herunter und folge den Installationsanweisungen.  
- **Linux:** Installation via: ```curl -fsSL https://ollama.com/install.sh | sh```

### 4.4. Überprüfung der Installation

Überprüfe, ob Ollama korrekt installiert wurde, indem du im Terminal eingibst:

```bash
ollama --version
```

### 4.5. Lokales Modell mit Ollama herunterladen und ausführen

Um ein lokales Modell herunterzuladen, ersetze <model-name> durch den gewünschten Modellnamen und führe folgenden Befehl aus:

```bash
ollama pull <model-name>
```

Starte das Modell anschließend mit:

```bash
ollama run <model-name>
```

Hinweis: Bitte beachte die Dokumentation von Ollama für verfügbare Modelle und weitere Optionen. Weitere Modelle findest du in der [Model Library](https://ollama.com/library).

Beispiel:
```bash
ollama run deepseek-r1:14b
ollama run llama3.2:1b
```

---

## 5. Zusätzliche Hinweise

- Stelle sicher, dass alle Pfade korrekt sind und du gegebenenfalls dein System neu startest, wenn Umgebungsvariablen aktualisiert wurden.  
- Sollten Probleme auftreten, konsultiere bitte die Dokumentation der jeweiligen Pakete und von Ollama.

---

## 6. Fehlerbehebung und weitere Ressourcen

- Falls Probleme bei der Installation oder Nutzung auftreten, vergewissere dich, dass:
  - Die Umgebung (venv oder Conda) korrekt aktiviert wurde.
  - Alle installierten Pakete in der richtigen Version vorhanden sind.
  - Die PATH-Variablen ggf. neu geladen wurden, nachdem Änderungen vorgenommen wurden.
- Schau in die Logdateien der Anwendungen:
  - Ollama (macOS: `~/.ollama/logs/server.log`, Linux: `journalctl -u ollama --no-pager`, Windows: `%LOCALAPPDATA%\Ollama\server.log`).
- Weitere Informationen und Neuigkeiten findest du auf der offiziellen Webseite von Ollama:  
  https://ollama.com/
- Falls du auf Probleme stösst, suche in den FAQ-Bereichen der genutzten Pakete oder kontaktiere den Support.
