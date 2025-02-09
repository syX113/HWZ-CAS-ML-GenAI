#!/usr/bin/env python3
"""
Aufgabe 6:
Ausgangslage: Ziel ist es, ein kleines RAG-System (Retrieval Augmented Generation) lokal aufzubauen und mit 3-4 PDF-Dateien via LLM zu interagieren.
Dazu kann z.B. Ollama in Verbindung mit Python genutzt werden.

Tasks:
- Installiere Ollama und stelle sicher, dass es lokal verfügbar ist.
- Verarbeite PDFs (Text extrahieren, in sinnvolle Chunks aufteilen, Embeddings berechnen).
- Implementiere eine Suchfunktion, die den relevantesten Textabschnitt zu einer Benutzeranfrage findet.
- Rufe das LLM (über Ollama) auf, um anhand des gefundenen Kontexts eine Antwort zu generieren.

Wichtige Hinweise zu Ollama und DeepSeek-Modellen:
- Ollama ermöglicht das Herunterladen und Ausführen von LLMs lokal. Weitere Informationen auf: https://ollama.com/
- Zum Herunterladen eines DeepSeek-Modells richte folgende Befehle ein:
    # Beispiel: Laden des 7B-Modells
    ollama pull llama3.1-cpu-only
    # Beispiel: Ausführen des geladenen Modells
    ollama run llama3.1-cpu-only
- Ähnliche Befehle gelten für andere Modelle (z.B. 14b, 70b) und deren quantisierte bzw. destillierte Varianten.
- Zur Nutzung der Modelle über die API sieh dir die Ollama-Dokumentation an.
"""

import os
import glob
import sys
import time
import subprocess
import re
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer, util
from ollama import Client

# Initialisiere den Ollama-Client, der über HTTP kommuniziert.
client = Client(host="http://localhost:11434")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrahiert den Text aus einer PDF-Datei.

    :param pdf_path: Pfad zur PDF-Datei.
    :return: Extrahierter Text als String.
    """
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def smart_split_text_into_chunks(text: str, max_words: int = 300, overlap: int = 50) -> list:
    """
    Teilt den Text in sinnvolle Chunks auf, basierend auf Satzgrenzen.
    
    Diese Funktion nutzt eine reguläre Ausdrucks-Satztrennung, um den Text in Sätze zu zerlegen.
    Anschliessend werden Sätze zu Chunks zusammengefasst, die eine maximale Wortanzahl (max_words)
    nicht überschreiten. Ein definierter Overlap (überlap) wird zwischen den Chunks beibehalten,
    um Kontextverlust zu minimieren.
    
    :param text: Der zu teilende Text.
    :param max_words: Maximale Anzahl an Wörtern pro Chunk.
    :param overlap: Anzahl an Wörtern, die zwischen benachbarten Chunks überlappen.
    :return: Liste von Text-Chunks.
    """
    # Zerlege den Text in Sätze. Der Regex teilt am Ende von Sätzen (., !, ?), gefolgt von Leerzeichen.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_word_count = len(sentence.split())
        # Wenn das Hinzufügen des nächsten Satzes den Chunk überladen würde,
        # speichere den aktuellen Chunk und starte einen neuen mit einem Overlap.
        if current_word_count + sentence_word_count > max_words and current_chunk:
            chunks.append(current_chunk.strip())
            # Beginne den nächsten Chunk mit den letzten 'overlap'-Wörtern des aktuellen Chunks
            if overlap > 0:
                words = current_chunk.split()
                current_chunk = " ".join(words[-overlap:])
                current_word_count = len(current_chunk.split())
            else:
                current_chunk = ""
                current_word_count = 0
        # Füge den Satz zum aktuellen Chunk hinzu.
        current_chunk += " " + sentence
        current_word_count += sentence_word_count

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def call_ollama(prompt: str) -> str:
    """
    Ruft das LLM über den Ollama-Client auf, um eine Antwort zu generieren.

    :param prompt: Der Eingabe-Prompt, der an das Modell übermittelt wird.
    :return: Die generierte Antwort des Modells.
    """
    try:
        # Sende eine Chat-Anfrage an das Modell. Hier wird "llama3.2:1b" als Beispiel genutzt.
        response = client.chat(model="llama3.2:1b", messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Fehler bei der Abfrage des Ollama-Modells: {e}"


def start_and_setup_ollama() -> subprocess.Popen:
    """
    Lädt zunächst das gewünschte Modell herunter und startet es anschliessend über Ollama.
    Um Probleme mit der Terminal-Eingabe zu vermeiden (z.B. Blockierung von STDIN und Lag),
    wird der Prozess so gestartet, dass er nicht mit dem Terminal interagiert.
    
    :return: Den Popen-Prozess des gestarteten Ollama-Modells, oder None im Fehlerfall.
    """
    try:
        print("Lade Modell 'llama3.2:1b' herunter...")
        subprocess.run(["ollama", "pull", "llama3.2:1b"], check=True)
        print("Starte Modell 'llama3.2:1b'...")
        proc = subprocess.Popen(
            ["ollama", "run", "llama3.2:1b"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # Warte einige Sekunden, bis das Modell vollständig gestartet ist.
        time.sleep(5)
        return proc
    except Exception as e:
        print(f"Fehler beim Start von Ollama: {e}")
        return None


def main():
    """
    Hauptfunktion des Skripts.
    
    Ablauf:
    1. Wenn als Kommandozeilenargument "setup" übergeben wird, 
       wird ausschliesslich das Modell heruntergeladen und gestartet, ohne weitere Logik.
    2. Andernfalls folgt die vollständige RAG-Logik:
       - PDFs werden verarbeitet (Text extrahieren, in sinnvolle Chunks aufteilen).
       - Embeddings für alle Chunks werden berechnet.
       - Die Benutzeranfrage wird entgegengenommen und der relevanteste Chunk ermittelt.
       - Das LLM (über Ollama) wird aufgerufen, um eine Antwort basierend auf dem Kontext zu generieren.
    """
    # Prüfe, ob das Skript im Setup-Modus gestartet wurde.
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        ollama_process = start_and_setup_ollama()
        if ollama_process:
            print("Ollama wurde gestartet. Drücke Strg+C, um den Prozess zu beenden.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Beende Ollama-Prozess...")
                ollama_process.terminate()
        sys.exit(0)

    # Starte und richte Ollama ein (Modell herunterladen und starten)
    ollama_process = start_and_setup_ollama()
    if ollama_process is None:
        print("Das Modell konnte nicht gestartet werden. Beende das Programm.")
        sys.exit(1)

    # Verzeichnis, in dem die PDF-Dateien gespeichert sind
    pdf_folder = "pdfs"
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        print("Keine PDF-Dateien im Ordner 'pdfs' gefunden.")
        ollama_process.terminate()
        sys.exit(1)

    # Initialisiere das Embedding-Modell (zum Beispiel 'all-MiniLM-L6-v2')
    print("Lade das Embedding-Modell 'all-MiniLM-L6-v2'...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Liste zur Speicherung aller Text-Chunks aus den PDFs
    corpus_chunks = []
    print("Verarbeite PDF-Dateien...")
    for pdf_file in pdf_files:
        print(f"Verarbeite: {pdf_file}")
        text = extract_text_from_pdf(pdf_file)
        chunks = smart_split_text_into_chunks(text, max_words=300, overlap=50)
        corpus_chunks.extend(chunks)

    if not corpus_chunks:
        print("Keine Text-Chunks aus den PDFs extrahiert.")
        ollama_process.terminate()
        sys.exit(1)

    # Berechne Embeddings für alle Text-Chunks
    print("Berechne Embeddings für alle Text-Chunks...")
    corpus_embeddings_tensor = embedding_model.encode(corpus_chunks, convert_to_tensor=True)

    # Benutzerabfrage: Nehme die Frage entgegen.
    sys.stdout.flush()  # Stelle sicher, dass alle Ausgaben angezeigt werden.
    query = input("Bitte gib deine Frage ein: ")
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    # Berechne die Kosinus-Ähnlichkeit zwischen der Benutzeranfrage und allen Chunks.
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings_tensor)[0]
    top_result_idx = int(np.argmax(cosine_scores.cpu().numpy()))
    relevant_chunk = corpus_chunks[top_result_idx]

    print("\nRelevanter Kontext aus den PDFs:")
    print(relevant_chunk)

    # Erstelle einen Prompt für das LLM, der den gefundenen Kontext und die Benutzerfrage kombiniert.
    prompt = (
        f"Nutze den folgenden Kontext aus PDFs, um die Frage zu beantworten:\n\n"
        f"Kontext: {relevant_chunk}\n\n"
        f"Frage: {query}\n\n"
        f"Antwort:"
    )
    print("\nGeneriere Antwort mit dem LLM (Ollama)...")
    answer = call_ollama(prompt)
    print("\nAntwort vom LLM:")
    print(answer)

    # Beende den Ollama-Prozess, sofern dieser gestartet wurde.
    if ollama_process:
        print("Beende Ollama-Prozess...")
        ollama_process.terminate()


if __name__ == "__main__":
    main()
