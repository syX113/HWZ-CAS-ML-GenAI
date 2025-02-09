#!/usr/bin/env python3
"""
Aufgabe 6:
Ausgangslage: Ziel ist es, ein kleines RAG-System (Retrieval Augmented Generation) lokal aufzubauen und mit 3-4 PDF-Dateien via LLM zu interagieren.
Dazu kann z.B. Ollama in Verbindung mit Python genutzt werden.

Tasks:
- Installiere Ollama und stelle sicher, dass es lokal verfügbar ist.
- Verarbeite PDFs (Text extrahieren, in Chunks aufteilen, Embeddings berechnen).
- Implementiere eine Suchfunktion, die den relevantesten Textabschnitt zu einer Benutzeranfrage findet.
- Rufe das LLM (über Ollama) auf, um anhand des gefundenen Kontexts eine Antwort zu generieren.

Wichtige Hinweise zu Ollama und DeepSeek-Modellen:
- Ollama ermöglicht das Herunterladen und Ausführen von LLMs lokal. Weitere Informationen auf: https://ollama.com/
- Zum Herunterladen eines DeepSeek-Modells richte folgende Befehle ein:
    # Beispiel: Laden des 7B-Modells
    ollama pull deepseek-r1:7b
    # Beispiel: Ausführen des geladenen Modells
    ollama run deepseek-r1:7b
- Ähnliche Befehle gelten für andere Modelle (z.B. 14b, 70b) und deren quantisierte bzw. destillierte Varianten.
- Zur Nutzung der Modelle über die API sieh dir die Ollama-Dokumentation an.
"""

import os
import glob
import subprocess
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer, util

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrahiert den Text aus einer PDF-Datei."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_text_into_chunks(text: str, chunk_size: int = 500) -> list:
    """Teilt den Text in kleinere Abschnitte (Chunks) auf."""
    words = text.split()
    # Erstelle Chunks basierend auf einer bestimmten Anzahl von Wörtern
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def call_ollama(prompt: str) -> str:
    """
    Ruft das lokale LLM (Ollama) auf, um eine Antwort zu generieren.
    
    Wichtige Hinweise:
    - Stelle sicher, dass Ollama installiert und gestartet wurde (siehe ENV_SETUP.md).
    - Um DeepSeek-Modelle zu verwenden, lade das gewünschte Modell mit:
        ollama pull deepseek-r1:<model_tag>
      Beispiel:
        ollama pull deepseek-r1:7b
    - Führe das Modell anschließend aus, oder nutze die API (ollama run deepseek-r1:<model_tag>).
    - Weitere Informationen findest du auf: https://ollama.com/
    
    Der hier verwendete Befehl ist ein Platzhalter und kann je nach Modell und Konfiguration angepasst werden.
    """
    try:
        # Beispiel: Ruft Ollama über subprocess auf. Der folgende Befehl ist ein Platzhalter.
        result = subprocess.run(
            ["ollama", "run", "deepseek-r1:7b", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Fehler bei der Abfrage des Ollama-Modells: {e}"

def main():
    # Verzeichnis mit den PDF-Dateien (z.B. 3-4 PDFs)
    pdf_folder = "pdfs"
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    
    if not pdf_files:
        print("Keine PDF-Dateien gefunden im Ordner 'pdfs'.")
        return
    
    # Initialisiere das Embedding-Modell (z.B. 'all-MiniLM-L6-v2')
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Liste zur Speicherung der Text-Chunks aus allen PDFs
    corpus_chunks = []
    
    # Verarbeite jede PDF-Datei: Text extrahieren und in Chunks aufteilen
    for pdf_file in pdf_files:
        print(f"Verarbeite Datei: {pdf_file}")
        text = extract_text_from_pdf(pdf_file)
        chunks = split_text_into_chunks(text)
        corpus_chunks.extend(chunks)
    
    if not corpus_chunks:
        print("Keine Text-Chunks aus den PDFs extrahiert.")
        return
    
    # Berechne Embeddings für alle Chunks
    corpus_embeddings_tensor = embedding_model.encode(corpus_chunks, convert_to_tensor=True)
    
    # Benutzerabfrage: Frage, die mit den PDF-Daten beantwortet werden soll
    query = input("Bitte gib deine Frage ein: ")
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    
    # Berechne die Kosinus-Ähnlichkeit zwischen der Abfrage und allen Chunks
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings_tensor)[0]
    
    # Finde den Chunk mit der höchsten Ähnlichkeit
    top_result_idx = int(np.argmax(cosine_scores.cpu().numpy()))
    relevant_chunk = corpus_chunks[top_result_idx]
    
    print("\nRelevanter Kontext aus den PDFs:")
    print(relevant_chunk)
    
    # Erstelle einen Prompt für das LLM, der den gefundenen Kontext und die Benutzerfrage kombiniert
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

if __name__ == "__main__":
    main()
