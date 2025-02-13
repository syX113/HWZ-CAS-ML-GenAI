#!/usr/bin/env python3
"""
Aufgabe 4:
Ausgangslage: Nutzung der OpenAI API für die Integration.
Tasks:
- Nutzt die OpenAI API, um Text zu generieren.
- Führt simple Tests der Endpoints durch.

API-Schlüssel in Umgebungsvariable laden:
Stelle sicher, dass du den API-Schlüssel als Umgebungsvariable via Terminal gesetzt hast:
    export OPENAI_API_KEY="api_schluessel"
"""
import os
import asyncio
from openai import AsyncOpenAI

def load_api_key() -> str:
    # Prüfen Umgebungsvariablen
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    # Falls API Key nicht vorhanden, prüfe File
    key_file = "api_key.txt"
    try:
        with open(key_file, "r") as f:
            api_key = f.read().strip()
        if api_key:
            return api_key
    except Exception:
        pass
    raise Exception("API key not found. Set OPENAI_API_KEY or provide the key in api_key.txt.")

# Erstelle einen asynchronen OpenAI-Client using the loaded API key.
client = AsyncOpenAI(api_key=load_api_key())

async def main() -> None:
    # Erkläre: Diese Funktion sendet eine Chat-Completion-Anfrage an das Modell "gpt-4o".
    # Sende eine Anfrage, die eine Testnachricht an das Modell übergibt.
    # Beispiel für Textklassifikation: https://platform.openai.com/docs/examples/default-tweet-classifier

    
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Erkläre das 3-Körper Problem in einfachen Worten und klar verständlich",  # Testnachricht
            }
        ],
        model="gpt-3.5-turbo",  # Modellname
    )

    print(f'Antwort von OpenAI API: {chat_completion.choices[0].message.content}')

# Führt die asynchrone main()-Funktion aus.
asyncio.run(main())
