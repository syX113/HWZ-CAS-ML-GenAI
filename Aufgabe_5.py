#!/usr/bin/env python3
"""
Aufgabe 5:
Ausgangslage: Nutzung der OpenAI API für die Integration.
Tasks:
- Nutzt die OpenAI API, um Text zu generieren.
- Führt simple Tests der Endpoints durch.

API-Schlüssel in Umgebungsvariable laden:
Stelle sicher, dass du den API-Schlüssel als Umgebungsvariable gesetzt hast:
    export OPENAI_API_KEY="dein_api_schluessel"
"""
import os
import asyncio
from openai import AsyncOpenAI

# Erstelle einen asynchronen OpenAI-Client. Der API-Schlüssel wird aus der Umgebungsvariable geladen.
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")  # Standard: API-Schlüssel wird hier automatisch geladen.
)

async def main() -> None:
    # Erkläre: Diese Funktion sendet eine Chat-Completion-Anfrage an das Modell "gpt-4o".
    # Sende eine Anfrage, die eine Testnachricht an das Modell übergibt.
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",  # Testnachricht
            }
        ],
        model="gpt-3.5-turbo",  # Modellname
    )
    # Optional: Ausgabe zur Überprüfung der Antwort
    print(chat_completion)

# Führt die asynchrone main()-Funktion aus.
asyncio.run(main())
