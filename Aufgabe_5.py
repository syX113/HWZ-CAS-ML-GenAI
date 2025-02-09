#!/usr/bin/env python3
"""
Aufgabe 5:
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

# Erstelle einen asynchronen OpenAI-Client. Der API-Schlüssel wird aus der Umgebungsvariable geladen.
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")  # API-Schlüssel wird hier geladen.
)

async def main() -> None:
    # Erkläre: Diese Funktion sendet eine Chat-Completion-Anfrage an das Modell "gpt-4o".
    # Sende eine Anfrage, die eine Testnachricht an das Modell übergibt.
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
