#!/usr/bin/env python3
"""
Aufgabe 5:
Ausgangslage: Nutzung der OpenAI API für die Integration.
Tasks:
- Nutzt die OpenAI API, um Text zu generieren.
- Führt simple Tests der Endpoints durch.
"""

import os
import openai

def main():
    # API-Schlüssel aus der Umgebungsvariable laden (sicherer Umgang mit sensiblen Daten)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Bitte setze die Umgebungsvariable OPENAI_API_KEY mit deinem OpenAI API-Schlüssel.")
    openai.api_key = openai_api_key

    # Beispielprompt zur Textgenerierung
    prompt = "Schreibe einen kurzen Absatz über die Bedeutung von Künstlicher Intelligenz im Alltag in der Schweiz."
    
    try:
        # Anfrage an die OpenAI API zur Textgenerierung (hier z.B. mit dem Modell text-davinci-003)
        response = openai.Completion.create(
            engine="text-davinci-003",  # Alternativ kann auch ein anderes Modell genutzt werden
            prompt=prompt,
            max_tokens=150,             # Maximale Anzahl an Tokens in der Antwort
            temperature=0.7             # Steuerung der "Kreativität" der Antwort
        )
        # Ausgabe der generierten Antwort
        generated_text = response.choices[0].text.strip()
        print("Generierter Text:")
        print(generated_text)
    except Exception as e:
        print("Fehler bei der Anfrage an die OpenAI API:", e)

if __name__ == "__main__":
    main()
