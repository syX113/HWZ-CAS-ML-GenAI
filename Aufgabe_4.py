#!/usr/bin/env python3
"""
Aufgabe 4:
Ausgangslage: Waehlt ein Modell auf Hugging Face (z.B. ein kleineres Modell wie 7b) zur Sentimentanalyse.
Das Modell soll genutzt werden, um Texte zu klassifizieren.

Tasks:
- Fuehrt eine Sentimentanalyse von Text durch.
- Diskutiert anschliessend, welche Metriken (z.B. Accuracy, Precision, Recall, F1-Score) sinnvoll waeren und warum.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def main():
    # Modellname auswählen (hier wird ein Beispielmodell für Sentimentanalyse verwendet)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Modell ohne GPU-spezifische Optionen laden (vollständig auf der CPU)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Erstelle eine Pipeline für die Sentimentanalyse
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    # Beispieltext, der analysiert werden soll
    text = "I absolutely love the new design of your product! It is amazing and well thought out."
    
    # Führe die Sentimentanalyse durch
    result = sentiment_pipeline(text)
    print("Sentimentanalyse Ergebnis:")
    print(result)
    
    # --- Evaluation ---
    # Als sinnvolle Metrik für die Evaluation einer Sentimentanalyse eignet sich z.B. die Accuracy,
    # da es sich meist um ein binäres Klassifikationsproblem (positiv/negativ) handelt.
    # Bei unbalancierten Datensätzen können zusätzlich Precision, Recall und F1-Score wichtige Einblicke liefern.
    #
    # Beispielcode (sofern ein Testdatensatz mit wahren Labels vorliegt):
    #
    # from sklearn.metrics import accuracy_score
    #
    # # Angenommene wahre Labels: 1 für positiv, 0 für negativ
    # true_labels = [1, 0, 1, 1, 0]  # Beispielhafte wahre Labels
    #
    # # Erhalte Vorhersagen (Annahme: 'POSITIVE' entspricht 1, 'NEGATIVE' entspricht 0)
    # predicted_labels = [1 if res['label'] == 'POSITIVE' else 0 for res in sentiment_pipeline(test_texts)]
    #
    # accuracy = accuracy_score(true_labels, predicted_labels)
    # print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
