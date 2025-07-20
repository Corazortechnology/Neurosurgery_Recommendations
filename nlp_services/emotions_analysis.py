from transformers import pipeline

class EmotionsAnalysis:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.pipe = pipeline("text-classification", model=model_name)

    def analyze(self, text):
        results = self.pipe(text)
        return results[0]['label'] if results else None

