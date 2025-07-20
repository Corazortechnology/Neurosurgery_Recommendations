from transformers import pipeline

class SentimeAnalysis:
    def __init__(self, model_name="hazarri/fine-tuned-roberta-sentiment"):
        self.pipe = pipeline("text-classification", model=model_name)

    def analyze(self, text):
        results = self.pipe(text)
        return results[0]['label'] if results else None
