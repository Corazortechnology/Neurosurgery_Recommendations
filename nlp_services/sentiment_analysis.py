from transformers import pipeline
from llm_service import call_gemini,call_groqapi

class SentimeAnalysis:
    def __init__(self, model_name="hazarri/fine-tuned-roberta-sentiment"):
        self.pipe = pipeline("text-classification", model=model_name)

    def analyze(self, text):
        results = self.pipe(text)
        return results[0]['label'] if results else None

    def sentiment_analyze(self, text):
        system_prompt = "You are an expert at analyzing sentiment in user text."
        prompt = f"""Analyze the following text for sentiment.
                    Only give sentiment label like positive, negative, neutral, etc and no other information.
                    For example
                    {{"sentiment": "Positive"}}.
                    Given text: {text}
                    only give the json output and nothing else."""
        response = call_groqapi(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.model
        )
        return response.strip().replace("```json","").replace("```","").replace("\n","") if response else None
