from transformers import pipeline
from llm_service import call_gemini,call_groqapi

class EmotionsAnalysis:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.pipe = pipeline("text-classification", model=model_name)

    def analyze(self, text):
        results = self.pipe(text)
        return results[0]['label'] if results else None

    def emotion_analysis(self, text):
        system_prompt = "You are an expert in analyzing emotional states from user text."
        prompt = f"""Analyze the following text for emotional states.
                    Only give emotion label like happy, sad, angry, anxious, calm, etc and no other information.
                    For example
                    {{"emotion": "Happy"}}.
                    Given text: {text}
                    only give the json output and nothing else."""
        
        response = call_groqapi(
            prompt=prompt,
            system_prompt=system_prompt,
            model="llama-3.3-70b-versatile"
        )
        
        return response.strip().replace("```json","").replace("```","").replace("\n","") if response else None

