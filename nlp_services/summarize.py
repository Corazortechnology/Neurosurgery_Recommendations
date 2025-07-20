from llm_service import call_gemini,call_groqapi,call_openai

class Summarizer:
    def __init__(self, model="gemini-1.5-flash", max_output_tokens=1024, temperature=0.2):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def analyze_gemini(self, text, context_vars=None):
        prompt = f"""
                    You are a medical summarization expert AI. Your task is to summarize patient profiles for inclusion in medical records.

                Please summarize the following patient profile while:

                Maintaining accuracy of medical terms and diagnoses.

                Keeping clear structure (e.g. history, current complaints, treatment, recommendations) if present in the input.

                Using concise clinical language suitable for a doctor’s note without losing essential meaning.

                Here is the patient profile to summarize:
                {text}"""
        if context_vars:
            prompt = prompt.format(**context_vars)
        
        response = call_gemini(
            prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature
        )
        
        return response.strip().replace("```json","").replace("```","").replace("\n","") if response else None

    def analyze(self, text,model="llama-3.3-70b-versatile"):
        system_prompt = "You are a medical summarization expert AI. Your task is to summarize patient profiles for inclusion in medical records in 200 words."
        prompt = f"""
                Please summarize the following patient profile while:

                Maintaining accuracy of medical terms and diagnoses.

                Keeping clear structure (e.g. history, current complaints, treatment, recommendations) if present in the input.

                Using concise clinical language suitable for a doctor’s note without losing essential meaning.

                Here is the patient profile to summarize:
                {text}"""
        
        # response = call_groqapi(
        #     prompt,
        #     system_prompt,
        #     model
        # )
        response = call_openai(prompt=prompt,system_prompt=system_prompt)
        
        return response.strip().replace("```json","").replace("```","").replace("\n","") if response else None


