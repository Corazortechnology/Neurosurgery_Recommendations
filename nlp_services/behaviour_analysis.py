from llm_service import call_gemini,call_groqapi

class BehaviourAnalysis:
    def __init__(self, model="gemini-1.5-flash", max_output_tokens=1024, temperature=0.2):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def analyze_gemini(self, text, context_vars=None):
        prompt = f"""Analyze the following text for behavioral patterns: {text}.

                    Only give behaviour label like meltdown,stimming,social,communication,focus,etc and no other information.
                    Also give summary of text including all information of text in 200 words.
                    For example
                    {{"label": "Aggressive","summary":str}}.
                    only give the json output and nothing else."""
        if context_vars:
            prompt = prompt.format(**context_vars)
        
        response = call_gemini(
            prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature
        )
        
        return response.strip().replace("```json","").replace("```","").replace("\n","") if response else None

    def analyze(self, text,model="llama-3.3-70b-versatile"):
        system_prompt = "You are an expert medical neurosurgeon that monitors behaviour of the user patient profile."
        prompt = f"""Analyze the following text for behavioral patterns: {text}.
                    Only give behaviour label like meltdown,stimming,social,communication,focus,etc and no other information.
                    Also give summary of text including all information of text in 200 words.
                    For example
                    {{"label": "Aggressive","summary":str}}.
                    only give the json output and nothing else."""
        
        response = call_groqapi(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model
        )
        
        return response.strip().replace("```json","").replace("```","").replace("\n","") if response else None


