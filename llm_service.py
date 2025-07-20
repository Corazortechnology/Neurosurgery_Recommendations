import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
import json
from groq import Groq
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
def call_gemini(prompt, context_vars=None, max_output_tokens=1024, temperature=0.2):
    
    if context_vars:
        prompt = prompt.format(**context_vars)
    response = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_output_tokens,
            "temperature": temperature
        }
    )
    return response.text.strip() if hasattr(response, "text") else str(response)

def call_openai(
    prompt,
    context_vars=None,
    system_prompt=None,
    model="gpt-4o-mini",
    max_tokens=1024,
    temperature=0.2,
):
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = []
    if context_vars:
        prompt = prompt.format(**context_vars)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    content = response.choices[0].message.content.strip() if response.choices else ""
    return content



def call_groqapi(prompt,system_prompt,context_vars=None,model=None):

    client = Groq(api_key = GROQ_API_KEY)
    if context_vars:
        prompt = prompt.format(**context_vars)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "system",
            "content": system_prompt
            },
            {
            "role": "user",
            "content": prompt
            }
        ],
        temperature=1,
        top_p=1,
        stream=True,
        stop=None,
    )
    full_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
    return full_response