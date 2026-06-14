import os
import json
import google.generativeai as genai

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-pro"
)

def extract_label_fields(text):

    prompt = f"""
    Extract:

    Brand Name
    ABV
    Net Contents
    Government Warning

    Return JSON only.

    Label Text:
    {text}
    """

    response = model.generate_content(prompt)

    return json.loads(response.text)
