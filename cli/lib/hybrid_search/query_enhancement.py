import os
from dotenv import load_dotenv
from google import genai
from lib.utils.constants import DEFAULT_GEMINI_MODEL

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = DEFAULT_GEMINI_MODEL

def spell_correct(query: str) ->str:
    prompt = f"""Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.
    Corrected:"""

    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query

def enhance_query(query: str, method: str = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case _:
            return query
