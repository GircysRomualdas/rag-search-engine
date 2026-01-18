from .constants import DEFAULT_GEMINI_MODEL
from .data_models import HybridRRFResult
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = DEFAULT_GEMINI_MODEL

def prompt_model(prompt: str) -> str | None:
    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected

def get_rrf_doc_list(rrf_results: list[HybridRRFResult]) -> str:
    doc_lines = []
    for r in rrf_results:
        line = f"ID: {r.movie_id}, Title: {r.title}, Description: {r.document}"
        doc_lines.append(line)

    doc_list_str = "\n".join(doc_lines)
    return doc_list_str
