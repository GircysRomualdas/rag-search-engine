from lib.utils.prompt import prompt_model_parts
import mimetypes
from google.genai import types

def describe_image_command(image_path: str, query: str) -> None:
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = analyzing_iamage_prompt(mime, image_bytes, query)
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

def analyzing_iamage_prompt(mime: str, image_bytes: bytes, query: str):
    prompt = f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""

    parts = [
        prompt,
        types.Part.from_bytes(data=image_bytes, mime_type=mime),
        query.strip(),
    ]

    return prompt_model_parts(parts)
