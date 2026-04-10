from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv
from typing import Optional
import os

load_dotenv(".env.local")

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Please add it to your .env.local file."
    )

client = Groq(api_key=GROQ_API_KEY)

class TextRequest(BaseModel):
    text: str = Field(..., min_length   =1, max_length=3000)

class SimplifyResponse(BaseModel):
    original_text: str
    simplified_text: str
    message: str


def build_prompt(text: str) -> str:
    return f"""
You are an accessibility assistant.

Rewrite the following text so it is easier for people with dyslexia to read.

Rules:
- Use short sentences.
- Use simple and common words.
- Keep the original meaning.
- Break long ideas into smaller parts.
- Avoid jargon where possible.
- Return only the simplified text.
- Do not add explanations, notes, or bullet points unless needed for clarity.

Text:
{text}
""".strip()


def simplify_text(text: str) -> str:
    clean_text = text.strip()

    if not clean_text:
        raise ValueError("Text cannot be empty.")

    prompt = build_prompt(clean_text)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You simplify text for accessibility and readability.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )

    result: Optional[str] = response.choices[0].message.content

    if not result or not result.strip():
        raise RuntimeError("Model returned an empty response.")

    return result.strip()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ai-simplifier-backend",
    }


@app.post("/simplify", response_model=SimplifyResponse)
def simplify(req: TextRequest):
    try:
        simplified = simplify_text(req.text)
        return SimplifyResponse(
            original_text=req.text,
            simplified_text=simplified,
            message="Text simplified successfully.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to simplify text.")