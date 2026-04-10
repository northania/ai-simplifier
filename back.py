from pathlib import Path
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from groq import APIConnectionError, Groq
import httpx
from pydantic import BaseModel, Field

load_dotenv(".env.local")

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
STATIC_DIR = PUBLIC_DIR / "static"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

http_client = httpx.Client(trust_env=False, timeout=30.0)
client = Groq(api_key=GROQ_API_KEY, http_client=http_client) if GROQ_API_KEY else None

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
- Use original English technical terms as-is.
- Use the same language as the input text.

Text:
{text}
""".strip()


def simplify_text(text: str) -> str:
    clean_text = text.strip()

    if not clean_text:
        raise ValueError("Text cannot be empty.")

    if client is None:
        raise RuntimeError("GROQ_API_KEY is not set.")

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


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(PUBLIC_DIR / "index.html")


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
    except APIConnectionError:
        raise HTTPException(
            status_code=502,
            detail="Cannot reach Groq API. Check your internet connection or proxy settings.",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to simplify text.")
