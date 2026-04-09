from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv(".env.local")

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class TextRequest(BaseModel):
    text: str

def simplify_text(text):
    prompt = f"Simplify this text for people with dyslexia. Use simple words and short sentences:\n\n{text}"
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return response.choices[0].message.content

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/simplify")
def simplify(req: TextRequest):
    try:
        result = simplify_text(req.text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))