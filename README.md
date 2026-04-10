## Setup
git clone https://github.com/northania/ai-simplifier
cd ai-simplifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
## Environment
Buat file `.env.local` dan isi:
GROQ_API_KEY=your_api_key
## Run
uvicorn back:app --reload
## Open App
Buka `http://127.0.0.1:8000/` untuk memakai frontend dan backend dalam satu server FastAPI.
## Features
- Simplify text via endpoint `POST /simplify`
- Text-to-speech via browser `SpeechSynthesis`
