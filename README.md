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
