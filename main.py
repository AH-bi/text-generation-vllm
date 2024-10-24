from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from vllm import LLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialization 
app = FastAPI()
templates = Jinja2Templates(directory="templates")  

# Load the model
model = LLM(model="gpt2", device="cpu")# using GPT-2 model 
  

def preprocess_text(text: str) -> str:
    """Clean the input text."""
    return text.strip().replace("\n", " ")

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    """Handle exceptions and log errors."""
    logging.error(f"Error occurred: {exc}")
    return JSONResponse(content={"detail": str(exc)}, status_code=500)

@app.post("/generate/")
async def generate_text(prompt: str, temperature: float = Query(0.7), max_tokens: int = Query(100)):
    """Generate text based on the user prompt."""
    cleaned_prompt = preprocess_text(prompt)
    # Check if `generate` is async or sync and call accordingly
    try:
        generated_text = model.generate(cleaned_prompt, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        logging.error(f"Error during text generation: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)
        
    return {"prompt": cleaned_prompt, "generated_text": generated_text}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML form for text input."""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
