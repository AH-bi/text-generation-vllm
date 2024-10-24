from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline, set_seed
import logging


logging.basicConfig(level=logging.INFO)

# Initialization
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# load the model using Hugging Face's pipeline
generator = pipeline('text-generation', model='gpt2')  # Using GPT-2
set_seed(42)  

def preprocess_text(text: str) -> str:
    """Clean the input text"""
    return text.strip().replace("\n", " ")

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    """Handle exceptions and log errors"""
    logging.error(f"Error occurred: {exc}")
    return JSONResponse(content={"detail": str(exc)}, status_code=500)

@app.post("/generate/")
async def generate_text(
    prompt: str = Form(...),
    max_tokens: int = Form(100),
    temperature: float = Form(0.7)
):
  
    cleaned_prompt = preprocess_text(prompt)

    try:
        # Generate text using Hugging Face's pipeline
        generated_output = generator(cleaned_prompt, max_length=max_tokens, temperature=temperature, num_return_sequences=1)
        generated_text = generated_output[0]['generated_text']
    except Exception as e:
        logging.error(f"Error during text generation: {e}")
        return JSONResponse(content={"detail": str(e)}, status_code=500)
        
    return {"prompt": cleaned_prompt, "generated_text": generated_text}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML form for text input"""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)