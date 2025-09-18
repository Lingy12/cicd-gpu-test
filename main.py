import logging
import os
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen Thinking Model API",
    description="API for Qwen3-4B-Thinking model inference",
    version="1.0.0",
)

# Global variables for model and tokenizer
model = None
tokenizer = None


class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class ChatResponse(BaseModel):
    thinking_content: str
    content: str
    prompt: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


def load_model():
    """Load the Qwen model and tokenizer"""
    global model, tokenizer

    if model is not None and tokenizer is not None:
        return

    model_name = "Qwen/Qwen3-4B-Thinking-2507-FP8"

    try:
        logger.info(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Loading model for {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="healthy", model_loaded=model is not None and tokenizer is not None, device=device
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate response using Qwen thinking model"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare the model input
        messages = [{"role": "user", "content": request.prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # Parse thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip(
            "\n"
        )
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return ChatResponse(
            thinking_content=thinking_content, content=content, prompt=request.prompt
        )

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Qwen Thinking Model API", "docs": "/docs"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
