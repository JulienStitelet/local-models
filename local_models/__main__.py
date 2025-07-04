import logging
import os
import platform
from contextlib import asynccontextmanager

import torch
import uvicorn
from llama_cpp import Llama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_pydantic_minifier.minifier_pydantic import MinifiedPydanticOutputParser

import re
import json

from local_models.prompts import get_prompt
from local_models.schema.resume import ProfileResponse, SkillsResponse

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Modèles Pydantic
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[str] = ["<|end_of_text|>"]


class PromptResponse(BaseModel):
    response: str
    model_info: dict


# Variable globale pour le modèle
llm = None

# parser = MinifiedPydanticOutputParser(pydantic_object=ProfileResponse)
parser = MinifiedPydanticOutputParser(pydantic_object=SkillsResponse)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global llm
    logger.info("Chargement du modèle...")

    # Configuration adaptée à l'environnement
    model_path = os.getenv(
        "MODEL_PATH",
        "/Users/jstitelet/.lmstudio/models/lmstudio-community/gemma-3n-E4B-it-text-GGUF/gemma-3n-E4B-it-Q4_K_M.gguf",
    )

    # Détection automatique Metal vs CPU
    if platform.system() == "Darwin":  # macOS
        n_gpu_layers = -1  # Utilise Metal
    elif torch.cuda.is_available():  # Linux avec NVIDIA
        n_gpu_layers = -1  # Utilise CUDA
    else:  # Linux sans GPU
        n_gpu_layers = 0  # CPU seulement

    try:
        llm = Llama.from_pretrained(
            repo_id="lmstudio-community/gemma-3n-E4B-it-text-GGUF",
            filename="gemma-3n-E4B-it-Q4_K_M.gguf",
            # cache_dir="",
            n_ctx=4096,
            n_threads=os.cpu_count(),
            n_gpu_layers=n_gpu_layers,
        )

        logger.info("Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise

    yield

    # Shutdown
    logger.info("Arrêt de l'application")


# Initialisation FastAPI
app = FastAPI(
    title="Gemma 3n API",
    description="API pour générer du texte avec Gemma 3 via llama.cpp",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"message": "Gemma 3n API is running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "platform": platform.system(),
    }


@app.post("/chat")
async def chat_completion(request: PromptRequest):
    """Route alternative avec format plus simple"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    prompt = get_prompt(parser, request.prompt)
    try:
        output = llm(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            echo=False,
        )

        # text to dict to obj
        text = output["choices"][0]["text"]
        print(text)
        # match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.DOTALL)
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        # if not match:
        #     match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)

        if match:
            json_str = match.group(1)
            # Optional: remove extra whitespace or newlines
            json_str_clean = json_str.strip()
            data = json.loads(json_str_clean)
            print(data)
            data = parser._remove_none_values(data)
            minified = parser.minified(**data)
            r = parser.get_original(minified)
            print(r)
            return r
        else:
            print(f"No JSON block found in {text}")
            return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
