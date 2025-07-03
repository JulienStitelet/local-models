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
from langchain_core.prompts import PromptTemplate
import re
import json

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

parser_profile = MinifiedPydanticOutputParser(pydantic_object=ProfileResponse)
parser_skills = MinifiedPydanticOutputParser(pydantic_object=SkillsResponse)


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
    title="Gemma 3 API",
    description="API pour générer du texte avec Gemma 3 via llama.cpp",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"message": "Gemma 3 API is running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "platform": platform.system(),
    }


@app.post("/generate", response_model=PromptResponse)
async def generate_text(request: PromptRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        logger.info(f"Génération pour prompt: {request.prompt[:50]}...")

        # Génération du texte
        output = llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            echo=False,  # Ne pas répéter le prompt
        )

        response_text = output["choices"][0]["text"].strip()

        # Informations sur le modèle
        model_info = {
            "tokens_generated": len(output["choices"][0]["text"].split()),
            "finish_reason": output["choices"][0].get("finish_reason", "length"),
            "platform": platform.system(),
        }

        logger.info("Génération terminée avec succès")

        return PromptResponse(response=response_text, model_info=model_info)

    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_completion(request: PromptRequest):
    """Route alternative avec format plus simple"""
    if llm is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    chat_prompt = PromptTemplate(
        template="{system_prompt}\n\n{format_instructions}\n\nHere is the resume:\n{query}\n.## IMPORTANT:{important}",
        input_variables=["human_message"],
        partial_variables={
            "system_prompt": "Extract information from the given raw text resume. Wrap the output in `json` tags",
            "format_instructions": parser_profile.get_format_instructions(),
            "important": """Return a JSON object containing only fields that have meaningful values.

- Do NOT include fields with null, empty strings, empty arrays, or missing data.
- Omit any field for which no reliable value is found.
- For example, if the value for "d" is not available, do not include "d" at all in the JSON output.

The final JSON should be as compact as possible with only non-empty, non-null fields.
Field names must be lower case""",
        },
    )

    prompt = chat_prompt.invoke({"query": request.prompt}).to_string()
    print(prompt)
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
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.DOTALL)
        # if not match:
        #     match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)

        if match:
            json_str = match.group(1)
            # Optional: remove extra whitespace or newlines
            json_str_clean = json_str.strip()
            data = json.loads(json_str_clean)
            print(data)
            data = parser_profile._remove_none_values(data)
            minified = parser_profile.minified(**data)
            r = parser_profile.get_original(minified)
            print(r)
            return r
        else:
            print(f"No JSON block found in {text}")
            return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
