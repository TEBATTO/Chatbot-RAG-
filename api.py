# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_chain import ask_question


# App FastAPI
app = FastAPI(
    title="Mistral RAG API",
    description="API RAG basée sur Mistral pour le portfolio de Tebatto Ulrich Iroba",
    version="1.0.0",
)


# CORS (OBLIGATOIRE pour Django)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en prod → domaine Django uniquement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 
# Modèle d'entrée
class Query(BaseModel):
    question: str

# Health check (debug)
@app.get("/")
def health():
    return {"status": "API RAG opérationnelle"}

# Endpoint chatbot
@app.post("/chat")
def chat(query: Query):
    """
    Reçoit une question utilisateur
    Retourne la réponse RAG + sources
    """
    try:
        answer, sources = ask_question(query.question)

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur RAG : {str(e)}"
        )