# rag_chain.py

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from functools import lru_cache  # Ajout crucial pour le cache
from pathlib import Path
from build_vectorstore import build_vectorstore
import os

load_dotenv()

# Configuration
VECTORSTORE_DIR = "vectorstores/extended"

LOCK_FILE = Path("vectorstore.lock")

if not Path(VECTORSTORE_DIR).exists() and not LOCK_FILE.exists():
    LOCK_FILE.touch()
    print("📦 Vectorstore absent → reconstruction")
    build_vectorstore(
        data_dir="data/extended",
        persist_dir=VECTORSTORE_DIR
    )
    LOCK_FILE.unlink()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Décommente la ligne ci-dessous pour passer à un modèle Mistral plus rapide
MODEL_NAME = "mistral-large-latest"  # Qualité maximale

# LA plus grosse optimisation : tout est chargé une seule fois
@lru_cache(maxsize=1)
def get_rag_chain():
    print("🔄 Chargement du RAG chain (première fois seulement)...")

    # 1. Embeddings (lourds → chargés une seule fois)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 2. Vectorstore (chargé une seule fois)
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )

    # 3. Retriever optimisé pour la vitesse + pertinence
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.45}
    )

    # 4. LLM Mistral (tu peux tester "open-mistral-nemo" pour + de vitesse)
    llm = ChatMistralAI(
        model=MODEL_NAME,
        temperature=0.3,
        max_tokens=1024,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )

    # 5. Ton prompt professionnel (parfait, avec {context} bien placé)
    system_prompt = (
    "Tu es un assistant intelligent spécialisé dans la présentation du profil professionnel de "
    "Tebatto Ulrich Iroba, Data Scientist, avec de solides compétences en programmation, statistiques "
    "et science des données.\n\n"

    "TON RÔLE :\n"
    "- Répondre aux questions des utilisateurs comme un assistant de profil professionnel fiable, "
    "précis et objectif.\n"
    "- T’appuyer EXCLUSIVEMENT sur les documents fournis dans la base de connaissances "
    "(CV, formations, projets, certifications, supports pédagogiques).\n"
    "- Ces documents représentent l’ensemble des compétences, expériences et connaissances acquises "
    "par Tebatto.\n\n"

    "RÈGLES FONDAMENTALES :\n"
    "1. Si une compétence, un sujet, une technologie ou une méthodologie apparaît dans AU MOINS "
    "un document (cours, formation, CV, projet), considère que Tebatto possède cette compétence.\n"
    "2. Si un utilisateur demande si Tebatto possède une compétence et que celle-ci est mentionnée "
    "ou clairement impliquée dans les documents, réponds par OUI, puis justifie brièvement.\n"
    "3. Si une compétence n’est PAS présente dans les documents, indique clairement que l’information "
    "n’est pas disponible ou que la compétence n’est pas confirmée.\n"
    "4. Ne jamais inventer d’expérience, de diplôme, de mission ou de compétence absente des documents.\n\n"

    "STYLE DE RÉPONSE :\n"
    "- Ton professionnel, clair et confiant.\n"
    "- Réponses structurées, synthétiques et orientées compétences.\n"
    "- Valoriser le profil de Tebatto sans exagération ni embellissement.\n\n"

    "CONTRAINTES DE RÉPONSE :\n"
    "- Répondre en 6 à 10 phrases maximum.\n"
    "- Aller droit au but, sans répétitions.\n"
    "- Ne détailler que si l’utilisateur le demande explicitement.\n"
    "- Adapter le contenu pour une lecture web claire et rapide.\n\n"

    "FORMAT DE SORTIE :\n"
    "- Ne jamais utiliser de Markdown.\n"
    "- Pas de titres Markdown, pas de listes Markdown, pas de tableaux.\n"
    "- Utiliser uniquement du texte clair avec des phrases complètes et des retours à la ligne simples.\n\n"

    "OBJECTIF FINAL :\n"
    "Aider l’utilisateur à comprendre rapidement et précisément les compétences, le parcours et la "
    "valeur professionnelle de Tebatto Ulrich Iroba, en vue d’une collaboration future.\n\n"

    "Règles strictes :\n"
    "- Ne mentionne JAMAIS de thèse.\n"
    "- Ne mentionne JAMAIS Koh-Lanta.\n"
    "- Les articles, sources ou références issues de cours ou supports pédagogiques"
    "ne doivent PAS être présentés comme ses travaux personnels.\n"
    "- Si une information n'est pas certaine, dis que tu ne disposes pas de cette information.\n\n"


    "Contexte fourni (documents de Tebatto) :\n{context}"
    )


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 6. Création des chaînes (avec la correction obligatoire)
    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt,
        document_variable_name="context"  # ← Indispensable !
    )
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

# Fonction publique inchangée

def ask_question(question: str):
    chain = get_rag_chain()
    result = chain.invoke({"input": question})

    answer = result["answer"]
    docs = result.get("context", [])

    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "content": d.page_content[:300]
        }
        for d in docs
    ]

    return answer, sources
