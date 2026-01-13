import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Recommandée
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 5000

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def build_vectorstore(data_dir: str, persist_dir: str):
    print(f"🚀 Début création : {persist_dir} à partir de {data_dir}")

    # Chargement des PDFs
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    print(f"📄 Documents (pages) chargés : {len(documents)}")

    # Découpage
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"🧩 Chunks générés : {len(chunks)}")

    # Suppression de l'ancienne base si elle existe (pour repartir proprement)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"🗑️ Ancienne base supprimée : {persist_dir}")

    # Création de la nouvelle base
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"✅ Vectorstore créé avec succès dans : {persist_dir}\n")

if __name__ == "__main__":
    # Crée les deux bases
    build_vectorstore("data/core", "vectorstores/core")
    build_vectorstore("data/extended", "vectorstores/extended")