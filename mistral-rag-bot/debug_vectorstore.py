from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Nom du modèle d'embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Créer l'objet embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Charger le vectorstore persistant
db = Chroma(
    persist_directory="vectorstores/extended",
    embedding_function=embeddings
)

# Test de recherche par similarité
query = "Explique-moi le fonctionnement du RAG"
top_k = 5  # nombre de résultats à retourner

results = db.similarity_search(query, k=top_k)

# Afficher les résultats
print(f"Résultats pour la requête : '{query}'\n")
for i, doc in enumerate(results, 1):
    print(f"--- Résultat {i} ---")
    print(doc.page_content)
    print()
