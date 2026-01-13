import streamlit as st
import json
import os
from datetime import datetime
from rag_chain import ask_question

# Configuration de la page
st.set_page_config(page_title="Tebatto Iroba — Assistant Profil", layout="wide")
CONV_DIR = "conversations"
os.makedirs(CONV_DIR, exist_ok=True)

# ================== FONCTIONS UTILITAIRES ==================
def get_chat_title(messages):
    """Retourne un titre basé sur les 3 premiers mots de la première question utilisateur"""
    for msg in messages:
        if msg["role"] == "user" and msg["content"].strip():
            words = msg["content"].strip().split()
            title = " ".join(words[:3])
            return title if len(title) <= 30 else title[:27] + "..."
    return "Nouvelle conversation"

def load_chat(chat_id):
    path = os.path.join(CONV_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("messages", [])
    return []

def save_chat(chat_id, messages):
    path = os.path.join(CONV_DIR, f"{chat_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"id": chat_id, "messages": messages}, f, ensure_ascii=False, indent=2)

def delete_chat(chat_id):
    path = os.path.join(CONV_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)

# ================== INITIALISATION ==================
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>Tebatto Iroba</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#64748b;'>Assistant Profil & Compétences</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("➕ Nouvelle conversation", use_container_width=True, type="secondary"):
        chat_id = datetime.now().strftime("chat_%Y%m%d_%H%M%S")
        st.session_state.current_chat = chat_id
        st.session_state.messages = []
        st.rerun()

    st.markdown("### 💬 Conversations récentes")

    # Liste des conversations triées par date (récentes en haut)
    chat_files = sorted([f for f in os.listdir(CONV_DIR) if f.endswith(".json")], reverse=True)
    
    for filename in chat_files:
        chat_id = filename.replace(".json", "")
        messages = load_chat(chat_id)
        title = get_chat_title(messages)
        
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(title, key=f"load_{chat_id}", use_container_width=True):
                st.session_state.current_chat = chat_id
                st.session_state.messages = messages
                st.rerun()
        with col2:
            if st.button("×", key=f"del_{chat_id}", help="Supprimer cette conversation", type="secondary"):
                delete_chat(chat_id)
                if st.session_state.current_chat == chat_id:
                    st.session_state.current_chat = None
                    st.session_state.messages = []
                st.rerun()

# ================== AFFICHAGE DU CHAT ==================
def bubble(text, role):
    bg = "#e3f2fd" if role == "user" else "#f5f5f5"
    align = "flex-end" if role == "user" else "flex-start"
    icon = "👤" if role == "user" else "🤖"
    st.markdown(
        f"""
        <div style='display:flex; justify-content:{align}; margin:15px 0;'>
            <div style='background:{bg}; padding:14px 18px; border-radius:18px; max-width:80%; 
                        box-shadow:0 2px 4px rgba(0,0,0,0.08); line-height:1.5;'>
                <b>{icon}</b> {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Titre de la conversation en cours
if st.session_state.current_chat and st.session_state.messages:
    current_title = get_chat_title(st.session_state.messages)
    st.markdown(f"<h3 style='text-align:center; color:#000000; margin-bottom:30px;'>{current_title}</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='text-align:center; color:#000000; margin-bottom:30px;'>Démarre une nouvelle conversation</h3>", unsafe_allow_html=True)

# Affichage des messages
for msg in st.session_state.messages:
    bubble(msg["content"], msg["role"])

# ================== SAISIE ==================
question = st.chat_input("Poser une question sur le profil, les compétences ou les formations de Tebatto...")

if question:
    # Création automatique d'un nouveau chat si aucun actif
    if not st.session_state.current_chat:
        chat_id = datetime.now().strftime("chat_%Y%m%d_%H%M%S")
        st.session_state.current_chat = chat_id
        st.session_state.messages = []

    # Ajout question utilisateur
    st.session_state.messages.append({"role": "user", "content": question})
    bubble(question, "user")

    # Génération réponse
    with st.spinner("Analyse des documents en cours…"):
        answer, sources = ask_question(question)

    # Affichage réponse
    st.session_state.messages.append({"role": "assistant", "content": answer})
    bubble(answer, "assistant")

    # Sources (discrètes mais accessibles)
    if sources:
        with st.expander("📄 Sources utilisées dans les documents"):
            for i, doc in enumerate(sources[:4]):
                st.caption(f"Source {i+1} — {doc['source']}")
                st.markdown(
                    f"<small>{doc['content']}</small>",
                    unsafe_allow_html=True
                )


    # Sauvegarde
    save_chat(st.session_state.current_chat, st.session_state.messages)
    st.rerun()

# ================== FOOTER ==================
st.markdown(
    "<div style='text-align:center; margin-top:50px; color:#94a3b8; font-size:0.9em; padding:20px;'>"
    "Réponses générées exclusivement à partir des documents personnels de Tebatto Ulrich Iroba (CV, cours)."
    "</div>",
    unsafe_allow_html=True
)