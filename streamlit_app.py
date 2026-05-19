import streamlit as st
import requests
import uuid
import os
import json
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Kokoro Medical Companion",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .app-header {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.1), rgba(66, 153, 225, 0.1));
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .app-logo {
        font-size: 3rem;
        margin-right: 1rem;
    }
    .app-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(to right, #48bb78, #4299e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #718096;
        margin-top: 0.2rem;
    }
    
    /* Status indicators */
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-active {
        background-color: rgba(72, 187, 120, 0.2);
        color: #48bb78;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
    .status-offline {
        background-color: rgba(229, 62, 62, 0.2);
        color: #e53e3e;
        border: 1px solid rgba(229, 62, 62, 0.3);
    }
    
    /* Info cards */
    .info-card {
        background-color: rgba(26, 32, 44, 0.4);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4299e1;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex[:6]}"
if "tts_queue" not in st.session_state:
    st.session_state.tts_queue = None

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.markdown("<h2 style='text-align: center; color: #48bb78;'>⚙️ Settings</h2>", unsafe_allow_html=True)
st.sidebar.divider()

# Connection Mode Selector
api_host = st.sidebar.text_input("Backend API URL", value="http://localhost:8000")
use_direct_pipeline = st.sidebar.checkbox("Direct Pipeline Mode (Bypass FastAPI)", value=False, 
                                          help="Run the LangGraph workflow directly inside Streamlit without invoking FastAPI. Useful for standalone local testing.")

# Check Backend API Health
api_active = False
if not use_direct_pipeline:
    try:
        # Simple health check, or just trying to reach the address
        response = requests.get(api_host, timeout=1.0)
        api_active = True
    except:
        api_active = False

# Display connection status
if use_direct_pipeline:
    st.sidebar.markdown('<div class="status-badge status-active">🟢 Running Standalone (Direct)</div>', unsafe_allow_html=True)
elif api_active:
    st.sidebar.markdown(f'<div class="status-badge status-active">🟢 Connected to API: {api_host}</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<div class="status-badge status-offline">🔴 FastAPI Offline (Defaulting to Direct Mode)</div>', unsafe_allow_html=True)
    # Automatically switch to direct pipeline if API is offline
    use_direct_pipeline = True

st.sidebar.divider()

# User Settings
st.sidebar.markdown("### 👤 User Settings")
role = st.sidebar.selectbox("Your Role", ["Patient", "Doctor"], index=0)
language = st.sidebar.selectbox("Language Preference", ["English", "Hinglish (Hindi in English Script)"], index=0)
lang_code = "en" if language == "English" else "hi"
user_id = st.sidebar.text_input("Session User ID (for Chat History)", value=st.session_state.user_id)
st.session_state.user_id = user_id

# Display loaded DB info
st.sidebar.markdown("### 🗄️ Connected Databases")
st.sidebar.info("""
- **Insurance DB**: Loaded (Local)
- **Gyno DB**: Loaded (Fallback/EFS)
- **Heart DB**: Loaded (Fallback/EFS)
""")

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.tts_queue = None
    st.rerun()

# ----------------- MAIN APP INTERFACE -----------------

# Header
st.markdown("""
<div class="app-header">
    <div class="app-logo">🩺</div>
    <div>
        <h1 class="app-title">Kokoro Medical Companion</h1>
        <div class="app-subtitle">Your intelligent medical companion. Supporting patients and professionals.</div>
        <div style="font-size: 0.85rem; color: #48bb78; margin-top: 0.3rem; font-weight: 600; display: flex; align-items: center; gap: 5px;">
            <span>✨</span> Finetuned Medical Chatbot Active
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Display guidelines/info card
with st.expander("ℹ️ How to use Kokoro"):
    st.markdown("""
    - **Role Selector**: Swap roles in the sidebar! The **Patient** role gets conversational, caring Hinglish/English advice. The **Doctor** role gets formatted clinical notes, insurance contexts, and professional outlines.
    - **Language Selector**: Choose Hinglish to get instructions and medical suggestions written in a mixture of Hindi and English (using English letters), making it very easy to understand.
    - **TTS Feature**: The bot will clean its response of all Markdown and emojis, preparing a clean text-to-speech output that plays in your browser.
    """)

# Display Chat Messages
for index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display play audio button if speech text is available
        if msg["role"] == "assistant" and "speech" in msg and msg["speech"]:
            # Custom Web Speech Synthesis button
            button_id = f"play_btn_{index}"
            escaped_speech = msg["speech"].replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
            
            # Simple Web Speech API integration
            play_js = f"""
            <button onclick="
                const synth = window.speechSynthesis;
                if (synth.speaking) {{
                    synth.cancel();
                }} else {{
                    const utter = new SpeechSynthesisUtterance('{escaped_speech}');
                    // Set language voice based on selection
                    utter.lang = '{'hi-IN' if lang_code == 'hi' else 'en-US'}';
                    synth.speak(utter);
                }}
            " style="
                background-color: transparent;
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: #a0aec0;
                padding: 4px 10px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.8rem;
                display: flex;
                align-items: center;
                gap: 5px;
                margin-top: 8px;
            ">
                🔊 Speak Response
            </button>
            """
            st.components.v1.html(play_js, height=45)

# Text-to-speech auto-trigger for the latest assistant message
if st.session_state.tts_queue:
    escaped_speech = st.session_state.tts_queue.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    auto_play_js = f"""
    <script>
        const synth = window.speechSynthesis;
        synth.cancel();
        setTimeout(() => {{
            const utter = new SpeechSynthesisUtterance('{escaped_speech}');
            utter.lang = '{'hi-IN' if lang_code == 'hi' else 'en-US'}';
            synth.speak(utter);
        }}, 500);
    </script>
    """
    st.components.v1.html(auto_play_js, height=0, width=0)
    # Clear the queue so it only plays once
    st.session_state.tts_queue = None

# Chat Input
if prompt := st.chat_input("Ask Kokoro... (e.g. What is the limit for ambulance expenses?)"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("*Kokoro is thinking...*")
        
        try:
            # Check connection mode
            if not use_direct_pipeline:
                # Call FastAPI backend
                payload = {
                    "message": prompt,
                    "user_id": user_id,
                    "language": lang_code,
                    "role": role.lower()
                }
                res = requests.post(f"{api_host}/rag", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("response", "No response received.")
                    speech = data.get("speech", "")
                else:
                    answer = f"Error from FastAPI backend: {res.status_code} - {res.text}"
                    speech = ""
            else:
                # Call LangGraph workflow directly
                from langgraph_workflow import run_rag_pipeline
                from utils import clean_text_for_speech
                
                answer = run_rag_pipeline(
                    message=prompt,
                    role=role.lower(),
                    language=lang_code,
                    user_id=user_id
                )
                speech = clean_text_for_speech(answer)
                
            response_placeholder.markdown(answer)
            
            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "speech": speech
            })
            
            # Queue the speech to auto-play
            if speech:
                st.session_state.tts_queue = speech
                st.rerun()
                
        except Exception as e:
            err_msg = f"Failed to get response. Details: {str(e)}"
            response_placeholder.error(err_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": err_msg,
                "speech": ""
            })
