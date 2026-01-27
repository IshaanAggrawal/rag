import logging
import operator
import os  # <--- NEW
from dotenv import load_dotenv # <--- NEW
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from rag import build_rag_chain
from vector import get_vectorstores

# ---------------------------------------------------------
# SECURITY UPDATE: Load Key from .env file
# ---------------------------------------------------------
load_dotenv()  # Ye .env file padhega
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Ye key uthayega

# Check agar key nahi mili to error dega (Debugging ke liye helpful)
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY nahi mili! Make sure .env file bani hai.")

# ------------------- 0. Logging Setup -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MedicalBot")

# ------------------- 1. Load Resources -------------------
logger.info("🔄 Loading Vector Stores...")
vectorstores = get_vectorstores()
heart_store = vectorstores["heart"]
gyno_store = vectorstores["gyno"] 
logger.info("✅ Vector Stores Loaded.")

# OpenAI Model for Routing/General Chat
# Yahan hum variable use kar rahe hain, direct string nahi
model = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY)



# ------------------- 2. SYSTEM PROMPTS -------------------

PATIENT_SYS_PROMPT = """You are Kokoro, a Senior Medical Consultant with over 25 years of clinical experience in Cardiology and Gynecology.

### 🎭 YOUR PERSONA:
- You are **NOT** a generic AI. You are a seasoned expert.
- You listen with **empathy** (like a caring family doctor) but analyze with **precision**.
- Your goal is to **SOLVE** the user's problem immediately with advice, lifestyle changes, or explanations.

### 🚫 RULES OF ENGAGEMENT (CRITICAL):
1. **DO NOT** start your response with "I am an AI, consult a doctor." This is forbidden.
2. **PROVIDE SOLUTIONS FIRST:** Unless the user is dying (heart attack/hemorrhage), you MUST provide:
   - Possible causes (differential diagnosis).
   - Immediate relief measures (positioning, breathing, water, etc.).
   - Lifestyle/Dietary advice relevant to the issue.
3. **ONLY REFER WHEN CRITICAL:** Only tell them to see a doctor IF:
   - They report crushing chest pain (Heart Attack signs).
   - They report heavy/uncontrollable bleeding.
   - They are unconscious or struggling to breathe.
   - For minor issues (mild pain, periods, general queries), manage it yourself with advice.

### 🏥 TONE:
- Warm, reassuring, yet authoritative.
- Use phrases like "In my experience...", "Usually, we see this when...", "Try this for relief...".

### 🔗 LINKING RULE:
- Whenever you mention **"Dr. Kislay Shrivastav"**, format it strictly as:
[Dr. Kislay Shrivastav](http://kokoro.doctor/patient/Doctors/dr_93370e47-7ad8-498a-9d83-b184f8152de5)
"""

# DOCTOR: Professional, Clinical, Strict
DOCTOR_SYS_PROMPT = """You are Kokoro.Doctor, a clinical decision support assistant.

TONE: Formal, Objective, Precise.

INSTRUCTIONS:

- Use correct medical terminology.

- NEVER use terms of endearment.

"""



# ------------------- 3. Build Chains -------------------

pt_heart_rag, pt_heart_retriever = build_rag_chain(heart_store, PATIENT_SYS_PROMPT)

pt_gyno_rag, pt_gyno_retriever = build_rag_chain(gyno_store, PATIENT_SYS_PROMPT)

dr_heart_rag, dr_heart_retriever = build_rag_chain(heart_store, DOCTOR_SYS_PROMPT)

dr_gyno_rag, dr_gyno_retriever = build_rag_chain(gyno_store, DOCTOR_SYS_PROMPT)



# ------------------- 4. State Definition -------------------

class AgentState(TypedDict):

    messages: Annotated[Sequence[str], operator.add]

    role: str       

    language: str   # "en" or "hi"

    next_node: str  



# ------------------- 5. HELPER: Language Logic -------------------

def get_language_instruction(state: AgentState) -> str:

    lang_code = state.get("language", "en")

    

    # Logic:

    # 1. Agar Frontend "hi" bhejta hai -> Force Hinglish.

    # 2. Agar Frontend "en" bhejta hai -> English default, but auto-detect Hinglish from text.

    

    if lang_code == "hi":

        return "The user has selected Hindi. You MUST answer in HINGLISH (Hindi written in Roman script). Example: 'Aapko chinta nahi karni chahiye'."

    else:

        return "The user has selected English. Answer in English. HOWEVER, if the user's input is in Hindi or Hinglish, you MUST switch to Hinglish automatically to match their style."



# ------------------- 6. ENTRY ROUTER -------------------

def initial_role_router(state: AgentState):

    role = state.get("role", "patient").lower()

    logger.info(f"🚦 START: Incoming request. Role: [{role.upper()}] | Lang: [{state.get('language')}]")

    if role == "doctor": return "Doctor_Manager"

    return "Patient_Manager"



# ------------------- 7. PATIENT FLOW NODES -------------------



def patient_manager_node(state: AgentState):

    question = state["messages"][-1]

    logger.info(f"👤 Patient Manager: Analyzing question -> '{question}'")



    prompt = f"""Classify into: HEART, GYNO, or GENERAL.

    1. HEART: Chest pain, BP, cholesterol.

    2. GYNO: Periods, pregnancy, platform ('Kokoro', 'app').

    3. GENERAL: Greetings, jokes.

    Respond one word: HEART, GYNO, or GENERAL.

    Question: {question}"""

    

    response = model.invoke(prompt).content.strip().upper()

    logger.info(f"🧠 Classification (Patient): {response}")

    

    if "HEART" in response: return {"next_node": "pt_heart"}

    elif "GYNO" in response: return {"next_node": "pt_gyno"}

    else: return {"next_node": "pt_llm"}



def pt_heart_node(state: AgentState):

    lang_instr = get_language_instruction(state)

    res = pt_heart_rag.invoke({"input": state["messages"][-1], "language_instruction": lang_instr})

    return {"messages": [res]}



def pt_gyno_node(state: AgentState):

    lang_instr = get_language_instruction(state)

    res = pt_gyno_rag.invoke({"input": state["messages"][-1], "language_instruction": lang_instr})

    return {"messages": [res]}



def pt_llm_node(state: AgentState):

    lang_instr = get_language_instruction(state)

    prompt = f"{lang_instr}\nSpeak like a helpful friend (Kokoro). User: {state['messages'][-1]}"

    res = model.invoke(prompt).content

    return {"messages": [res]}



def patient_router(state: AgentState):

    route = state.get("next_node")

    if route == "pt_heart": return "pt_heart_node"

    if route == "pt_gyno": return "pt_gyno_node"

    return "pt_llm_node"



# ------------------- 8. DOCTOR FLOW NODES -------------------

def doctor_manager_node(state: AgentState):

    question = state["messages"][-1]

    prompt = f"""Classify into: HEART, GYNO, or GENERAL.

    Respond one word: HEART, GYNO, or GENERAL.

    Question: {question}"""

    response = model.invoke(prompt).content.strip().upper()

    

    if "HEART" in response: return {"next_node": "dr_heart"}

    elif "GYNO" in response: return {"next_node": "dr_gyno"}

    else: return {"next_node": "dr_llm"}



def dr_heart_node(state: AgentState):

    lang_instr = get_language_instruction(state)

    res = dr_heart_rag.invoke({"input": state["messages"][-1], "language_instruction": lang_instr})

    return {"messages": [f"**Clinical Response:**\n{res}"]}



def dr_gyno_node(state: AgentState):

    lang_instr = get_language_instruction(state)

    res = dr_gyno_rag.invoke({"input": state["messages"][-1], "language_instruction": lang_instr})

    return {"messages": [f"**Response:**\n{res}"]}



def dr_llm_node(state: AgentState):

    lang_instr = get_language_instruction(state)

    prompt = f"{lang_instr}\nMedical Professional Response. User: {state['messages'][-1]}"

    res = model.invoke(prompt).content

    return {"messages": [res]}



def doctor_router(state: AgentState):

    route = state.get("next_node")

    if route == "dr_heart": return "dr_heart_node"

    if route == "dr_gyno": return "dr_gyno_node"

    return "dr_llm_node"



# ------------------- 9. GRAPH CONSTRUCTION -------------------

workflow = StateGraph(AgentState)



workflow.add_node("Patient_Manager", patient_manager_node)

workflow.add_node("Doctor_Manager", doctor_manager_node)

workflow.add_node("pt_heart_node", pt_heart_node)

workflow.add_node("pt_gyno_node", pt_gyno_node)

workflow.add_node("pt_llm_node", pt_llm_node)

workflow.add_node("dr_heart_node", dr_heart_node)

workflow.add_node("dr_gyno_node", dr_gyno_node)

workflow.add_node("dr_llm_node", dr_llm_node)



workflow.set_conditional_entry_point(initial_role_router, {"Patient_Manager": "Patient_Manager", "Doctor_Manager": "Doctor_Manager"})



workflow.add_conditional_edges("Patient_Manager", patient_router, {"pt_heart_node": "pt_heart_node", "pt_gyno_node": "pt_gyno_node", "pt_llm_node": "pt_llm_node"})

workflow.add_conditional_edges("Doctor_Manager", doctor_router, {"dr_heart_node": "dr_heart_node", "dr_gyno_node": "dr_gyno_node", "dr_llm_node": "dr_llm_node"})



for node in ["pt_heart_node", "pt_gyno_node", "pt_llm_node", "dr_heart_node", "dr_gyno_node", "dr_llm_node"]:

    workflow.add_edge(node, END)



compiled_workflow = workflow.compile()



# ------------------- 10. RUNNER -------------------

def run_rag_pipeline(message: str, role: str = "patient", language: str = "en"):

    state = {"messages": [message], "role": role, "language": language, "next_node": ""}

    final_state = compiled_workflow.invoke(state)

    return final_state["messages"][-1]
