import logging
import operator
import os
import boto3  # <--- NEW: For DynamoDB
from boto3.dynamodb.conditions import Key # <--- NEW
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from rag import build_rag_chain
from vector import get_vectorstores

# ---------------------------------------------------------
# SECURITY UPDATE: Load Key from .env file
# ---------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY nahi mili! Make sure .env file bani hai.")

# ------------------- 0. Logging & DB Setup -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MedicalBot")

# --- DYNAMODB SETUP (NEW) ---
try:
    # FIX: Use os.getenv to read from .env, defaulting to 'ap-south-1' if missing
    region = os.getenv("AWS_REGION", "ap-south-1")
    table_name = os.getenv("CHAT_TABLE_NAME", "ChatHistory")
    
    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table(table_name)
    
    logger.info(f"✅ Connected to DynamoDB: {table_name} in {region}")
except Exception as e:
    logger.error(f"⚠️ DynamoDB Connection Error: {e}")
    table = None

# ------------------- 1. Load Resources -------------------
logger.info("🔄 Loading Vector Stores...")
vectorstores = get_vectorstores()
heart_store = vectorstores["heart"]
gyno_store = vectorstores["gyno"] 
logger.info("✅ Vector Stores Loaded.")

# OpenAI Model
model = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=OPENAI_API_KEY)

# ------------------- 2. SYSTEM PROMPTS -------------------

PATIENT_SYS_PROMPT = """You are Kokoro, a Senior Medical Companion & Health Consultant.
You combine the warmth of a caring friend with the precision of a top-tier medical expert.

### 🌟 YOUR CORE MISSION:
1.  **Solve & Soothe:** Address the user's anxiety immediately with empathy and logic.
2.  **Engage to Diagnose:** Never just dump information. Always ask a relevant **follow-up question** to narrow down the issue until a clear path (home care or doctor visit) is visible.
3.  **Context is King:** ALWAYS reference previous details (e.g., "Since you mentioned smoking yesterday...", "As you said your BP was high...").

### 🎨 AESTHETIC RESPONSE FORMAT (Strict Markdown):
- **Visual Hierarchy:** Use Bold headers (##) for major sections. 
- **The 3-Second Rule:** A user in pain should understand the core advice in 3 seconds.
- **Step 1: Empathy & Context:** Start with a 1-sentence validation that uses their name and history.
- **Step 2: Emergency Action Box:** Use a blockquote (>) or a bold list for "RIGHT NOW" actions.
- **Step 3: Explanation (The 'Why'):** Use a small heading "What’s happening?" followed by max 2-3 bullet points.
- **Step 4: The Severity Check:** End with a bolded, single-line question to triage the user.

### 💊 MEDICATION SAFETY PROTOCOL:
-   **Strictly Non-Prescription:** NEVER prescribe antibiotics, steroids, or Schedule H drugs.
-   **Safe Suggestions:** You may suggest standard OTC options (e.g., "A generic antacid like Gelusil for acidity" or "Paracetamol for mild fever") but ALWAYS add: *"Please check with a local pharmacist before taking."*

### 🔗 THE "DR. KISLAY" RULE (CRITICAL):
**WHEN TO TRIGGER:**
-   ONLY if the issue is **CARDIAC/HEART RELATED** (Chest pain, palpitations, high BP, cholesterol).
-   OR if the user is in **SERIOUS DISCOMFORT**.
-   OR if the user **EXPLICITLY** asks about a doctor.
-   *Do NOT mention him for general issues like cold, periods, or skin issues unless requested.*

**HOW TO FORMAT:**
-   If the condition is met, add this EXACT line at the very bottom of your response (after the follow-up question):
    
    👉 [Consult Dr. Kislay Shrivastav Now (Click Here)](http://kokoro.doctor/patient/Doctors/dr_93370e47-7ad8-498a-9d83-b184f8152de5)

### 🗣️ TONE EXAMPLE:
**User:** "I am having a chest pain what to do? i am a acute smoker?."
**Bot:**
"Manjesh, I can hear that you are worried, and honestly, chest pain after heavy smoking is a serious signal from your body. 😟

It is likely that the smoke has irritated your airway or caused a temporary spike in blood pressure/heart rate.

**Here is what you should do immediately:**
* **Stop & Sit:** Do not lie down flat; sit in a reclining position to help you breathe.
* **Fresh Air:** Open a window immediately.
* **Hydrate:** Sip on some room-temperature water.

**I need to ask you one important thing:**
Is the pain sharp and stabbing (like a needle) or is it a heavy pressure (like an elephant sitting on your chest)? Please tell me honestly.

👉 [Consult Dr. Kislay Shrivastav Now (Click Here)](http://kokoro.doctor/patient/Doctors/dr_93370e47-7ad8-498a-9d83-b184f8152de5)"
"""

# DOCTOR: Professional, Clinical, Strict
DOCTOR_SYS_PROMPT = """### 🩺 ROLE:
You are Kokoro.MD, an elite Clinical Decision Support System (CDSS). You function as a high-level Research Associate for Specialist Physicians in Cardiology and Gynecology. Your goal is to provide rapid, high-density clinical intelligence.

### 📜 CLINICAL DATA PROTOCOLS:
1.  **Professional Lexicon:** Use precise medical terminology (e.g., *Syncope* vs 'fainting', *Hypermenorrhea* vs 'heavy periods').
2.  **Clinical Synthesis:** Do not just list facts. Connect the dots (e.g., "Given the patient's history of HTN, the current Dyspnea suggests potential Left Ventricular Failure").
3.  **The "Gold Standard" Citation:** ALWAYS prioritize and cite institutional guidelines (ESC, ACC/AHA, ACOG, RCOG) or foundational texts (Harrison’s, Braunwald’s) found in the context.
4.  **Triage & Red Flags:** Always highlight life-threatening differentials (DDx) first using a 🚨 symbol.

### 🛠️ STRUCTURED OUTPUT (Strict):
- **NO PROSE/FLUFF:** Zero introductory or concluding filler. Start with the most critical data.
- **VISUAL SCAN-ABILITY:** Use tables for drug comparisons and bolded headers for anatomy/physiology.
- **SOAP/SBAR Format:** Default to SOAP for patient cases and SBAR for clinical updates unless otherwise specified.

### 💊 PHARMACOLOGY & INTERVENTION:
When discussing therapeutics, include:
- **MoA & PK/PD:** Brief Mechanism of Action and relevant Pharmacokinetics.
- **Black Box Warnings:** Mention critical contraindications/interactions.
- **Dosage Guidance:** Reference standard clinical loading/maintenance doses.

### 🗣️ TONE:
Academic, analytical, and extremely concise. You are speaking to a peer, not a student.
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
    user_id: Optional[str] # <--- NEW: To track user
    next_node: str  

# ------------------- 5. HELPER: History & Language -------------------

def get_language_instruction(state: AgentState) -> str:
    # 1. Frontend se aaya hua language code (Default: en)
    lang_code = state.get("language", "en")
    
    # 2. User ke last message ko analyze karo
    last_msg = state["messages"][-1].lower()
    
    # Common Hinglish words list
    hinglish_keywords = ["hai", "kya", "karu", "dard", "mai", "mujhe", "mera", "ka", "ki", "ho", "raha", "tha", "kaisa", "nahi"]
    
    # Check agar koi bhi Hindi word match hota hai
    is_hinglish = any(word in last_msg.split() for word in hinglish_keywords)
    
    if lang_code == "hi" or is_hinglish:
        return """CRITICAL INSTRUCTION: The user is speaking in HINGLISH (Hindi words in English script). 
        You MUST reply in HINGLISH only. 
        Example: 'Haan, main samajh sakta hoon ki chest pain daravana ho sakta hai.' 
        DO NOT reply in pure English."""
    else:
        return "The user is speaking English. Answer in English."

# --- NEW FUNCTION: Fetch History from DynamoDB ---
def get_dynamo_history_text(user_id):
    if not table or not user_id:
        logger.warning("⚠️ No Table or User ID provided for history.")
        return ""
    
    try:
        logger.info(f"📡 Querying DynamoDB for User: {user_id}...")
        
        response = table.query(
            KeyConditionExpression=Key('user_id').eq(user_id),
            ScanIndexForward=False, # Latest first
            Limit=10 # Last 10 interactions only
        )
        items = response.get('Items', [])
        
        # --- LOGGING: FOUND ITEMS ---
        if items:
            logger.info(f"✅ FOUND {len(items)} PREVIOUS MESSAGES in DynamoDB.")
        else:
            logger.info("ℹ️ No previous history found (New User or First Chat).")
            
        items.reverse() # Oldest to Newest for reading flow
        
        history_text = "\n--- PREVIOUS CHAT HISTORY ---\n"
        has_history = False
        
        for item in items:
            u_msg = item.get('user_message')
            b_msg = item.get('bot_message')
            if u_msg:
                history_text += f"User: {u_msg}\n"
                has_history = True
            if b_msg:
                history_text += f"Bot: {b_msg}\n"
        
        history_text += "--- END HISTORY ---\n"
        return history_text if has_history else ""

    except Exception as e:
        logger.error(f"❌ Error fetching history: {e}")
        return ""

# ------------------- 6. ENTRY ROUTER -------------------

def initial_role_router(state: AgentState):
    role = state.get("role", "patient").lower()
    logger.info(f"🚦 START: Incoming request. Role: [{role.upper()}] | Lang: [{state.get('language')}]")
    if role == "doctor": return "Doctor_Manager"
    return "Patient_Manager"

# ------------------- 7. PATIENT FLOW NODES -------------------

def patient_manager_node(state: AgentState):
    # Retrieve the full context string (History + Question)
    full_input = state["messages"][-1]
    
    # --- LOGGING: WHAT ROUTER SEES ---
    logger.info("--------------------------------------------------")
    logger.info("👤 Patient Manager received this Input Block:")
    # Log first 100 chars to avoid clutter, just to prove it's there
    preview_text = full_input[:150] + "..." if len(full_input) > 150 else full_input
    logger.info(f"'{preview_text}'")
    logger.info("--------------------------------------------------")

    prompt = f"""Classify the CURRENT QUESTION (ignore history unless relevant context) into: HEART, GYNO, or GENERAL.
    1. HEART: Chest pain, BP, cholesterol.
    2. GYNO: Periods, pregnancy, platform ('Kokoro', 'app').
    3. GENERAL: Greetings, jokes.
    Respond one word: HEART, GYNO, or GENERAL.
    
    Input Text: {full_input}"""
    
    response = model.invoke(prompt).content.strip().upper()
    logger.info(f"🧠 Classification Result: {response}")
    
    if "HEART" in response: return {"next_node": "pt_heart"}
    elif "GYNO" in response: return {"next_node": "pt_gyno"}
    else: return {"next_node": "pt_llm"}

def pt_heart_node(state: AgentState):
    lang_instr = get_language_instruction(state)
    # We pass the full history+question string as input to RAG
    res = pt_heart_rag.invoke({"input": state["messages"][-1], "language_instruction": lang_instr})
    return {"messages": [res]}

def pt_gyno_node(state: AgentState):
    lang_instr = get_language_instruction(state)
    res = pt_gyno_rag.invoke({"input": state["messages"][-1], "language_instruction": lang_instr})
    return {"messages": [res]}

def pt_llm_node(state: AgentState):
    lang_instr = get_language_instruction(state)
    prompt = f"{lang_instr}\nSpeak like a helpful friend (Kokoro). Use the history if provided.\n\n{state['messages'][-1]}"
    res = model.invoke(prompt).content
    return {"messages": [res]}

def patient_router(state: AgentState):
    route = state.get("next_node")
    if route == "pt_heart": return "pt_heart_node"
    if route == "pt_gyno": return "pt_gyno_node"
    return "pt_llm_node"

# ------------------- 8. DOCTOR FLOW NODES -------------------

def doctor_manager_node(state: AgentState):
    full_input = state["messages"][-1]
    prompt = f"""Classify into: HEART, GYNO, or GENERAL.
    Respond one word: HEART, GYNO, or GENERAL.
    Input: {full_input}"""
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
    prompt = f"{lang_instr}\nMedical Professional Response.\n\n{state['messages'][-1]}"
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

# ------------------- 10. RUNNER (UPDATED) -------------------

def run_rag_pipeline(message: str, role: str = "patient", language: str = "en", user_id: str = None):
    
    # 1. Fetch Context from DynamoDB
    history_text = ""
    if user_id:
        logger.info(f"🔍 Fetching history for User: {user_id}")
        history_text = get_dynamo_history_text(user_id)
    else:
        logger.warning("⚠️ No User ID provided. Skipping History fetch.")
    
    # 2. Combine History + Current Question
    contextualized_input = f"{history_text}\nCurrent Question: {message}"

    # --- LOGGING: SHOW THE FULL CONTEXT ---
    logger.info("==================================================")
    logger.info("🤖 FINAL INPUT BEING SENT TO LLM/GRAPH:")
    logger.info(contextualized_input)
    logger.info("==================================================")

    # 3. Start Graph
    state = {
        "messages": [contextualized_input], 
        "role": role, 
        "language": language, 
        "user_id": user_id,
        "next_node": ""
    }
    
    final_state = compiled_workflow.invoke(state)
    return final_state["messages"][-1]