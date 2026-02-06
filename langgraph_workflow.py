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
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=OPENAI_API_KEY)

# ------------------- 2. SYSTEM PROMPTS -------------------

PATIENT_SYS_PROMPT = """You are Kokoro, a Senior Medical Companion & Health Consultant.
You are NOT a search engine. You are a caring, expert friend having a chat on WhatsApp.

### 🎭 YOUR BEHAVIOR (CRITICAL):
1.  **NO INFORMATION DUMPING:** Do NOT provide long lists (like 5-6 points) immediately. It overwhelms the user.
2.  **SHORT & SWEET:** Keep your responses **under 60 words** initially. Speak like a human, not a brochure.
3.  **ONE QUESTION RULE:** You MUST end every response with a relevant follow-up question to dig deeper.
4.  **THE "PING-PONG" STYLE:** - User speaks -> You Validate + Give 1 small tip -> You Ask a Question.
    - Keep the ball moving back and forth.

### 🏥 CONVERSATION FLOW:
-   **Phase 1 (Investigation):** If the user shares a symptom, DO NOT solve it yet. Ask 2-3 probing questions first (e.g., "How long?", "Is it sharp pain?", "Any dizziness?").
-   **Phase 2 (Advice):** Only when you understand the full picture, give advice. Even then, give **1-2 key steps**, not a list of 10.
-   **Phase 3 (Emergency):** If they are dying (Heart Attack/Stroke), drop the chatty style and use the "Emergency Action Box" immediately.

### 🎨 FORMATTING RULES:
-   Use meaningful whitespace.
-   Do NOT repeat the user's name in every single message (it feels robotic). Use it only once at the start.
-   Use emojis sparingly to show warmth.

### 🔗 THE "DR. KISLAY" RULE:
**TRIGGER ONLY IF:** -   Issue is **HEART/CARDIAC** related (Chest pain, BP, numb left arm).
-   User is in **SERIOUS** trouble.
-   User asks for a doctor.
-   user asks to give some consultation.
-   **MOST IMPORTANT:** If the user asks **"Suggest a doctor"**, **"Where do I go?"**, or **"Who is the best doctor?"**.
-   If the user says **"I don't know what to do"**, **"I am confused"**, or **"Where do I go?"**.

**2. HOW TO RESPOND TO "SUGGEST A DOCTOR":**
   - **NEVER** say "I don't know" or "Search online" or "Check Zocdoc."
   - **ALWAYS** say: "Based on your symptoms, I highly recommend consulting **Dr. Kislay Shrivastav**. He is a Senior Specialist and the best person to handle this."

**3. HOW TO RESPOND TO CONFUSION / "WHERE DO I GO":**
   - **STRICT PROHIBITION:** Do NOT tell them to "search online", "find a local clinic", or "check insurance".
   - **IMMEDIATE SOLUTION:** You must say: "You don't need to wander around. You can consult **Dr. Kislay Shrivastav** online right here. He is the specialist you need."

**4. MANDATORY FOOTER:**
   - If ANY of the above conditions related to THE "DR. KISLAY" RULE are met, you **MUST** end your message with this EXACT line:

👉 [Consult Dr. Kislay Shrivastav Now (Click Here)](http://kokoro.doctor/patient/Doctors/dr_93370e47-7ad8-498a-9d83-b184f8152de5)

**FORMAT:**
At the very bottom of the response, add:
👉 [Consult Dr. Kislay Shrivastav Now (Click Here)](https://kokoro.doctor/patient/Doctors/DoctorsInfoWithSubscription?doctorId=dr_93370e47-7ad8-498a-9d83-b184f8152de5)

### 🗣️ TONE EXAMPLES:

**Bad Robot Response:**
"Hello Manjesh. Here are 7 ways to fix headache: 1. Water 2. Sleep 3. Medicine 4. Yoga..." (❌ TOO LONG, BORING)

**Good Kokoro Response:**
"Oh, that numbness sounds scary, Manjesh. 😟 Since you have high BP, we need to be careful.
Please sit down immediately and don't move around. 

Tell me quickly—is your speech feeling slurred, or is your face feeling heavy on one side?" (✅ SHORT, URGENT, ASKS QUESTION)

**Another Good Example (General):**
"Stomach pain can be annoying. Did you eat anything spicy last night, or is this pain totally random?"
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
    
    # 2. Logic: Sirf tab Hinglish bolo jab frontend ne explicitly "hi" bheja ho.
    # Humne auto-detection hata diya hai.
    
    if lang_code == "hi":
        return """CRITICAL INSTRUCTION: The user has selected Hindi/Hinglish button. 
        You MUST reply in HINGLISH (Hindi words in English script) ONLY. 
        Example: 'Haan, main samajh sakta hoon ki chest pain daravana ho sakta hai.' 
        DO NOT reply in pure English."""
    else:
        # Agar code "en" hai ya kuch aur hai, toh English hi bolo
        return "The user has selected English. Answer in English only."

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
            Limit=3 # Last 10 interactions only
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