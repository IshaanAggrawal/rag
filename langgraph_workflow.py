import logging
import operator
import os
import boto3
from boto3.dynamodb.conditions import Key
from typing import TypedDict, Annotated, Sequence, Optional
from langgraph.graph import StateGraph, END
from llm_config import get_llm
from rag import build_rag_chain
from vector import get_vectorstores

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("MedicalBot")

try:
    region = os.getenv("AWS_REGION", "ap-south-1")
    table_name = os.getenv("CHAT_TABLE_NAME", "ChatHistory")
    dynamodb = boto3.resource('dynamodb', region_name=region)
    table = dynamodb.Table(table_name)
    logger.info(f"✅ Connected to DynamoDB: {table_name} in {region}")
except Exception as e:
    logger.error(f"⚠️ DynamoDB Connection Error: {e}")
    table = None

logger.info("🔄 Loading Vector Stores...")
vectorstores = get_vectorstores()
heart_store = vectorstores.get("heart")
gyno_store = vectorstores.get("gyno")
insurance_store = vectorstores.get("insurance")
logger.info("✅ Vector Stores Loaded.")

# Dynamic LLM instance for the Router nodes
model = get_llm(temperature=0.1)

# --- KEEP YOUR EXACT PROMPTS HERE ---
PATIENT_SYS_PROMPT = """You are Kokoro, a Senior Medical Companion... (Keep your full prompt here)"""
DOCTOR_SYS_PROMPT = """### 🩺 ROLE: You are Kokoro.MD... (Keep your full prompt here)"""

# Build Chains
pt_heart_rag, _ = build_rag_chain(heart_store, PATIENT_SYS_PROMPT) if heart_store else (None, None)
pt_gyno_rag, _ = build_rag_chain(gyno_store, PATIENT_SYS_PROMPT) if gyno_store else (None, None)
dr_heart_rag, _ = build_rag_chain(heart_store, DOCTOR_SYS_PROMPT) if heart_store else (None, None)
dr_gyno_rag, _ = build_rag_chain(gyno_store, DOCTOR_SYS_PROMPT) if gyno_store else (None, None)

# Insurance Chains
pt_ins_rag, _ = build_rag_chain(insurance_store, PATIENT_SYS_PROMPT) if insurance_store else (None, None)
dr_ins_rag, _ = build_rag_chain(insurance_store, DOCTOR_SYS_PROMPT) if insurance_store else (None, None)

class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    role: str       
    language: str   
    user_id: Optional[str] 
    next_node: str  

def get_language_instruction(state: AgentState) -> str:
    if state.get("language", "en") == "hi":
        return "CRITICAL INSTRUCTION: The user has selected Hindi/Hinglish button. You MUST reply in HINGLISH ONLY."
    return "The user has selected English. Answer in English only."

def get_dynamo_history_text(user_id):
    if not table or not user_id: return ""
    try:
        response = table.query(KeyConditionExpression=Key('user_id').eq(user_id), ScanIndexForward=False, Limit=3)
        items = response.get('Items', [])
        items.reverse()
        history_text = "\n--- PREVIOUS CHAT HISTORY ---\n"
        has_history = False
        for item in items:
            if item.get('user_message'):
                history_text += f"User: {item.get('user_message')}\n"
                has_history = True
            if item.get('bot_message'): history_text += f"Bot: {item.get('bot_message')}\n"
        return history_text + "--- END HISTORY ---\n" if has_history else ""
    except: return ""

def initial_role_router(state: AgentState):
    return "Doctor_Manager" if state.get("role", "patient").lower() == "doctor" else "Patient_Manager"

def patient_manager_node(state: AgentState):
    prompt = f"""Classify the CURRENT QUESTION into exactly ONE of the following: HEART, GYNO, INSURANCE, or GENERAL.
    1. HEART: Chest pain, BP, heart attacks.
    2. GYNO: Periods, pregnancy, 'Kokoro' platform, pricing.
    3. INSURANCE: Health insurance, coverage, claims, PDF details.
    4. GENERAL: Basic greetings.
    Respond with ONE WORD only. Input: {state["messages"][-1]}"""
    
    response = model.invoke(prompt).content.strip().upper()
    if "HEART" in response: return {"next_node": "pt_heart"}
    elif "GYNO" in response: return {"next_node": "pt_gyno"}
    elif "INSURANCE" in response: return {"next_node": "pt_insurance"}
    else: return {"next_node": "pt_llm"}

def pt_heart_node(state: AgentState):
    if not pt_heart_rag:
        return pt_llm_node(state)
    return {"messages": [pt_heart_rag.invoke({"input": state["messages"][-1], "language_instruction": get_language_instruction(state)})]}

def pt_gyno_node(state: AgentState):
    if not pt_gyno_rag:
        return pt_llm_node(state)
    return {"messages": [pt_gyno_rag.invoke({"input": state["messages"][-1], "language_instruction": get_language_instruction(state)})]}

def pt_insurance_node(state: AgentState):
    if not pt_ins_rag: return {"messages": ["Insurance database not loaded."]}
    return {"messages": [pt_ins_rag.invoke({"input": state["messages"][-1], "language_instruction": get_language_instruction(state)})]}

def pt_llm_node(state: AgentState):
    return {"messages": [model.invoke(f"{get_language_instruction(state)}\nSpeak like Kokoro.\n\n{state['messages'][-1]}").content]}

def patient_router(state: AgentState):
    return state.get("next_node") + "_node"

def doctor_manager_node(state: AgentState):
    prompt = f"""Classify the CURRENT QUESTION into exactly ONE of the following: HEART, GYNO, INSURANCE, or GENERAL.
    Respond with ONE WORD only. Input: {state["messages"][-1]}"""
    
    response = model.invoke(prompt).content.strip().upper()
    if "HEART" in response: return {"next_node": "dr_heart"}
    elif "GYNO" in response: return {"next_node": "dr_gyno"}
    elif "INSURANCE" in response: return {"next_node": "dr_insurance"}
    else: return {"next_node": "dr_llm"}

def dr_heart_node(state: AgentState):
    if not dr_heart_rag:
        return dr_llm_node(state)
    return {"messages": [f"**Clinical Response:**\n{dr_heart_rag.invoke({'input': state['messages'][-1], 'language_instruction': get_language_instruction(state)})}"]}

def dr_gyno_node(state: AgentState):
    if not dr_gyno_rag:
        return dr_llm_node(state)
    return {"messages": [f"**Response:**\n{dr_gyno_rag.invoke({'input': state['messages'][-1], 'language_instruction': get_language_instruction(state)})}"]}

def dr_insurance_node(state: AgentState):
    if not dr_ins_rag: return {"messages": ["System Error: Insurance DB missing."]}
    return {"messages": [f"**Insurance Context:**\n{dr_ins_rag.invoke({'input': state['messages'][-1], 'language_instruction': get_language_instruction(state)})}"]}

def dr_llm_node(state: AgentState):
    return {"messages": [model.invoke(f"{get_language_instruction(state)}\nMedical Professional Response.\n\n{state['messages'][-1]}").content]}

def doctor_router(state: AgentState):
    return state.get("next_node") + "_node"

workflow = StateGraph(AgentState)
workflow.add_node("Patient_Manager", patient_manager_node)
workflow.add_node("Doctor_Manager", doctor_manager_node)
workflow.add_node("pt_heart_node", pt_heart_node)
workflow.add_node("pt_gyno_node", pt_gyno_node)
workflow.add_node("pt_insurance_node", pt_insurance_node)
workflow.add_node("pt_llm_node", pt_llm_node)
workflow.add_node("dr_heart_node", dr_heart_node)
workflow.add_node("dr_gyno_node", dr_gyno_node)
workflow.add_node("dr_insurance_node", dr_insurance_node)
workflow.add_node("dr_llm_node", dr_llm_node)

workflow.set_conditional_entry_point(initial_role_router, {"Patient_Manager": "Patient_Manager", "Doctor_Manager": "Doctor_Manager"})
workflow.add_conditional_edges("Patient_Manager", patient_router, {"pt_heart_node": "pt_heart_node", "pt_gyno_node": "pt_gyno_node", "pt_insurance_node": "pt_insurance_node", "pt_llm_node": "pt_llm_node"})
workflow.add_conditional_edges("Doctor_Manager", doctor_router, {"dr_heart_node": "dr_heart_node", "dr_gyno_node": "dr_gyno_node", "dr_insurance_node": "dr_insurance_node", "dr_llm_node": "dr_llm_node"})

for node in ["pt_heart_node", "pt_gyno_node", "pt_insurance_node", "pt_llm_node", "dr_heart_node", "dr_gyno_node", "dr_insurance_node", "dr_llm_node"]:
    workflow.add_edge(node, END)

compiled_workflow = workflow.compile()

def run_rag_pipeline(message: str, role: str = "patient", language: str = "en", user_id: str = None):
    history_text = get_dynamo_history_text(user_id) if user_id else ""
    contextualized_input = f"{history_text}\nCurrent Question: {message}"
    state = {"messages": [contextualized_input], "role": role, "language": language, "user_id": user_id, "next_node": ""}
    final_state = compiled_workflow.invoke(state)
    return final_state["messages"][-1]