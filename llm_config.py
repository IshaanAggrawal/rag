import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_llm(temperature=0.1):
    """
    Returns the appropriate LangChain Chat model based on environment configuration.
    Supports:
    1. Groq (Fast Cloud LLM)
    2. OpenAI (GPT Models)
    3. Ollama (100% Local LLM) - Default
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()
    
    # Auto-detect provider based on keys if not explicitly set
    if not provider:
        if os.getenv("GROQ_API_KEY"):
            provider = "groq"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "ollama"
            
    print(f"🤖 LLM Provider Selected: {provider.upper()}")
    
    if provider == "groq":
        from langchain_openai import ChatOpenAI
        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY must be set in .env when LLM_PROVIDER is groq")
            
        return ChatOpenAI(
            openai_api_base=groq_base_url,
            openai_api_key=groq_api_key,
            model_name=groq_model,
            temperature=temperature
        )
        
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env when LLM_PROVIDER is openai")
            
        return ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=openai_model,
            temperature=temperature
        )
        
    else:
        # Default to Ollama
        from langchain_ollama import ChatOllama
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        
        if not ollama_base_url:
            # Check if running inside Docker container
            if os.path.exists("/.dockerenv") or os.environ.get("RUNNING_IN_DOCKER") == "true":
                ollama_base_url = "http://host.docker.internal:11434"
            else:
                ollama_base_url = "http://localhost:11434"
                
        print(f"🔗 Ollama Connecting to: {ollama_base_url} (model: {ollama_model})")
        return ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=temperature
        )
