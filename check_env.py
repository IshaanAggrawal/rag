#!/usr/bin/env python3
"""
Environment Variables Validation Script
Checks if all required API keys and credentials are properly set
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding issues with emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def check_env_vars(env_path=".env"):
    """Validate environment variables from .env file"""
    
    # Check if .env file exists
    if not Path(env_path).exists():
        print(f"❌ ERROR: {env_path} file not found!")
        print(f"\n💡 Create it by copying the example:")
        print(f"   cp {env_path}.example {env_path}")
        return False
    
    # Load .env file manually (without python-dotenv dependency)
    env_vars = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    
    # Define required and optional variables based on LLM_PROVIDER
    provider = env_vars.get('LLM_PROVIDER', '').lower()
    required = {}
    
    if provider == 'groq':
        required['GROQ_API_KEY'] = 'Groq API key for LLM models'
    elif provider == 'openai':
        required['OPENAI_API_KEY'] = 'OpenAI API key for GPT models'
    elif provider == 'ollama':
        pass
    else:
        if env_vars.get('GROQ_API_KEY'):
            required['GROQ_API_KEY'] = 'Groq API key for LLM models'
        elif env_vars.get('OPENAI_API_KEY'):
            required['OPENAI_API_KEY'] = 'OpenAI API key for GPT models'
        else:
            required['GROQ_API_KEY'] = 'Groq API key (Required if LLM_PROVIDER is groq)'
    
    optional = {
        'AWS_REGION': 'AWS Region (default: ap-south-1)',
        'AWS_ACCESS_KEY_ID': 'AWS Access Key (required for DynamoDB/Lambda)',
        'AWS_SECRET_ACCESS_KEY': 'AWS Secret Key (required for DynamoDB/Lambda)',
        'CHAT_TABLE_NAME': 'Chat history table (default: ChatHistory)',
        'USERS_TABLE_NAME': 'Users table (default: Users)'
    }
    
    print("=" * 70)
    print(f"ENVIRONMENT VARIABLES CHECK: {env_path}")
    print("=" * 70)
    
    # Check required
    print("\n🔴 REQUIRED:")
    missing = []
    for key, desc in required.items():
        value = env_vars.get(key, '')
        if value and not value.startswith('your-') and not value.startswith('sk-proj-your'):
            # Mask the key for security
            if len(value) > 12:
                masked = value[:8] + "..." + value[-4:]
            else:
                masked = "***"
            print(f"  ✅ {key}: {masked}")
            print(f"     ({desc})")
        else:
            print(f"  ❌ {key}: MISSING or PLACEHOLDER")
            print(f"     ({desc})")
            missing.append(key)
    
    # Check optional
    print("\n🟡 OPTIONAL:")
    for key, desc in optional.items():
        value = env_vars.get(key, '')
        if value and not value.startswith('your-'):
            if 'SECRET' in key or 'KEY' in key:
                # Mask sensitive values
                if len(value) > 12:
                    masked = value[:8] + "..." + value[-4:]
                else:
                    masked = "***"
                print(f"  ✅ {key}: {masked}")
            else:
                print(f"  ✅ {key}: {value}")
            print(f"     ({desc})")
        else:
            print(f"  ⚠️  {key}: Not set or placeholder")
            print(f"     ({desc})")
    
    # Summary
    print("\n" + "=" * 70)
    if missing:
        print(f"❌ VALIDATION FAILED")
        print(f"\n{len(missing)} required variable(s) missing or using placeholder:")
        for key in missing:
            print(f"  - {key}")
        print(f"\n💡 Edit {env_path} and replace placeholders with actual values")
        return False
    else:
        print("✅ VALIDATION PASSED")
        print("\nAll required variables are properly set!")
        print("\n💡 Next steps:")
        print("  1. Keep your .env file secure (never commit to Git)")
        print("  2. Test your application: python app.py")
        return True

def main():
    """Main function to check both systems"""
    
    print("\n" + "🔍 CHECKING ENVIRONMENT VARIABLES" + "\n")
    
    # Check Main RAG
    print("\n📁 Main RAG System:")
    main_rag_ok = check_env_vars(".env")
    
    # Check Booking Agent (only if directory exists)
    booking_agent_dir = Path("Booking_agent")
    booking_agent_ok = True
    has_booking_agent = booking_agent_dir.exists() and booking_agent_dir.is_dir()
    
    if has_booking_agent:
        print("\n\n📁 Booking Agent System:")
        booking_agent_ok = check_env_vars("Booking_agent/.env")
    
    # Final summary
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if main_rag_ok:
        print("✅ Main RAG: Ready to use")
    else:
        print("❌ Main RAG: Configuration needed")
        
    if has_booking_agent:
        if booking_agent_ok:
            print("✅ Booking Agent: Ready to use")
        else:
            print("❌ Booking Agent: Configuration needed")
    
    print("\n")
    
    # Exit code
    if main_rag_ok and (not has_booking_agent or booking_agent_ok):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
