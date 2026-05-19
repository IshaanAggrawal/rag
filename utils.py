import re
import emoji

def clean_text_for_speech(text: str) -> str:
    if not text:
        return ""
    
    # 1. Emojis hatana
    text = emoji.replace_emoji(text, replace='')
    
    # 2. Markdown symbols hatana (*, #, -, `)
    text = re.sub(r'[\*\#\`\-\_]', '', text)
    
    # 3. Links/URLs hatana
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 4. Brackets aur unke andar ka text hatana (Jaise [Click Here])
    text = re.sub(r'\[.*?\]', '', text)
    
    # 5. Extra spaces clean karna
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text