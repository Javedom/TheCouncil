from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from typing import List
import os

_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return _client

def convert_messages(messages: List[BaseMessage]) -> List[types.Content]:
    contents = []
    
    # We consolidate history into a single 'user' turn if possible, 
    # or ensure we never send 'model' -> 'model'.
    # A robust strategy for Multi-Agent debate is to treat the history 
    # as a transcript provided by the 'user'.
    
    running_transcript = []
    
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
            
        role_label = "User"
        if isinstance(msg, AIMessage):
            # Use the name if available, otherwise 'Agent'
            role_label = getattr(msg, 'name', 'Agent')
        
        # Format: [Architect]: content...
        running_transcript.append(f"[{role_label}]: {msg.content}")

    # Combine everything into one big context block for the model
    # This prevents the "Alternating Role" error from Google API
    full_history = "\n\n".join(running_transcript)
    
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=full_history)]
        )
    )
    
    return contents