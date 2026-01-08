from .prompts import CHAIRMAN_PROMPT
from .client import get_client, convert_messages
from google.genai import types
from pydantic import BaseModel
from typing import Literal

# Define the routing schema strictly
class Route(BaseModel):
    next_step: Literal["Architect", "Writer", "Skeptic", "Exec", "Researcher", "Coder", "AdHoc", "FINISH"]
    # If next_step is AdHoc, this field is required (e.g., "The Chef")
    dynamic_persona: str = None

def chairman_node(state):
    messages = state['messages']
    client = get_client()
    
    # helper to convert LangChain messages to Gemini SDK contents
    contents = convert_messages(messages)
    
    # We send the history + system prompt via config
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=CHAIRMAN_PROMPT,
            response_mime_type="application/json",
            response_json_schema=Route.model_json_schema()
        )
    )
    
    # Parse the result
    try:
        route = Route.model_validate_json(response.text)
        return {
        "next_agent": route.next_step,
        "adhoc_persona": route.dynamic_persona # Pass this to the state
    }
    except Exception as e:
        # Fallback or error handling
        print(f"Error parsing routing decision: {e}")
        return {"next_agent": "Exec"}
