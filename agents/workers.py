from .prompts import ARCHITECT_PROMPT, WRITER_PROMPT, SKEPTIC_PROMPT, EXEC_PROMPT, RESEARCHER_PROMPT, CODER_PROMPT
from .client import get_client, convert_messages
from google.genai import types
from langchain_core.messages import AIMessage
import re

def create_agent_node(name, system_prompt, tools=None):
    def agent_node(state):
        messages = state['messages']
        
        # 1. LUKU: Haetaan nykyinen muistio tilasta
        current_memo = state.get("memo", "Ei merkintöjä.")
        
        # 2. INJEKTOINTI: Lisätään muistio System Promptin loppuun
        augmented_system_prompt = f"""{system_prompt}

        === SHARED MEMORY (READ-ONLY) ===
        The Council maintains a shared memo for key facts, decisions, and constraints.
        CURRENT MEMO:
        {current_memo}
        
        === HOW TO UPDATE THE MEMO ===
        To update this memo, include a block in your response like this:
        [MEMO_UPDATE]
        - User prefers spicy food.
        - Budget is 500 euros.
        [/MEMO_UPDATE]
        Anything inside these tags will overwrite the previous memo. Keep it concise.
        """
        
        client = get_client()
        contents = convert_messages(messages)
        
        config = types.GenerateContentConfig(
                system_instruction=augmented_system_prompt,
                safety_settings=[ # Relax safety settings for creative writing
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH"
                    ),
                ]
            )
        
        if tools:
            config.tools = tools
            # If using tools, we might want to use Pro for better reasoning/tool use
            model_name = "gemini-3-pro-preview" 
        else:
            model_name = "gemini-2.5-flash"

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
            final_text = response.text
        except Exception as e:
            final_text = f"I encountered a technical issue: {str(e)}. Let me try to rephrase or approach this differently."

        # Fallback if text is empty but no exception raised
        if not final_text:
             if response.candidates and response.candidates[0].finish_reason:
                 final_text = f"*[System Note: The Agent generated no content. Finish Reason: {response.candidates[0].finish_reason}]*"
             else:
                 final_text = "I have nothing to add at this moment."

        # 3. KIRJOITUS & PARSINTA: Etsitään päivitys tekstistä
        new_memo = current_memo # Oletuksena vanha säilyy
        
        # Etsitään [MEMO_UPDATE]...[/MEMO_UPDATE] lohko
        if final_text:
            match = re.search(r'\[MEMO_UPDATE\](.*?)\[/MEMO_UPDATE\]', final_text, re.DOTALL)
            if match:
                # Löydettiin päivitys!
                memo_content = match.group(1).strip()
                new_memo = memo_content
                # Poistetaan tekninen blokki lopullisesta viestistä
                final_text = final_text.replace(match.group(0), "").strip()

        return {
            "messages": [AIMessage(content=final_text, name=name)],
            "memo": new_memo 
        }
    return agent_node

architect_node = create_agent_node("Architect", ARCHITECT_PROMPT)
writer_node = create_agent_node("Writer", WRITER_PROMPT)
skeptic_node = create_agent_node("Skeptic", SKEPTIC_PROMPT)
exec_node = create_agent_node("Exec", EXEC_PROMPT)
researcher_node = create_agent_node("Researcher", RESEARCHER_PROMPT, tools=[{"google_search": {}}])
coder_node = create_agent_node("Coder", CODER_PROMPT)

def adhoc_node(state):
    # 1. Retrieve the persona requested by the Chairman
    persona = state.get("adhoc_persona", "A Helpful Assistant")
    current_memo = state.get("memo", "No notes yet.")
    
    # 2. Construct a System Prompt on the fly (INCLUDE MEMO)
    system_prompt = f"""You are {persona}. 
    Your goal is to fulfill the user's specific request using the expertise of this role.
    Do not break character.
    
    === SHARED MEMORY ===
    CURRENT MEMO: {current_memo}
    """
    
    # 3. Call Gemini with this specific persona
    messages = state['messages']
    client = get_client()
    contents = convert_messages(messages)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_prompt)
    )
    
    # 4. Return message with the dynamic name
    return {"messages": [AIMessage(content=response.text, name=persona)]}