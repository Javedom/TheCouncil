import operator
from typing import Annotated, List, Union, TypedDict
from langchain_core.messages import BaseMessage

class CouncilState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str
    adhoc_persona: str
    memo: str