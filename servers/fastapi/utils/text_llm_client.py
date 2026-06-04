from typing import Any

from llmai import get_client

from enums.llm_provider import LLMProvider
from utils.custom_completion_path_client import get_custom_completion_path_client
from utils.llm_config import get_llm_config
from utils.llm_provider import get_llm_provider


def get_text_llm_client() -> Any:
    if get_llm_provider() == LLMProvider.CUSTOM:
        custom_client = get_custom_completion_path_client()
        if custom_client is not None:
            return custom_client
    return get_client(config=get_llm_config())
