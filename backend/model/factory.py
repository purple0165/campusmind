import os
from typing import TYPE_CHECKING, Any, Dict, Optional

try:
    from langchain_community.chat_models import ChatTongyi
    from langchain_community.embeddings import DashScopeEmbeddings
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False
    if TYPE_CHECKING:
        from langchain_community.chat_models import ChatTongyi
        from langchain_community.embeddings import DashScopeEmbeddings

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False


class ModelFactory:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _has_api_key(self) -> bool:
        env_api_key = os.getenv("DASHSCOPE_API_KEY")
        if env_api_key:
            return True
        model_cfg = self.config.get("model", {})
        api_key = model_cfg.get("api_key")
        return bool(api_key)

    def _get_api_key(self) -> str:
        env_api_key = os.getenv("DASHSCOPE_API_KEY")
        if env_api_key:
            return env_api_key
        model_cfg = self.config.get("model", {})
        api_key = model_cfg.get("api_key")
        if not api_key:
            raise ValueError(
                "Missing DashScope API key. Please set DASHSCOPE_API_KEY "
                "or provide `model.api_key` in config."
            )
        return api_key

    def get_chat_model(self) -> Optional["ChatTongyi"]:
        if not HAS_DASHSCOPE or not self._has_api_key():
            return None
        model_cfg = self.config.get("model", {})
        return ChatTongyi(
            model_name=model_cfg.get("chat_model", "qwen3-max"),
            dashscope_api_key=self._get_api_key(),
            temperature=model_cfg.get("temperature", 0.3),
            streaming=True,
        )

    def get_embedding_model(self) -> Optional[object]:
        if HAS_DASHSCOPE and self._has_api_key():
            model_cfg = self.config.get("model", {})
            return DashScopeEmbeddings(
                model=model_cfg.get("embedding_model", "text-embedding-v2"),
                dashscope_api_key=self._get_api_key(),
            )
        if HAS_HUGGINGFACE:
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return None