from pathlib import Path
from typing import Dict, Generator, List, Optional

try:
    from langchain_core.documents import Document
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


class RAGService:
    def __init__(self, config: dict):
        self.config = config
        self.vector_store = None
        self.model_factory = None
        self._init_services()

    def _init_services(self):
        try:
            from rag.vector_store import VectorStoreService
            from model.factory import ModelFactory

            self.vector_store = VectorStoreService(config=self.config)
            self.vector_store.load_or_create_vectorstore()
            self.model_factory = ModelFactory(config=self.config)
        except Exception:
            pass

    def retrieve(self, query: str) -> str:
        if not HAS_LANGCHAIN or self.vector_store is None:
            return "知识检索服务未配置。"

        top_k = self.config["rag"].get("top_k", 4)
        docs = self.vector_store.similarity_search(query=query, k=top_k)
        if not docs:
            return "未检索到相关校园知识。"

        blocks = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            blocks.append(f"[文档 {idx}] 来源={source}\n{doc.page_content}")
        return "\n\n".join(blocks)

    def ingest_uploaded_files(self, file_paths: List[Path]) -> int:
        if not HAS_LANGCHAIN or self.vector_store is None:
            return 0
        return self.vector_store.add_files(file_paths)

    def get_vectorstore_stats(self) -> Dict[str, int]:
        if not HAS_LANGCHAIN or self.vector_store is None:
            return {"doc_count": 0, "chunk_count": 0}
        return self.vector_store.get_stats()

    def delete_document(self, source: str) -> bool:
        if not HAS_LANGCHAIN or self.vector_store is None:
            return False
        return self.vector_store.delete_by_source(source)

    def _get_prompt_template(self) -> str:
        base_dir = Path(__file__).resolve().parent.parent.parent
        prompts_dir = base_dir / self.config.get("paths", {}).get("prompts_dir", "./prompts")
        prompt_file = prompts_dir / "rag_prompt.txt"

        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")

        return """你是一个校园智能问答助手，请根据提供的参考资料回答用户的问题。

参考资料：
{context}

用户问题：
{question}

请基于上述参考资料，用自然、友好的语言回答用户的问题。如果参考资料中没有相关信息，请明确说明。回答要简洁明了。"""

    def rag_summarize(self, query: str) -> str:
        if not HAS_LANGCHAIN or self.vector_store is None or self.model_factory is None:
            return "RAG服务未完全配置。请检查API密钥和依赖包是否正确安装。"

        top_k = self.config.get("rag", {}).get("top_k", 4)
        docs = self.vector_store.similarity_search(query=query, k=top_k)

        if not docs:
            return "未检索到相关资料，无法生成基于知识库的回答。"

        context_blocks: List[str] = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            context_blocks.append(f"[文档 {idx}] 来源={source}\n{doc.page_content}")
        context = "\n\n".join(context_blocks)

        prompt_template = self._get_prompt_template()
        if "{context}" in prompt_template and "{question}" in prompt_template:
            full_prompt = prompt_template.format(context=context, question=query)
        else:
            full_prompt = (
                f"{prompt_template}\n\n"
                f"参考资料：\n{context}\n\n"
                f"用户问题：\n{query}"
            )

        chat_model = self.model_factory.get_chat_model()
        if chat_model is None:
            return "聊天模型未配置，请检查API密钥。"

        try:
            result = chat_model.invoke(full_prompt)
            return str(getattr(result, "content", result))
        except Exception as e:
            return f"生成回答时出错: {str(e)}"

    def rag_summarize_stream(self, query: str) -> Generator[str, None, None]:
        if not HAS_LANGCHAIN or self.vector_store is None or self.model_factory is None:
            yield "RAG服务未完全配置。请检查API密钥和依赖包是否正确安装。"
            return

        top_k = self.config.get("rag", {}).get("top_k", 4)
        docs = self.vector_store.similarity_search(query=query, k=top_k)

        if not docs:
            yield "未检索到相关资料，无法生成基于知识库的回答。"
            return

        context_blocks: List[str] = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            context_blocks.append(f"[文档 {idx}] 来源={source}\n{doc.page_content}")
        context = "\n\n".join(context_blocks)

        prompt_template = self._get_prompt_template()
        if "{context}" in prompt_template and "{question}" in prompt_template:
            full_prompt = prompt_template.format(context=context, question=query)
        else:
            full_prompt = (
                f"{prompt_template}\n\n"
                f"参考资料：\n{context}\n\n"
                f"用户问题：\n{query}"
            )

        try:
            chat_model = self.model_factory.get_chat_model()
        except Exception as e:
            yield f"模型初始化失败: {str(e)}"
            return

        if chat_model is None:
            yield "聊天模型未配置，请检查API密钥。"
            return

        try:
            for chunk in chat_model.stream(full_prompt):
                content = getattr(chunk, "content", chunk)
                if content:
                    yield str(content)
        except Exception as e:
            yield f"生成回答时出错: {str(e)}"