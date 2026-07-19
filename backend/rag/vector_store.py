import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set

try:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    if TYPE_CHECKING:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document


class VectorStoreService:
    def __init__(self, config: dict):
        self.config = config
        self.store: Optional["Chroma"] = None

    def load_or_create_vectorstore(self) -> Optional["Chroma"]:
        if not HAS_LANGCHAIN:
            return None

        if self.store is not None:
            return self.store

        from model.factory import ModelFactory

        model_factory = ModelFactory(self.config)
        embeddings = model_factory.get_embedding_model()
        if embeddings is None:
            return None

        vector_db_path = self._get_vector_db_path()

        store = Chroma(
            collection_name="campusmind_docs",
            embedding_function=embeddings,
            persist_directory=str(vector_db_path) if vector_db_path else None,
        )

        documents: List["Document"] = []
        data_dir = self._get_data_dir()

        if data_dir and data_dir.exists():
            for file_path in data_dir.glob("*.*"):
                documents.extend(self._load_document(file_path))

        if documents:
            chunks = self._split_documents(documents)
            store.add_documents(chunks)

        self.store = store
        return store

    def _get_data_dir(self) -> Optional[Path]:
        base_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = base_dir / self.config["paths"].get("data_dir", "./data")
        return data_dir if data_dir.exists() else None

    def _get_vector_db_path(self) -> Optional[Path]:
        base_dir = Path(__file__).resolve().parent.parent.parent
        vector_db = base_dir / self.config["paths"].get("vector_db", "./vectorstore")
        return vector_db

    def _load_document(self, file_path: Path) -> List["Document"]:
        if not file_path.exists():
            return []

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            try:
                loader = PyPDFLoader(str(file_path))
                return loader.load()
            except Exception:
                return []
        elif ext == ".docx":
            try:
                loader = Docx2txtLoader(str(file_path))
                return loader.load()
            except Exception:
                return []
        return []

    def _split_documents(self, docs: List["Document"]) -> List["Document"]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["rag"].get("chunk_size", 500),
            chunk_overlap=self.config["rag"].get("chunk_overlap", 100),
        )
        return splitter.split_documents(docs)

    def similarity_search(self, query: str, k: int = 4) -> List["Document"]:
        if not HAS_LANGCHAIN:
            return []

        if self.store is None:
            self.load_or_create_vectorstore()

        if self.store is None:
            return []

        return self.store.similarity_search(query, k=k)

    def add_files(self, file_paths: List[Path]) -> int:
        if not HAS_LANGCHAIN:
            return 0

        if self.store is None:
            self.load_or_create_vectorstore()

        if self.store is None:
            return 0

        documents: List["Document"] = []
        for file_path in file_paths:
            if not file_path.exists():
                continue

            ext = file_path.suffix.lower()
            if ext not in [".pdf", ".docx"]:
                continue

            source = str(file_path)
            existing = self.store.get(where={"source": source}, limit=1)
            if existing.get("ids"):
                continue

            documents.extend(self._load_document(file_path))

        if not documents:
            return 0

        chunks = self._split_documents(documents)
        self.store.add_documents(chunks)
        return len(chunks)

    def get_stats(self) -> Dict[str, int]:
        if not HAS_LANGCHAIN:
            return {"doc_count": 0, "chunk_count": 0}

        if self.store is None:
            self.load_or_create_vectorstore()

        if self.store is None:
            return {"doc_count": 0, "chunk_count": 0}

        chunk_count = int(self.store._collection.count())

        payload = self.store.get(include=["metadatas"])
        metadatas = payload.get("metadatas", []) or []

        unique_sources: Set[str] = set()
        for metadata in metadatas:
            if isinstance(metadata, dict):
                source = metadata.get("source")
                if source:
                    unique_sources.add(str(source))

        return {"doc_count": len(unique_sources), "chunk_count": chunk_count}

    def delete_by_source(self, source: str) -> bool:
        if not HAS_LANGCHAIN or self.store is None:
            return False

        try:
            result = self.store.get(where={"source": source}, include=["ids"])
            ids = result.get("ids", [])
            if ids:
                self.store.delete(ids=ids)
                return True
            return False
        except Exception:
            return False