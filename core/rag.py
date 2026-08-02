"""
RAG (Retrieval-Augmented Generation) 检索增强生成模块
- 支持上传 PDF / TXT / DOCX 文档
- 文档切片 + 向量化存储到 ChromaDB
- 提问时检索相关文档片段，注入LLM上下文
- 解决大模型幻觉、知识过时问题
- 硅谷GenAI项目核心加分项
"""
import os
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document

from core.model import init_embeddings
from core.logger import logger


# 向量数据库持久化目录
VECTOR_DB_PATH = Path("data/vectordb")
UPLOAD_DIR = Path("data/uploads")
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class RAGEngine:
    """RAG检索引擎"""

    def __init__(self):
        self.embeddings = init_embeddings()
        self.vector_store = Chroma(
            persist_directory=str(VECTOR_DB_PATH),
            embedding_function=self.embeddings,
            collection_name="documents"
        )
        # 文档切片器：按字符数切分，有重叠
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,       # 每个片段1000字符
            chunk_overlap=200,     # 重叠200字符，保证上下文连贯
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        ext = Path(file_path).suffix.lower()
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
        }
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {list(loaders.keys())}")

        loader_cls = loaders[ext]
        loader = loader_cls(file_path)
        return loader.load()

    def add_document(self, file_path: str, filename: str) -> int:
        """
        添加文档到向量库
        返回切分后的片段数量
        """
        logger.info(f"Loading document: {filename}")
        docs = self._load_document(file_path)

        # 切片
        chunks = self.text_splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks")

        # 添加元数据
        for chunk in chunks:
            chunk.metadata["source"] = filename

        # 存入向量库
        self.vector_store.add_documents(chunks)
        logger.info(f"Document {filename} indexed successfully")
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """检索与问题最相关的top_k个文档片段"""
        results = self.vector_store.similarity_search(query, k=top_k)
        return results

    def format_context(self, query: str, top_k: int = 3) -> Optional[str]:
        """检索相关文档并格式化为上下文字符串"""
        docs = self.retrieve(query, top_k)
        if not docs:
            return None

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(f"[Document {i} from {source}]\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def get_document_count(self) -> int:
        """获取已索引的文档片段数量"""
        return self.vector_store._collection.count()

    def clear_all(self):
        """清空向量库"""
        self.vector_store._collection.delete(where={})
        logger.info("Vector store cleared")


# 全局RAG引擎单例
rag_engine = RAGEngine()
