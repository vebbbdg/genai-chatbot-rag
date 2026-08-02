"""
大模型初始化模块
- LLM 对话模型（DeepSeek）
- Embedding 向量模型（用于RAG）
- 解耦设计，切换模型只需改这里
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)


def init_llm_model():
    """初始化LLM对话模型（流式输出）"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        api_key=api_key,
        base_url=base_url,
        streaming=True,
        temperature=0.7,
    )
    return model


def init_embeddings():
    """
    初始化Embedding向量模型
    国内网络环境默认使用FakeEmbeddings快速启动
    如需真实RAG语义检索：pip install sentence-transformers 并设置 USE_HF_EMBEDDING=true
    """
    # 如果没有显式开启真实embedding，直接用轻量方案
    if os.getenv("USE_HF_EMBEDDING", "false").lower() != "true":
        try:
            from langchain_community.embeddings import FakeEmbeddings
            print("ℹ️  RAG: Using FakeEmbeddings (fast startup). Set USE_HF_EMBEDDING=true for real semantic search.")
            return FakeEmbeddings(size=384)
        except Exception:
            return None

    # 尝试真实HuggingFace embeddings（需要网络）
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        # 设置超时避免长时间卡住
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "5"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("✅ RAG: Using real HuggingFace embeddings (all-MiniLM-L6-v2)")
        return embeddings
    except Exception as e:
        print(f"⚠️  HuggingFace embeddings failed ({e}), falling back to FakeEmbeddings")
        try:
            from langchain_community.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=384)
        except Exception:
            return None
