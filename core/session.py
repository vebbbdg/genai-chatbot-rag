"""
多会话管理器
- 支持创建多个独立对话
- 每个会话有自己的消息历史
- 类似ChatGPT左侧会话列表功能
- 内存存储，重启服务会清空（生产环境可换Redis/SQLite）
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional


SYSTEM_PROMPT = (
    "You are a helpful, professional AI assistant. "
    "Answer user questions accurately and concisely. "
    "If you don't know the answer, say so honestly."
)


class SessionManager:
    """会话管理器：管理多个独立对话"""

    def __init__(self):
        # session_id -> {"messages": [...], "created_at": ..., "title": ...}
        self._sessions: Dict[str, Dict] = {}

    def create_session(self, title: str = "New Chat") -> str:
        """创建新会话，返回session_id"""
        session_id = str(uuid.uuid4())[:8]  # 取前8位，简短好记
        self._sessions[session_id] = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT}
            ],
            "created_at": datetime.now().isoformat(),
            "title": title,
        }
        return session_id

    def get_messages(self, session_id: str) -> List[Dict]:
        """获取指定会话的消息历史"""
        if session_id not in self._sessions:
            return []
        return self._sessions[session_id]["messages"]

    def add_message(self, session_id: str, role: str, content: str):
        """向指定会话添加消息"""
        if session_id not in self._sessions:
            return
        self._sessions[session_id]["messages"].append({
            "role": role,
            "content": content
        })
        # 自动用第一条用户消息作为会话标题
        if role == "user" and self._sessions[session_id]["title"] == "New Chat":
            self._sessions[session_id]["title"] = content[:30] + ("..." if len(content) > 30 else "")

    def reset_session(self, session_id: str):
        """重置指定会话"""
        if session_id in self._sessions:
            self._sessions[session_id]["messages"] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

    def delete_session(self, session_id: str):
        """删除指定会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> List[Dict]:
        """列出所有会话（用于左侧边栏）"""
        result = []
        for sid, data in self._sessions.items():
            result.append({
                "id": sid,
                "title": data["title"],
                "created_at": data["created_at"],
            })
        # 按创建时间倒序
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        return session_id in self._sessions


# 全局单例
session_manager = SessionManager()
