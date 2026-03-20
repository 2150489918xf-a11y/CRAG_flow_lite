"""
LLM Chat 客户端
支持 OpenAI 兼容协议的远程 LLM API（用于图谱提取和查询改写）
"""
import asyncio
import json
import logging
import re
from typing import Optional

from openai import OpenAI

from rag.llm.base import BaseChatClient
from rag.settings import get_config
from common.registry import chat_registry

logger = logging.getLogger(__name__)


@chat_registry.register("openai")
class ChatClient(BaseChatClient):
    """OpenAI 兼容协议的 Chat 客户端"""

    def __init__(self, api_key=None, model_name=None, base_url=None):
        cfg = get_config().get("llm", {})
        self.api_key = api_key or cfg.get("api_key", "")
        self.model_name = model_name or cfg.get("model_name", "")
        self.base_url = base_url or cfg.get("base_url", "https://api.openai.com/v1")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                             timeout=30.0)

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.1, max_tokens: int = 4096) -> str:
        """发送 chat 请求，返回文本响应（同步）"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            raise

    async def achat(self, system_prompt: str, user_prompt: str,
                    temperature: float = 0.1, max_tokens: int = 4096) -> str:
        """异步版 chat — 在线程池中执行，避免阻塞 FastAPI 事件循环"""
        return await asyncio.to_thread(
            self.chat, system_prompt, user_prompt, temperature, max_tokens
        )

    def chat_json(self, system_prompt: str, user_prompt: str,
                  temperature: float = 0.1, max_tokens: int = 4096) -> dict:
        """发送 chat 请求，解析 JSON 响应（同步）"""
        raw = self.chat(system_prompt, user_prompt, temperature, max_tokens)
        return self._parse_json(raw)

    async def achat_json(self, system_prompt: str, user_prompt: str,
                         temperature: float = 0.1, max_tokens: int = 4096) -> dict:
        """异步版 chat_json"""
        raw = await self.achat(system_prompt, user_prompt, temperature, max_tokens)
        return self._parse_json(raw)

    def _parse_json(self, raw: str) -> dict:
        """从 LLM 原始响应中解析 JSON"""
        # 尝试从 markdown code block 提取 JSON
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 尝试修复常见 JSON 错误
            try:
                import json_repair
                return json_repair.loads(raw)
            except Exception:
                pass
            logger.warning(f"Failed to parse LLM JSON response: {raw[:200]}")
            return {}
