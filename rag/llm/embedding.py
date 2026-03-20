"""
远程 Embedding 客户端
支持 OpenAI 兼容协议的远程 Embedding API
"""
import logging

import numpy as np
from openai import OpenAI

from rag.nlp import truncate
from rag.llm.base import BaseEmbedding

logger = logging.getLogger(__name__)


class RemoteEmbedding(BaseEmbedding):
    """
    OpenAI 兼容协议的 Embedding 客户端
    支持 SiliconFlow / OpenAI / 其他兼容 API
    """

    def __init__(self, api_key, model_name, base_url="https://api.openai.com/v1"):
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_length = 8191

    def encode(self, texts: list) -> tuple:
        """
        批量编码
        返回 (向量矩阵 np.ndarray, token消耗 int)
        """
        batch_size = 16
        texts = [truncate(t, self.max_length) for t in texts]
        # 确保没有空文本
        texts = [t if t and t.strip() else " " for t in texts]

        ress = []
        total_tokens = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                res = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name,
                    encoding_format="float",
                )
                ress.extend([d.embedding for d in res.data])
                total_tokens += res.usage.total_tokens if res.usage else 0
            except Exception as e:
                logger.error(f"Embedding encode error: {e}")
                raise

        return np.array(ress), total_tokens

    def encode_queries(self, text: str) -> tuple:
        """
        查询编码
        返回 (向量 np.ndarray, token消耗 int)
        """
        text = truncate(text, self.max_length)
        if not text or not text.strip():
            text = " "
        try:
            res = self.client.embeddings.create(
                input=[text],
                model=self.model_name,
                encoding_format="float",
            )
            return np.array(res.data[0].embedding), res.usage.total_tokens if res.usage else 0
        except Exception as e:
            logger.error(f"Embedding encode_queries error: {e}")
            raise
