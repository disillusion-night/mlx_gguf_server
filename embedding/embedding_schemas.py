from dataclasses import dataclass
from typing import List

@dataclass
class OpenAICompatibleEmbedding:
    """
    响应主体的嵌入 API。由 OpenAI API 引用。
    Ref: https://platform.openai.com/docs/api-reference/embeddings/object
    """
    object: str
    embedding: List[float]
    index: int

@dataclass
class OpenAICompatibleEmbeddings:
    """
    OpenAICompatibleEmbedding 的列表。
    """
    embeddings: List[OpenAICompatibleEmbedding]

    def __iter__(self):
        return iter(self.embeddings)

    def __len__(self):
        return len(self.embeddings)