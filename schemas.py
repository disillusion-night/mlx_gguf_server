from pydantic import BaseModel, model_validator, Field
from typing_extensions import Self
from typing import Optional, Dict, List, Union, Literal

class CompletionParams(BaseModel):
    model: str = "dummy"
    prompt: str = ""
    messages: List[Dict] = []
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    stream: bool = False
    apply_chat_template: bool = False
    complete_text: bool = False
    top_p: Optional[float] = None
    # 通用 stop（用于 mlx 与 llama-cpp，不建议同时提供 prompt 与 messages）
    stop: Optional[list] = None
    logit_bias: Optional[Dict[int, float]] = None # 仅用于 mlx
    repetition_penalty: Optional[float] = None    # 仅用于 mlx
    repetition_context_size: Optional[int] = 20   # 仅用于 mlx
    use_kv_cache: bool = False          # 仅用于 mlx
    tools: Optional[list] = None        # 仅用于 mlx

    # llama-cpp 特有参数
    top_k: int = 40                     # 仅用于 llama-cpp
    min_p: float = 0.05                 # 仅用于 llama-cpp
    typical_p: float = 1.0              # 仅用于 llama-cpp
    frequency_penalty: float = 0.0      # 仅用于 llama-cpp
    presence_penalty: float = 0.0       # 仅用于 llama-cpp
    repet_penalty: float = 1.1          # 仅用于 llama-cpp
    mirostat_mode: int = 0              # 仅用于 llama-cpp
    mirostat_tau: float = 5.0           # 仅用于 llama-cpp
    mirostat_eta: float = 0.1           # 仅用于 llama-cpp

    chat_format: Optional[str] = None # 仅用于 llama-cpp

    @model_validator(mode='after')
    def validate_prompt_and_messages(self) -> Self:
        prompt = self.prompt
        messages = self.messages
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        return self

class TokenCountParams(BaseModel):
    model: str = "dummy"
    prompt: str = ""
    messages: list[dict] = []

    @model_validator(mode='after')
    def validate_prompt_and_messages(self) -> Self:
        prompt = self.prompt
        messages = self.messages
        if prompt and messages:
            raise ValueError("Only one of 'prompt' or 'messages' should be provided.")
        return self

class ModelLoadParams(BaseModel):
    llm_model_name: str
    llm_model_path: str = Field(default="", exclude=True)
    adapter_name: Optional[str] = None
    adapter_path: Optional[str] = Field(default=None, exclude=True)
    chat_format: Optional[str] = None # 仅用于 llama-cpp
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    logit_bias: Optional[Dict[int, float]] = None
    repetition_penalty: Optional[float] = None
    repetition_context_size: Optional[int] = None
    top_p: Optional[float] = None

class ProcessCleanParams(BaseModel):
    timeout: int

class KokoroTtsParams(BaseModel):
    text: str
    lang_code: str    = "a"
    voice: str        = "af_heart"
    speed: int        = 1
    split_pattern:str = r'\n+'


class EmbeddingsParams(BaseModel):
    """
    Parameters of Embedding API. Referred by OpenAI API.
    Refs:
        https://platform.openai.com/docs/api-reference/embeddings/create
    """
    input: Union[str, List[str]] = Field(
        ...,
        description="Input text to embed, encoded as a string or array of strings. "
                    "To embed multiple inputs in a single request, pass an array of strings. "
                    "The input must not exceed the max input tokens for the model."
    )
    encoding_format: Optional[str] = Field(
        default="float",
        description="The format to return the embeddings in. Can be either float or base64."
    )
    dimensions: Optional[Literal[32, 64, 128, 256, 512, 768, 1024]] = Field(
        default=None,
        description="The number of dimensions the resulting output embeddings should have."
    )
