import os
from typing import Optional, Dict, Any

class AudioTranscriber:
    """
    Whisper 转录工具封装，提供模型加载、转录与临时文件管理功能。
    """

    def __init__(self, model_path: Optional[str] = None, file_path: Optional[str] = None):
        """
        AudioTranscriber 的构造函数。

        :param model_path: Hugging Face 模型名或本地目录路径
        :param file_path: 待处理的音频文件路径（可选）
        """
        self.model_path = model_path
        self.file_path = file_path

    def set_file_path(self, file_path: str):
        """
        设置要处理的音频文件路径。

        :param file_path: 音频文件路径
        """
        self.file_path = file_path

    def transcribe(self, language: Optional[str] = None) -> dict:
        """
        对已保存的音频文件进行转录并返回结果字典，包含文本或错误信息。
        """
        if not self.file_path:
            raise ValueError("File path is not set. Use set_file_path() or provide it during initialization.")
        try:
            # 延迟导入以避免在服务启动时加载大型依赖（例如 scipy）
            import mlx_whisper

            path_or_hf_repo = self.model_path or ""
            if language:
                result = mlx_whisper.transcribe(self.file_path, path_or_hf_repo=path_or_hf_repo, language=language)
            else:
                result = mlx_whisper.transcribe(self.file_path, path_or_hf_repo=path_or_hf_repo)

            return {"text": result["text"]}
        except Exception as e:
            return {"error": f"Transcription failed: {str(e)}"}

    def delete_file(self):
        """
        删除临时保存的音频文件。
        """
        if self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)
            self.file_path = None