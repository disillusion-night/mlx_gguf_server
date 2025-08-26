import os
import math


def get_model_size(model_path: str):
    model_size = 0
    if os.path.isfile(model_path):
        model_size = os.path.getsize(model_path)

    elif os.path.isdir(model_path):
        for file in os.scandir(model_path):
            if os.path.isfile(file):
                model_size += os.path.getsize(file)

    model_size_in_gb = model_size / (1024 * 1024 * 1024)
    model_size_in_gb = math.floor(model_size_in_gb)
    return model_size_in_gb


def create_model_list():
    models ={}
    for f in os.listdir("models"):
        if os.path.isdir(os.path.join("models", f)) or f.endswith(".gguf"):
            model_name = f
            model_path = os.path.join('models', model_name)
            model_size = get_model_size(model_path)
            models[model_name] = {'path': model_path, 'size': model_size }

    return models


def create_adapter_list():
    """
    扫描仓库下的 adapters/ 目录并返回 {adapter_name: {path, size}} 的映射。
    适配器可以是单文件（例如 .safetensors）或目录。
    """
    adapters = {}
    adapters_dir = "adapters"
    if not os.path.exists(adapters_dir):
        return adapters

    for f in os.listdir(adapters_dir):
        full = os.path.join(adapters_dir, f)
        if os.path.isdir(full) or os.path.isfile(full):
            adapter_name = f
            adapter_path = full
            adapter_size = get_model_size(adapter_path)
            adapters[adapter_name] = {"path": adapter_path, "size": adapter_size}

    return adapters


