"""TTS inference parameter presets based on text length and position. / 基于文本长度和位置的 TTS 推理参数预设。"""

import os


def get_sovits_params(text: str, is_first_sentence: bool = False):
    """Return inference parameters tuned for the given text length. / 返回根据给定文本长度调整的推理参数。

    CUDA Graph is controlled by ENABLE_CUDA_GRAPH env var; static KV Cache is
    always enabled. / CUDA Graph 由 ENABLE_CUDA_GRAPH 环境变量控制；静态 KV Cache 始终启用。
    """
    length = len(text.strip())
    cuda_graph_env = os.environ.get("ENABLE_CUDA_GRAPH", "0") == "1"

    if is_first_sentence:
        max_sec_override = max(3.5, min(8.0, length * 0.25 or 3.5))
        return {
            "text_language": "日文",
            "prompt_language": "日文",
            "top_k": 5,
            "top_p": 1,
            "temperature": 0.6,
            "sample_steps": 8,
            "if_sr": False,
            "how_to_cut": "不切",
            "speed": 1.1,
            "pause_second": 0.1,
            "if_freeze": False,
            "enable_cuda_graph": cuda_graph_env,
            "enable_static_kv": True,
            "max_sec_override": max_sec_override,
        }

    if length < 45:
        return {
            "text_language": "日文",
            "prompt_language": "日文",
            "top_k": 5,
            "top_p": 1,
            "temperature": 0.6,
            "sample_steps": 16,
            "if_sr": False,
            "how_to_cut": "不切",
            "speed": 1,
            "pause_second": 0.2,
            "if_freeze": False,
            "enable_cuda_graph": cuda_graph_env,
            "enable_static_kv": True,
        }

    return {
        "text_language": "日文",
        "prompt_language": "日文",
        "top_k": 5,
        "top_p": 1,
        "temperature": 0.6,
        "sample_steps": 32,
        "if_sr": False,
        "how_to_cut": "凑四句一切",
        "speed": 1,
        "pause_second": 0.35,
        "if_freeze": False,
        "enable_cuda_graph": cuda_graph_env,
        "enable_static_kv": True,
    }
