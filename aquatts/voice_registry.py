# -*- coding: utf-8 -*-
"""Voice registry — name, reference audio, prompt text, and language.
/ 声音注册表 — 名称、参考音频、提示文本和语言。

Maps voice names to (ref_audio_path, prompt_text, language) so users
can say ``voice="alice"`` instead of passing file paths every time.
/ 将声音名称映射到 (ref_audio_path, prompt_text, language)，让用户可以使用
``voice="alice"`` 而不必每次都传递文件路径。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

logger = logging.getLogger("aquatts.voice_registry")


@dataclass(frozen=True)
class Voice:
    name: str
    ref_audio_path: str
    prompt_text: str = ""
    prompt_language: str = "日文"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Voice":
        return cls(
            name=d["name"],
            ref_audio_path=d["ref_audio_path"],
            prompt_text=d.get("prompt_text", ""),
            prompt_language=d.get("prompt_language", "日文"),
        )


class VoiceRegistry:
    """In-memory voice registry with optional JSON persistence. / 基于内存的声音注册表，支持可选的 JSON 持久化。"""

    def __init__(self, json_path: Optional[str] = None):
        self._voices: Dict[str, Voice] = {}
        self._json_path = json_path
        if json_path and os.path.isfile(json_path):
            self._load()

    # ── public API (公共 API) ──────────────────────────────────────────────────────

    def add(self, voice: Voice) -> None:
        self._voices[voice.name] = voice
        self._save()

    def remove(self, name: str) -> bool:
        existed = self._voices.pop(name, None) is not None
        if existed:
            self._save()
        return existed

    def get(self, name: str) -> Optional[Voice]:
        return self._voices.get(name)

    def list(self) -> Iterator[Voice]:
        yield from self._voices.values()

    def names(self) -> Iterable[str]:
        return self._voices.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._voices

    def __len__(self) -> int:
        return len(self._voices)

    # ── persistence (持久化) ─────────────────────────────────────────────────────

    def _save(self) -> None:
        if not self._json_path:
            return
        try:
            data = [v.to_dict() for v in self._voices.values()]
            os.makedirs(os.path.dirname(self._json_path) or ".", exist_ok=True)
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning(f"Failed to save voice registry: {exc}")

    def _load(self) -> None:
        try:
            with open(self._json_path, encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                voice = Voice.from_dict(entry)
                self._voices[voice.name] = voice
            logger.info(
                f"Loaded %d voice(s) from %s", len(self._voices), self._json_path
            )
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning(f"Failed to load voice registry: {exc}")


# ── built-in convenience (内置便捷函数) ──────────────────────────────────────────────────

def registry_from_env() -> VoiceRegistry:
    """Create a VoiceRegistry with path from AQUA_VOICE_JSON env var.
    / 从 AQUA_VOICE_JSON 环境变量获取路径并创建 VoiceRegistry。

    Falls back to ``./voices.json`` relative to the process CWD if the env
    var is not set.  The file is NOT created until the first ``add()`` call.
    / 如果未设置环境变量，则回退到相对于进程当前工作目录的 ``./voices.json``。
    文件在第一次调用 ``add()`` 之前不会被创建。
    """
    path = os.environ.get("AQUA_VOICE_JSON")
    if not path:
        path = os.path.join(os.getcwd(), "voices.json")
    return VoiceRegistry(json_path=str(Path(path)))
