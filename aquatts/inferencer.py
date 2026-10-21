# -*- coding: utf-8 -*-
import os
import time
import threading
import logging
import traceback
import inspect
from pathlib import Path
from contextlib import nullcontext
from string import punctuation

import torch
import numpy as np
from aquatts.inference.onset import LeadingSilence, leading_cut
from aquatts.inference.workspace import resolve_kv_cache_layout
from aquatts.inference.cuda_runtime import resolve_cuda_device, resolve_vocoder_warmup, warmup_vocoder
from aquatts.inference.reference_cache import REFERENCE_LANGUAGE_CODES, ReferenceCache, normalize_reference, reference_key
from aquatts.inference.offline import check_frontend, require_checkpoint, require_files, require_transformer
from aquatts.voice_registry import Voice

from aquatts.inference.checkpoints import PRO_MODELS, detect_sovits_version
from aquatts.inference.semantic_stability import (
    SemanticGenerationError,
    assess_semantic_candidate,
)
from aquatts.upstream import (
    configured_gpt_sovits_home,
    upstream_t2s_model_path,
)

try:
    import soundfile as sf
except ImportError:
    sf = None  # friendly error in TTSInferencer.__init__

# TTS language config — set via environment variables.
# TTS 语言配置 — 通过环境变量控制。
_TTS_OUTPUT_LANGUAGE = os.environ.get("TTS_OUTPUT_LANGUAGE", "日文")
_TTS_REF_TEXT_JA = os.environ.get("TTS_REF_TEXT_JA", "こんにちは。今日はいい天気ですね。")
_TTS_REF_TEXT_EN = os.environ.get("TTS_REF_TEXT_EN", "")

def _default_lang_code() -> str:
    """返回当前语言对应的 GPT-SoVITS 内部语言代码（用于 fallback）。"""
    return "en" if _TTS_OUTPUT_LANGUAGE == "英文" else "ja"

def _default_ref_free_prompt() -> str:
    """v3 ref_free 兜底 prompt 文本。"""
    if _TTS_OUTPUT_LANGUAGE == "英文":
        return _TTS_REF_TEXT_EN or "I see, let me think about that."
    return _TTS_REF_TEXT_JA or "こんにちは。今日はいい天気ですね。"


def _env_bool(*names: str, default: bool = False) -> bool:
    """Read the first defined boolean environment variable from names."""
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        normalized = value.strip().lower()
        if normalized in {"", "auto"}:
            continue
        return normalized in {"1", "true", "yes", "on"}
    return default


# Set up logging
# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tts_inference')

# Package directory for vendored-file references.
# sys.path is configured by Aqua/__init__.py before this module loads.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

def _gpt_sovits_home() -> str:
    """Return the main GPT-SoVITS repo root, or '' if not configured."""
    home = configured_gpt_sovits_home()
    return str(home) if home is not None else ""


# Import required modules from GPT-SoVITS
# 导入必要的模块
try:
    import librosa
    import re
    try:
        from tools.i18n import I18nAuto
    except ImportError:
        from tools.i18n.i18n import I18nAuto
    from GPT_SoVITS.text.LangSegmenter import LangSegmenter
    from GPT_SoVITS.text import cleaned_text_to_sequence
    from GPT_SoVITS.text.cleaner import clean_text
    from GPT_SoVITS.module.mel_processing import spectrogram_torch, mel_spectrogram_torch
    from GPT_SoVITS.text import chinese
    from GPT_SoVITS.feature_extractor import cnhubert
    from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule

    """Load SoVITS model
加载SoVITS模型"""
    from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3
    from GPT_SoVITS.process_ckpt import load_sovits_new
    from peft import LoraConfig, get_peft_model

    from GPT_SoVITS.BigVGAN import bigvgan

    from aquatts.inference.presets import (
        resolve_cuda_graph_config as _resolve_cg_config,
        apply_preset as _apply_preset,
    )

except ImportError as e:
    _missing = str(e)
    if "librosa" in _missing:
        logger.error("librosa is required. Install with: pip install Aqua-TTS[runtime]")
    elif "peft" in _missing:
        logger.error("peft is required. Install with: pip install Aqua-TTS[runtime]")
    elif "GPT_SoVITS" in _missing:
        logger.error(
            "GPT-SoVITS not found. Set GPT_SOVITS_HOME to your GPT-SoVITS repo root."
        )
    else:
        logger.error(f"Import failed: {_missing}")
    raise


_V3_SPEC_MIN = -12
_V3_SPEC_MAX = 2


def _v3_mel_spectrogram(audio):
    return mel_spectrogram_torch(
        audio,
        n_fft=1024,
        win_size=1024,
        hop_size=256,
        num_mels=100,
        sampling_rate=24000,
        fmin=0,
        fmax=None,
        center=False,
    )


def _normalize_v3_spec(spec):
    return (spec - _V3_SPEC_MIN) / (_V3_SPEC_MAX - _V3_SPEC_MIN) * 2 - 1


def _denormalize_v3_spec(spec):
    return (spec + 1) / 2 * (_V3_SPEC_MAX - _V3_SPEC_MIN) + _V3_SPEC_MIN


class TTSInferencer:
    def __init__(self,
                 device="cuda",
                 gpt_path=None,
                 sovits_path=None,
                 bert_path=None,
                 cnhubert_path=None,
                 sovits_pretrain_path=None,
                 bigvgan_path=None,
                 fast_langdetect_path=None,
                 language="Auto",
                 cuda_graph_preset="full",
                 use_flash_attn=None,
                 flash_attn_mode=None,
                 sv_model_path=None,
                 kv_cache_layout=None,
                 preload_bert=False,
                 warmup_references=None,
                 warmup_strict=True,
                 bigvgan_warmup_shapes=None,
                 bigvgan_warmup_repeats=None,
                 offline_only=False,
                 cfm_unpadded=False):
        """
        Initialize the TTS inferencer.
        初始化TTS推理器

        Args:
            device: Inference device, default "cuda"
                    推理设备，默认为"cuda"
            gpt_path: Path to GPT T2S checkpoint (.ckpt)
                      GPT模型路径，如果为None则使用默认路径
            sovits_path: Path to SoVITS checkpoint (.pth)
                         SoVITS模型路径，如果为None则使用默认路径
            bert_path: Path to BERT model directory
                       BERT模型路径，如果为None则使用默认路径
            cnhubert_path: Path to CNHuBERT model directory
                            CNHuBERT模型路径，如果为None则使用默认路径
            sovits_pretrain_path: Path to the v3 SoVITS base checkpoint.
            bigvgan_path: Path to the local BigVGAN model directory.
            fast_langdetect_path: Directory containing the cached lid.176.bin.
            sv_model_path: ERes2Net checkpoint required only by v2Pro/v2ProPlus.
            language: Default language — "Auto", "中文", "英文", "日文", etc.
                      默认语言
            cuda_graph_preset: CUDA Graph capture strategy — "full" (all buckets pre-captured),
                "minimal" (subset), "lazy" (capture on first use), "off" (static KV only)
                CUDA Graph 捕获策略 — "full" (全桶预捕获), "minimal" (最小桶集合),
                "lazy" (惰性捕获), "off" (纯 static KV)
            use_flash_attn: Optional FlashAttention2 KV-cache decode path for T2S.
                None means read AQUATTS_T2S_FLASH_ATTN / TTS_T2S_FLASH_ATTN,
                then auto-enable when FlashAttention2 is available.
            flash_attn_mode: "valid" (fused KV update) or "bucket" (explicit
                scatter); both attend only the true contiguous KV prefix.
            bigvgan_warmup_shapes: Optional positive mel lengths; [] disables warmup.
                None reads AQUATTS_BIGVGAN_WARMUP_SHAPES (default 40,70,128).
            bigvgan_warmup_repeats: Positive repeat count; None reads
                AQUATTS_BIGVGAN_WARMUP_REPEATS (default 5). Explicit arguments win.
            warmup_references: Optional Voice/reference dictionaries to prepare before readiness.
                Only supplied references are warmed; unique entries must fit the cache capacity.
            warmup_strict: Raise on warmup failure (default); False reports best-effort results.
            preload_bert: Load Chinese BERT during construction instead of first use.
                English/Japanese-only requests otherwise leave BERT unloaded.
            offline_only: Require local model resources without changing global Hub policy.
                Frontend checks cover en/ja/zh with the documented upstream resource layout.
            cfm_unpadded: Experimental V3 single-item unpadded CFM path; default False.
            kv_cache_layout: "bucket" (independent bucket storage) or "nested"
                (shared maximum root); None reads AQUATTS_KV_CACHE_LAYOUT once.
        """
        if not isinstance(offline_only, bool):
            raise TypeError("offline_only must be a bool")
        self.offline_only = offline_only
        if not isinstance(cfm_unpadded, bool):
            raise TypeError("cfm_unpadded must be a bool")
        self.cfm_unpadded = cfm_unpadded
        self._cfm_adapter = None
        self._segmenter = None
        self._bigvgan_warmup_shapes, self._bigvgan_warmup_repeats = resolve_vocoder_warmup(
            bigvgan_warmup_shapes, bigvgan_warmup_repeats)
        self.vocoder_warmup_stats = {"status": "not_applicable"}
        self.vocoder_shape_counts = {}
        self.bigvgan_kernel_mode = None
        self.bigvgan_kernel_fallback_reason = None
        self.tokenizer = None
        self.bert_model = None
        self.bert_load_seconds = None
        self._bert_load_lock = threading.Lock()
        self.sr_model = None
        self._sr_load_lock = threading.Lock()
        self._init_reference_cache()
        self._cuda_graph_preset = cuda_graph_preset
        self._cuda_graph_config = _resolve_cg_config(cuda_graph_preset)
        self._kv_cache_layout = resolve_kv_cache_layout(kv_cache_layout)
        self._use_flash_attn = use_flash_attn
        self._flash_attn_mode = (
            flash_attn_mode
            or os.environ.get(
                "AQUATTS_T2S_FLASH_ATTN_MODE",
                os.environ.get("TTS_T2S_FLASH_ATTN_MODE", "valid"),
            )
        ).strip().lower()
        try:
            # 确定模型路径
            base_dir = _gpt_sovits_home() or _PACKAGE_DIR
            self.device = resolve_cuda_device(device)
            self.is_half = self.device.type == "cuda"
            self._tts_device_idx = self.device.index if self.is_half else None

            # 默认模型路径
            default_gpt_path = os.path.join(base_dir, "GPT_weights_v3", "xxx-e15.ckpt")
            default_sovits_path = os.path.join(base_dir, "SoVITS_weights_v3", "xxx_e2_s174_l32.pth")
            default_sovits_pretrain_path = os.path.join(base_dir, "GPT_SoVITS", "pretrained_models", "s2Gv3.pth")
            default_bert_path = os.path.join(base_dir, "GPT_SoVITS", "pretrained_models",
                                             "chinese-roberta-wwm-ext-large")
            default_cnhubert_path = os.path.join(base_dir, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base")
            default_bigvgan_path = os.path.join(
                base_dir,
                "GPT_SoVITS",
                "pretrained_models",
                "models--nvidia--bigvgan_v2_24khz_100band_256x",
            )
            default_fast_langdetect_path = os.path.join(
                base_dir, "GPT_SoVITS", "pretrained_models", "fast_langdetect"
            )

            # 使用传入参数或默认路径
            self.gpt_path = gpt_path or default_gpt_path
            self.sovits_path = sovits_path or default_sovits_path
            self.bert_path = bert_path or default_bert_path
            self.cnhubert_path = cnhubert_path or default_cnhubert_path
            self.sovits_pretrain_path = sovits_pretrain_path or default_sovits_pretrain_path
            self.bigvgan_path = bigvgan_path or default_bigvgan_path
            self.sv_model_path = sv_model_path or os.path.join(
                base_dir, "GPT_SoVITS", "pretrained_models", "sv",
                "pretrained_eres2netv2w24s4ep4.ckpt",
            )
            self.sv_model = None
            self.bigvgan_model = None
            self.fast_langdetect_path = (
                fast_langdetect_path or default_fast_langdetect_path
            )

            # 检查必要文件是否存在
            for path, desc in [
                (self.gpt_path, "GPT权重"),
                (self.sovits_path, "SoVITS权重"),
                (self.cnhubert_path, "CNHuBERT模型"),
                (self.fast_langdetect_path, "fast-langdetect模型"),
            ]:
                if not os.path.exists(path):
                    logger.warning(f"必要文件不存在: {path} ({desc})，请确保路径正确")

            self._configure_fast_langdetect()
            if self.offline_only:
                require_checkpoint("GPT", self.gpt_path)
                require_checkpoint("SoVITS", self.sovits_path)

            # 初始化国际化
            self.i18n = I18nAuto(language=language)

            # 初始化语言字典
            self._init_language_dict()

            # Chinese features load BERT on first use; explicit preload is optional.
            if preload_bert:
                self.preload_bert_model()
            self._init_ssl_model()

            # 加载GPT和SoVITS模型
            self._load_models()

            # 仅在需要做性能剖析时开启；默认关闭以避免流式首句每块都强制同步 GPU。
            self._stream_sync_timing_enabled = os.environ.get("TTS_STREAM_SYNC_TIMING", "0") == "1"
            self.t2s_stats = []
            if warmup_references is not None:
                self.reference_warmup_results = self._warm_reference_list(warmup_references, strict=warmup_strict)
            logger.info("⭐️ TTS推理器初始化完成")

        except Exception as e:
            logger.error(f"❌ 初始化TTS推理器失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _configure_fast_langdetect(self):
        """Bind this instance to its cache without changing the global detector."""

        import fast_langdetect

        infer = fast_langdetect.infer
        from aquatts.inference.language_detector import bind_segmenter
        config = {"cache_dir": Path(self.fast_langdetect_path)}
        if self.offline_only:
            root = require_files("fast-langdetect", self.fast_langdetect_path, ["lid.176.bin"])
            # custom_model_path is a supported local-only API: even a later
            # deletion fails locally rather than entering load_with_download.
            config["custom_model_path"] = str(root / "lid.176.bin")
        self._language_detector = infer.LangDetector(infer.LangDetectConfig(**config))
        self._segmenter = bind_segmenter(LangSegmenter.getTexts, self._language_detector)

    def _segment_text(self, text):
        return (getattr(self, "_segmenter", None) or LangSegmenter.getTexts)(text)

    def _get_effective_max_sec(self, override_value):
        """Compute the effective max duration (seconds) for current inference.
        计算当前推理应当使用的最大时长限制（秒）"""
        effective = float(self.max_sec)
        if override_value is not None:
            try:
                override = float(override_value)
                if override > 0:
                    effective = min(effective, override)
            except (TypeError, ValueError):
                pass
        return effective

    def _sync_t2s_timing(self):
        if self.is_half and str(self.device).lower().startswith("cuda") and torch.cuda.is_available():
            with torch.cuda.device(self._tts_device_idx):
                torch.cuda.synchronize(self.device)

    def _record_t2s_stat(self, text_item: str, tokens: int, elapsed_sec: float):
        rate = float(tokens) / elapsed_sec if elapsed_sec > 0 else 0.0
        stat = {
            "text": text_item,
            "tokens": int(tokens),
            "elapsed_sec": float(elapsed_sec),
            "tokens_per_sec": rate,
        }
        self.t2s_stats.append(stat)
        logger.info(
            f"[t2s] tokens={stat['tokens']} elapsed={stat['elapsed_sec']:.3f}s "
            f"speed={stat['tokens_per_sec']:.0f} it/s"
        )
        return stat

    @staticmethod
    def _semantic_guard_enabled() -> bool:
        raw = os.environ.get(
            "AQUATTS_SEMANTIC_GUARD",
            os.environ.get("TTS_SEMANTIC_GUARD", "1"),
        )
        return raw.strip().lower() not in {"0", "false", "off", "no"}

    def _infer_semantic_with_guard(
        self,
        *,
        text_item: str,
        target_phone_count: int,
        all_phoneme_ids,
        all_phoneme_len,
        prompt,
        bert,
        top_k,
        top_p,
        temperature,
        effective_max_sec: float,
        enable_cuda_graph: bool,
        enable_static_kv: bool,
    ):
        """Generate one admissible semantic candidate before vocoder work."""

        max_generation_tokens = int(self.hz * effective_max_sec)
        guard_enabled = self._semantic_guard_enabled()
        penalties = (1.35, 1.5) if guard_enabled else (1.35,)
        last_assessment = None

        for attempt, repetition_penalty in enumerate(penalties, start=1):
            prediction, idx = self.t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                early_stop_num=max_generation_tokens,
                enable_cuda_graph=enable_cuda_graph,
                enable_static_kv=enable_static_kv,
            )
            reported_count = max(0, int(idx))
            # Upstream uses idx=0 as a ref-free sentinel even though
            # ``prediction`` contains the complete generated sequence.
            generated_count = (
                reported_count
                if reported_count
                else int(prediction.shape[-1])
            )
            candidate = (
                prediction[:, -generated_count:]
                if generated_count
                else prediction[:, :0]
            )

            if not guard_enabled:
                return candidate.unsqueeze(0), generated_count, attempt

            assessment = assess_semantic_candidate(
                candidate.detach().reshape(-1).cpu().long().tolist(),
                target_phone_count=target_phone_count,
                max_generation_tokens=max_generation_tokens,
            )
            last_assessment = assessment
            if assessment.accepted:
                if attempt > 1:
                    logger.info(
                        "[semantic-guard] retry recovered text=%r tokens=%d "
                        "penalty=%.2f",
                        text_item[:80],
                        assessment.token_count,
                        repetition_penalty,
                    )
                return candidate.unsqueeze(0), generated_count, attempt

            logger.warning(
                "[semantic-guard] rejected attempt=%d/%d text=%r reasons=%s "
                "tokens=%d budget=%d equal_run=%d periodic_run=%d "
                "top_ratio=%.3f",
                attempt,
                len(penalties),
                text_item[:80],
                ",".join(assessment.reasons),
                assessment.token_count,
                assessment.soft_token_budget,
                assessment.longest_equal_run,
                assessment.longest_periodic_run,
                assessment.top_token_ratio,
            )

        detail = "unknown"
        if last_assessment is not None:
            detail = ",".join(last_assessment.reasons) or "rejected"
        raise SemanticGenerationError(
            f"semantic generation collapsed after {len(penalties)} attempts: {detail}"
        )

    def _init_language_dict(self):
        """Initialize language dictionary mappings.
        初始化语言字典"""
        # Detect model version to select the correct language dictionary
        # 检测模型版本，根据版本确定语言字典
        self.model_version = self._detect_model_version()

        dict_language_v1 = {
            "中文": "all_zh",
            "英文": "en",
            "日文": "all_ja",
            "中英混合": "zh",
            "日英混合": "ja",
            "多语种混合": "auto",
        }

        dict_language_v2 = {
            "中文": "all_zh",
            "英文": "en",
            "日文": "all_ja",
            "粤语": "all_yue",
            "韩文": "all_ko",
            "中英混合": "zh",
            "日英混合": "ja",
            "粤英混合": "yue",
            "韩英混合": "ko",
            "多语种混合": "auto",
            "多语种混合(粤语)": "auto_yue",
        }

        labels = dict_language_v1 if self.sovits_version == "v1" else dict_language_v2
        # Keep the documented API labels stable when the host UI uses another locale.
        self.dict_language = {**labels, **{self.i18n(key): value for key, value in labels.items()}}
        self.splits = {"、", "，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}

    def _detect_model_version(self):
        """Read the architecture separately from the text-symbol version."""
        self.sovits_version, model, self._detected_lora = detect_sovits_version(self.sovits_path)
        self.is_v2pro = model in PRO_MODELS
        return model

    @property
    def bert_loaded(self):
        """Whether both Chinese tokenizer and model are ready on this instance."""
        return self.tokenizer is not None and self.bert_model is not None

    def _ensure_bert_model(self):
        if self.bert_loaded:
            return
        with self._bert_load_lock:
            if not self.bert_loaded:
                self._init_bert_model()

    def preload_bert_model(self):
        """Load Chinese features now; return the original successful load duration."""
        self._ensure_bert_model()
        return self.bert_load_seconds

    def _init_bert_model(self):
        """Build locally, then publish the complete tokenizer/model pair."""
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        started = time.perf_counter()
        logger.info("加载BERT模型: %s", self.bert_path)
        options = {}
        if getattr(self, "offline_only", False):
            require_transformer("BERT", self.bert_path, tokenizer=True)
            options["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(self.bert_path, **options)
        model = AutoModelForMaskedLM.from_pretrained(self.bert_path, **options)
        if self.is_half:
            model = model.half()
        model = model.to(self.device).eval()
        self.bert_load_seconds = time.perf_counter() - started
        self.tokenizer, self.bert_model = tokenizer, model
        logger.info("BERT ready in %.3fs", self.bert_load_seconds)

    def _init_ssl_model(self):
        """Initialize CNHuBERT SSL model for semantic feature extraction.
        初始化SSL模型"""


        logger.info(f"加载CNHubert模型: {self.cnhubert_path}")
        if getattr(self, "offline_only", False):
            require_transformer("CNHuBERT", self.cnhubert_path, processor=True)
        self.ssl_model = cnhubert.CNHubert(self.cnhubert_path).eval()

        if self.is_half:
            self.ssl_model = self.ssl_model.half().to(self.device)
        else:
            self.ssl_model = self.ssl_model.to(self.device)

    def _load_models(self):
        """Load GPT, SoVITS, and BigVGAN models.
        加载GPT和SoVITS模型"""
        # 加载GPT模型
        self._load_gpt_model()

        # 加载SoVITS模型
        self._load_sovits_model()

        # 如果是v3模型，还需加载BigVGAN
        if self.model_version == "v3":
            self._load_bigvgan_model()
        if self.cfm_unpadded:
            self._prepare_cfm_adapter()

    def _load_gpt_model(self):
        """Load GPT T2S model and pre-capture CUDA Graphs.
        加载GPT模型"""

        logger.info(f"加载GPT模型: {self.gpt_path}")
        dict_s1 = torch.load(self.gpt_path, map_location="cpu")
        self.gpt_config = dict_s1["config"]
        self.hz = 50  # 默认值
        self.max_sec = self.gpt_config["data"]["max_sec"]

        self.t2s_model = Text2SemanticLightningModule(self.gpt_config, "****", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])

        if self.is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()

        self._apply_aqua_t2s_patch()
        self._maybe_enable_flash_attn_t2s()
        self._maybe_precapture_t2s_graph()
        if hasattr(self, "_reference_cache"):
            self.clear_reference_cache()

    def _apply_aqua_t2s_patch(self):
        """Patch the selected upstream decoder without replacing its module."""

        decoder = getattr(self.t2s_model, "model", None)
        if decoder is None:
            raise RuntimeError("GPT-SoVITS checkpoint did not expose a T2S decoder")

        selected = upstream_t2s_model_path()
        source = Path(inspect.getfile(decoder.__class__)).resolve()
        if selected is not None and source != selected.resolve():
            raise RuntimeError(
                "Aqua loaded a different GPT-SoVITS T2S module than the configured "
                f"upstream checkout: loaded={source}, expected={selected}"
            )

        from aquatts.modeling.t2s_streaming import apply_cuda_graph_patch

        apply_cuda_graph_patch(decoder, graph_config=self._cuda_graph_config,
                               kv_cache_layout=self._kv_cache_layout)
        logger.info("[Aqua] patched upstream GPT-SoVITS decoder: %s", source)

    def _warmup_bigvgan_shapes(self):
        self.vocoder_warmup_stats = warmup_vocoder(
            self.bigvgan_model, self._bigvgan_warmup_shapes, self._bigvgan_warmup_repeats)
        if self.vocoder_warmup_stats["status"] == "failed":
            logger.warning("BigVGAN warmup failed: %s", self.vocoder_warmup_stats["error"])
        else:
            logger.info("BigVGAN warmup: %s", self.vocoder_warmup_stats)

    def _run_bigvgan(self, mel):
        frames = int(mel.shape[-1])
        self.vocoder_shape_counts[frames] = self.vocoder_shape_counts.get(frames, 0) + 1
        context = torch.cuda.device(mel.device) if mel.is_cuda else nullcontext()
        with context, torch.inference_mode():
            return self.bigvgan_model(mel)

    def _maybe_enable_flash_attn_t2s(self):
        """Prefer FlashAttention2 KV-cache decoding before graph capture."""
        env_names = ("AQUATTS_T2S_FLASH_ATTN", "TTS_T2S_FLASH_ATTN")
        explicitly_configured = self._use_flash_attn is not None or any(
            os.environ.get(name, "").strip().lower() not in {"", "auto"}
            for name in env_names
        )
        enabled = (
            bool(self._use_flash_attn)
            if self._use_flash_attn is not None
            else _env_bool(*env_names, default=True)
        )
        if not enabled:
            return

        decoder = getattr(self.t2s_model, "model", None)
        if decoder is None:
            return

        try:
            from aquatts.modeling.t2s_flash_attn import (
                apply_flash_attn_patch,
                get_flash_attn_error,
                is_flash_attn_available,
            )

            if not is_flash_attn_available():
                log = logger.warning if explicitly_configured else logger.info
                log(
                    "T2S FlashAttention2 unavailable; using SDPA path: %s",
                    get_flash_attn_error(),
                )
                return

            if apply_flash_attn_patch(decoder, mode=self._flash_attn_mode):
                logger.info(
                    "[T2S FlashAttn] enabled flash_attn_with_kvcache "
                    "(mode=%s)",
                    getattr(decoder, "flash_attn_kvcache_mode", self._flash_attn_mode),
                )
        except Exception as exc:
            logger.warning("T2S FlashAttention2 setup failed; using SDPA path: %s", exc)

    def _maybe_precapture_t2s_graph(self):
        """根据 CUDA Graph preset 预捕获 T2S 阶段的 CUDA Graph。

        环境变量（ENABLE_CUDA_GRAPH, ENABLE_CUDA_GRAPH_PRECAPTURE,
        CUDA_GRAPH_PRECAPTURE_BUCKETS）可作为 preset 的 override。
        """
        try:
            # Resolve CUDA Graph preset (env vars override the named preset)
            cg = self._cuda_graph_config

            if not cg["enable"]:
                logger.info("[precapture] CUDA Graph disabled by preset/env")
                return

            decoder = getattr(self.t2s_model, "model", None)
            if decoder is None:
                return

            can_use_graph = (getattr(decoder, "cuda_graph_enabled", False)
                             or getattr(decoder, "use_static_kv_cache", False))
            if not can_use_graph:
                return

            available_buckets = list(getattr(decoder, "kv_cache_buckets", []) or [])
            logger.info(f"[precapture] preset={self._cuda_graph_preset!r} "
                        f"available={available_buckets} cg={cg}")
            if not available_buckets:
                return

            if not cg["precapture"]:
                logger.info("[precapture] lazy mode — skipping pre-capture")
                return

            # Select buckets — use preset's list or default to all available
            buckets = cg.get("buckets") or list(available_buckets)
            buckets = [b for b in buckets if b in available_buckets]
            if not buckets:
                return

            logger.info(f"[precapture] starting — target_buckets={buckets}")
            results = decoder.precapture_cuda_graph(buckets)
            total_graphs = len(decoder.bucket_graphs)
            logger.info(f"[precapture] done — buckets_ok={results} total_graphs={total_graphs} "
                        f"keys={sorted(decoder.bucket_graphs.keys())}")
        except Exception as exc:
            logger.warning(f"CUDA Graph 预捕获失败: {exc}")

    def _load_sovits_model(self):


        # 加载SoVITS配置
        dict_s2 = load_sovits_new(self.sovits_path)
        self.hps = DictToAttrRecursive(dict_s2["config"])
        self.hps.model.semantic_frame_rate = "25hz"

        # 确定SoVITS版本
        if self.is_v2pro:
            self.hps.model.version = self.model_version
        elif 'enc_p.text_embedding.weight' not in dict_s2['weight']:
            self.hps.model.version = "v2"  # v3model,v2symbols
        elif dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            self.hps.model.version = "v1"
        else:
            self.hps.model.version = "v2"

        self.sovits_version = "v2" if self.is_v2pro else self.hps.model.version
        logger.info(f"SoVITS版本: {self.sovits_version}, 模型版本: {self.model_version}")

        # 根据模型版本创建模型
        if self.model_version != "v3":
            self.vq_model = SynthesizerTrn(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model
            )
        else:
            # v3 使用 v2 文字符号，但声学模型必须保留 v3 架构版本。
            self.hps.model.version = self.model_version
            self.vq_model = SynthesizerTrnV3(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model
            )

        if self.is_v2pro:
            if "sv_emb" not in inspect.signature(self.vq_model.decode).parameters or not all(
                hasattr(self.vq_model, name) for name in ("sv_emb", "ge_to512", "prelu")
            ):
                raise RuntimeError(
                    "Selected GPT-SoVITS checkout lacks the v2Pro/v2ProPlus decoder; "
                    "update GPT_SOVITS_HOME to a compatible upstream checkout"
                )

        # 处理预训练模型
        if "pretrained" not in self.sovits_path:
            try:
                del self.vq_model.enc_q
            except AttributeError:
                pass

        # 转换模型类型并加载到设备
        if self.is_half:
            self.vq_model = self.vq_model.half().to(self.device)
        else:
            self.vq_model = self.vq_model.to(self.device)

        self.vq_model.eval()

        # 检查是否是LoRA模型 — 通过 checkpoint key 名称判断，比文件大小可靠。
        # LoRA checkpoint 含 "lora_rank" 顶层键，或 weight keys 中含 "lora_" 前缀。
        self.if_lora_v3 = self.model_version == "v3" and (
            self._detected_lora
            or "lora_rank" in dict_s2
            or any("lora_" in k for k in dict_s2.get("weight", {}))
        )
        if self.if_lora_v3:
            logger.info(f"检测到LoRA模型 (by checkpoint keys): {self.sovits_path}")

        # 加载权重
        if not self.if_lora_v3:
            logger.info(f"加载sovits_{self.model_version}模型权重")
            incompatible = self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
            if self.is_v2pro:
                # enc_q is training-only and may be absent. Conditioning weights
                # may not be silently skipped or left randomly initialized.
                missing = [key for key in incompatible.missing_keys if not key.startswith("enc_q.")]
                unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("enc_q.")]
                if missing or unexpected:
                    raise RuntimeError(
                        f"Incompatible {self.model_version} checkpoint: "
                        f"missing={missing}, unexpected={unexpected}"
                    )
        else:
            # 加载预训练模型和LoRA权重
            if getattr(self, "offline_only", False):
                require_checkpoint("SoVITS V3 base", self.sovits_pretrain_path)
            if not os.path.exists(self.sovits_pretrain_path):
                raise FileNotFoundError(f"SoVITS V3底模不存在: {self.sovits_pretrain_path}")

            logger.info(f"加载sovits_v3预训练权重: {self.sovits_pretrain_path}")
            self.vq_model.load_state_dict(load_sovits_new(self.sovits_pretrain_path)["weight"], strict=False)

            # 应用LoRA
            lora_rank = dict_s2.get("lora_rank", 32)  # 默认值，应该从模型中读取
            lora_config = LoraConfig(
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
            )
            logger.info(f"应用LoRA配置，rank={lora_rank}")
            self.vq_model.cfm = get_peft_model(self.vq_model.cfm, lora_config)

            # 加载LoRA权重
            self.vq_model.load_state_dict(dict_s2["weight"], strict=False)

            # 合并LoRA权重
            self.vq_model.cfm = self.vq_model.cfm.merge_and_unload()
            self.vq_model.eval()

        if self.is_v2pro:
            from aquatts.inference.speaker import SpeakerEncoder

            if getattr(self, "offline_only", False):
                require_checkpoint("Pro speaker encoder", self.sv_model_path)
            self.sv_model = SpeakerEncoder(self.sv_model_path, self.device, self.is_half)
        if hasattr(self, "_reference_cache"):
            self.clear_reference_cache()

    def _load_bigvgan_model(self):
        """Load BigVGAN vocoder model (required for v3).
        加载BigVGAN模型（v3模型需要）"""
        if self.model_version != "v3":
            self.bigvgan_model = None
            return

        try:


            bigvgan_path = self.bigvgan_path
            logger.info(f"加载BigVGAN模型: {bigvgan_path}")
            options = {}
            if getattr(self, "offline_only", False):
                require_files("BigVGAN", bigvgan_path, ["config.json", "bigvgan_generator.pt"])
                options["local_files_only"] = True

            # use_cuda_kernel=True：编译 anti-aliased activation 的融合 CUDA kernel，
            # 减少内存读写，对 CUDA Graph 之后的 L2 cache miss 不敏感。
            # 优先检查预编译 .pyd 缓存（无需 cl.exe/nvcc），否则尝试现场编译。
            from aquatts.bigvgan.cuda.load import (
                _get_cache_root,
                _get_gpu_cache_suffix,
            )

            _use_cuda_kernel = False
            kernel_override = os.environ.get("BIGVGAN_USE_CUDA_KERNEL", "").strip().lower()
            if self.device.type == "cuda" and kernel_override not in {"0", "false", "off", "no"}:
                if kernel_override in {"1", "true", "on", "yes"}:
                    _use_cuda_kernel = True
                else:
                    cache_binary = (_get_cache_root()
                                    / f"build_{_get_gpu_cache_suffix(self._tts_device_idx)}"
                                    / "anti_alias_activation_cuda.pyd")
                    if cache_binary.exists():
                        _use_cuda_kernel = True
                        logger.info("[BigVGAN] Loading cached CUDA kernel")
                    else:
                        try:
                            import subprocess
                            probe = subprocess.run(["nvcc", "--version"], capture_output=True, timeout=5)
                            _use_cuda_kernel = probe.returncode == 0
                        except (OSError, subprocess.TimeoutExpired):
                            logger.info("[BigVGAN] nvcc unavailable; using PyTorch")

            # activation1d.py 在首次 import 时执行模块级 load.load()，
            # 若此时无设备上下文则 CUDA kernel 内部状态绑定到 cuda:0，
            # 之后模型移到 cuda:1 会触发 CUDNN_STATUS_MAPPING_ERROR。
            # 用 torch.cuda.device() 确保 kernel 初始化在正确设备上进行。
            try:
                with torch.cuda.device(self.device) if self.is_half else nullcontext():
                    self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_path, use_cuda_kernel=_use_cuda_kernel, **options)
            except Exception as _kernel_err:
                if _use_cuda_kernel:
                    self.bigvgan_kernel_fallback_reason = str(_kernel_err)
                    _use_cuda_kernel = False
                    logger.warning(f"[BigVGAN] CUDA kernel 编译失败，回退到 PyTorch 实现: {_kernel_err}")
                    with torch.cuda.device(self.device) if self.is_half else nullcontext():
                        self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_path, use_cuda_kernel=False, **options)
                else:
                    raise
            self.bigvgan_kernel_mode = "cuda" if _use_cuda_kernel else "pytorch"
            self.bigvgan_model.remove_weight_norm()
            self.bigvgan_model = self.bigvgan_model.eval()

            if self.is_half:
                self.bigvgan_model = self.bigvgan_model.half().to(self.device)
            else:
                self.bigvgan_model = self.bigvgan_model.to(self.device)

            self._warmup_bigvgan_shapes()
        except Exception as e:
            logger.error(f"加载BigVGAN模型失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _prepare_cfm_adapter(self):
        if self.model_version != "v3":
            raise ValueError("cfm_unpadded requires a V3 model")
        from aquatts.inference.cfm import UnpaddedCFM
        self._cfm_adapter = UnpaddedCFM(self.vq_model.cfm)

    def _run_cfm(self, fea, lengths, prompt, steps):
        cfm = self.vq_model.cfm
        if not getattr(self, "cfm_unpadded", False):
            return cfm.inference(fea, lengths, prompt, steps, inference_cfg_rate=0)
        if self._cfm_adapter is None or self._cfm_adapter.cfm is not cfm:
            self._prepare_cfm_adapter()
        return self._cfm_adapter(fea, lengths, prompt, steps,
                                 unpadded_length=fea.size(1), inference_cfg_rate=0)

    def _init_reference_cache(self):
        self._reference_generation = 0
        self._reference_state_lock = threading.RLock()
        self._sovits_decode_lock = threading.Lock()
        self._reference_cache = ReferenceCache(
            int(os.environ.get("AQUA_SESSION_CACHE_MAX", "8")),
            record_ready=self._record_reference_ready, wait_ready=self._wait_reference_ready)
        self.reference_warmup_results = []

    def _normalize_reference_prompt(self, audio_path, text, language, *, ref_free=False):
        return normalize_reference(
            audio_path, text, language, model_version=self.model_version,
            languages=self.dict_language, splits=self.splits,
            fallback_text=_default_ref_free_prompt(),
            fallback_language="en" if _TTS_OUTPUT_LANGUAGE == "英文" else "all_ja",
            ref_free=ref_free)

    def _reference_runtime_identity(self):
        try:
            parameter = next(self.vq_model.parameters())
            return parameter.device, parameter.dtype
        except (AttributeError, StopIteration):
            device = torch.device(self.device)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            return device, torch.float16 if self.is_half else torch.float32

    def _reference_key(self, reference):
        device, dtype = self._reference_runtime_identity()
        return reference_key(reference, generation=self._reference_generation,
                             version=self.model_version, device=device, dtype=dtype,
                             is_pro=self.is_v2pro)

    def _record_reference_ready(self):
        device, _ = self._reference_runtime_identity()
        if device.type != "cuda":
            return None
        with torch.cuda.device(device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
        return device, event

    @staticmethod
    def _wait_reference_ready(ready):
        if ready is not None:
            device, event = ready
            torch.cuda.current_stream(device).wait_event(event)

    def clear_reference_cache(self):
        """Invalidate references after explicit clearing or model replacement."""
        with self._reference_state_lock:
            count = self._reference_cache.clear()
            self._reference_generation += 1
            return count

    def _validate_reference_features(self, reference, entry):
        required = {"refer_spec"}
        if not reference.ref_free:
            required.update(("prompt", "phones1", "bert1"))
        if self.is_v2pro:
            required.add("sv_embedding")
        if self.model_version == "v3":
            required.update(("mel2_norm", "prompt_fea_ref", "prompt_ge"))
        missing = sorted(key for key in required if entry.get(key) is None)
        if missing:
            raise RuntimeError(f"Incomplete reference features: {', '.join(missing)}")
        for key in required:
            value = entry[key]
            if (torch.is_tensor(value) and value.numel() == 0) or (isinstance(value, list) and not value):
                raise RuntimeError(f"Empty reference feature: {key}")

    def _get_reference_entry(self, reference):
        with self._reference_state_lock:
            key = self._reference_key(reference)
            def build():
                with self._sovits_decode_lock, torch.no_grad():
                    entry = self._build_reference_features(reference)
                self._validate_reference_features(reference, entry)
                if self._reference_key(reference) != key:
                    raise RuntimeError("Reference changed during preparation; retry with a stable file")
                return entry
            entry, status, evicted = self._reference_cache.get_or_build(key, build)
            return entry, status, key, evicted

    def warm_reference_cache(self, audio_path, text="", language="日文", *, strict=True):
        """Prepare the same normalized reference used by infer/infer_stream.

        Returns status/ready/key/resident_count/evicted_key. Strict failure raises;
        best-effort failure returns status='failed' and never publishes partial data.
        """
        try:
            reference = self._normalize_reference_prompt(audio_path, text, language)
            _, status, key, evicted = self._get_reference_entry(reference)
            device, _ = self._reference_runtime_identity()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ready = self._reference_cache.contains(key)
            result = {"status": status if ready else "evicted", "ready": ready,
                      "key": key, "resident_count": len(self._reference_cache),
                      "evicted_key": evicted}
            if strict and not ready:
                raise RuntimeError("Reference was evicted before warmup completed")
            return result
        except Exception as error:
            if strict:
                raise
            logger.warning("Reference warmup failed: %s", error)
            return {"status": "failed", "ready": False, "error": str(error)}

    def _warm_reference_list(self, references, *, strict):
        arguments = []
        normalized = set()
        for reference in references:
            if isinstance(reference, Voice):
                reference = reference.to_dict()
            if not isinstance(reference, dict):
                raise TypeError("warmup_references entries must be Voice objects or reference dictionaries")
            args = (reference["ref_audio_path"], reference.get("prompt_text", ""),
                    reference.get("prompt_language", "日文"))
            item = self._normalize_reference_prompt(*args)
            normalized.add((os.path.normcase(item.audio_path), item.text, item.language, item.ref_free))
            arguments.append(args)
        if len(normalized) > self._reference_cache.capacity:
            raise ValueError("warmup_references exceeds reference cache capacity")
        results = [self.warm_reference_cache(*args, strict=strict) for args in arguments]
        for result in results:
            if result["ready"] and not self._reference_cache.contains(result["key"]):
                result.update(status="evicted", ready=False)
                if strict:
                    raise RuntimeError("Reference was evicted during batch warmup")
        return results

    def _build_session_cache(self, ref_audio_path: str, prompt_text: str, prompt_language_code: str):
        try:
            reference = self._normalize_reference_prompt(ref_audio_path, prompt_text, prompt_language_code)
            return self._get_reference_entry(reference)[0]
        except Exception:
            logger.warning("构建会话缓存失败，回退到逐句计算。", exc_info=True)
            return {}

    def _normalize_semantic_reference(self, value, *, prompt_text, prompt_language, ref_free):
        if value is None:
            return None
        if self.model_version != "v3":
            raise ValueError("semantic_reference is supported only for V3")
        if ref_free:
            raise ValueError("semantic_reference cannot be combined with ref_free=True")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError("Mixed references require an explicit acoustic prompt_text")
        if isinstance(value, Voice):
            value = value.to_dict()
        if not isinstance(value, dict):
            raise TypeError("semantic_reference must be a Voice or reference dictionary")
        path, text = value.get("ref_audio_path"), value.get("prompt_text")
        if not isinstance(path, (str, os.PathLike)) or not os.fspath(path).strip():
            raise ValueError("semantic_reference requires ref_audio_path")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("semantic_reference requires a nonempty prompt_text")
        language = value.get("prompt_language", "日文")
        for component, label in (("acoustic", prompt_language), ("semantic", language)):
            if not isinstance(label, str) or (label not in self.dict_language and label not in REFERENCE_LANGUAGE_CODES):
                raise ValueError(f"Unknown {component} reference language: {label!r}")
        return self._normalize_reference_prompt(path, text, language)

    def _reference_session_for_request(self, acoustic, semantic=None):
        if semantic is None:
            return self._build_session_cache(acoustic.audio_path, acoustic.text, acoustic.language)
        # Keep both lookups in one model generation. A capacity-one cache may
        # evict the first entry, but the request retains its own complete values.
        with self._reference_state_lock:
            acoustic_entry = self._get_reference_entry(acoustic)[0]
            semantic_entry = self._get_reference_entry(semantic)[0]
            self._validate_reference_features(acoustic, acoustic_entry)
            self._validate_reference_features(semantic, semantic_entry)
            sources = [(semantic_entry, ("prompt", "phones1", "bert1")),
                       (acoustic_entry, ("refer_spec", "prompt_fea_ref", "prompt_ge", "mel2_norm"))]
            return {key: self._clone_cached_value(entry[key]) for entry, keys in sources for key in keys}


    @torch.no_grad()
    def _build_reference_features(self, reference):
        ref_audio_path, prompt_text, prompt_language_code = reference.audio_path, reference.text, reference.language
        cache_item = {}
        if not reference.ref_free:
            # 1) Prompt 相关（SSL → prompt）
            with torch.no_grad():
                wav16k, _ = librosa.load(ref_audio_path, sr=16000)
                wav16k = torch.from_numpy(wav16k)
                if self.is_half:
                    wav16k = wav16k.half().to(self.device)
                else:
                    wav16k = wav16k.float().to(self.device)
                # 追加极短静音，避免边界截断
                tail = torch.zeros(int(1600), dtype=wav16k.dtype, device=wav16k.device)
                wav16k = torch.cat([wav16k, tail])
                ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
                codes = self.vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                cache_item["prompt"] = prompt_semantic.unsqueeze(0).to(self.device)

            # 2) Prompt 文本 phones/bert
            phones1, bert1, _ = self.get_phones_and_bert(prompt_text or "", prompt_language_code or _default_lang_code())
            cache_item["phones1"] = phones1
            cache_item["bert1"] = bert1

        # 3) 参考频谱（所有版本可用）
        refer = self.get_spepc(ref_audio_path).to(self.device)
        if self.is_half:
            refer = refer.half()
        else:
            refer = refer.float()
        cache_item["refer_spec"] = refer
        if self.is_v2pro:
            cache_item["sv_embedding"] = self.sv_model.encode(ref_audio_path)

        # 4) v3 额外缓存：ref_audio 24k 的 mel2（归一化后）
        if self.model_version == "v3":
            import torchaudio
            ref_audio, ref_sr = torchaudio.load(ref_audio_path)
            ref_audio = ref_audio.to(self.device)
            if self.is_half:
                ref_audio = ref_audio.half()
            else:
                ref_audio = ref_audio.float()
            if ref_audio.shape[0] == 2:
                ref_audio = ref_audio.mean(0).unsqueeze(0)
            if ref_sr != 24000:
                ref_audio = self._resample(ref_audio, ref_sr)

            mel2 = _normalize_v3_spec(_v3_mel_spectrogram(ref_audio))
            cache_item["mel2_norm"] = mel2

            # 5) v3 额外缓存：prompt 侧 decode_encp 结果，避免每句重复做同一份参考编码
            phoneme_ids0 = torch.LongTensor(phones1).to(self.device).unsqueeze(0)
            with torch.no_grad():
                prompt_fea_ref, prompt_ge = self.vq_model.decode_encp(
                    cache_item["prompt"].unsqueeze(0),
                    phoneme_ids0,
                    refer,
                )
            cache_item["prompt_fea_ref"] = prompt_fea_ref
            cache_item["prompt_ge"] = prompt_ge

        return cache_item

    def _clone_cached_value(self, value):
        """Clone cached tensors to avoid mutation by downstream inference paths.
        避免读取会话缓存后被后续推理路径原地复用/污染。"""
        if torch.is_tensor(value):
            if value.is_cuda:
                # An entry may be evicted while another stream clones it.
                value.record_stream(torch.cuda.current_stream(value.device))
            return value.clone()
        if isinstance(value, list):
            return [self._clone_cached_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_cached_value(item) for item in value)
        if isinstance(value, dict):
            return {k: self._clone_cached_value(v) for k, v in value.items()}
        return value

    def get_bert_feature(self, text, word2ph):
        """Extract BERT features for Chinese text at phone level.
        获取BERT特征"""
        self._ensure_bert_model()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)

        return phone_level_feature.T

    def get_phones_and_bert(self, text, language, final=False):
        """Extract phoneme sequence and BERT features from text.
        获取音素和BERT特征"""
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")

            # 处理中文中的英文字符
            if language == "all_zh" and re.search(r'[A-Za-z]', formattext):

                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return self.get_phones_and_bert(formattext, "zh")

            phones, word2ph, norm_text = self.clean_text_inf(formattext, language)

            if language == "all_zh":
                bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            else:
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if self.is_half else torch.float32,
                ).to(self.device)

        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist = []
            langlist = []

            # 处理多语言混合
            if language == "auto":
                for tmp in self._segment_text(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in self._segment_text(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in self._segment_text(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # Can't distinguish CJK characters — use user-provided language
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])

            logger.debug(f"文本分段: {textlist}")
            logger.debug(f"语言分段: {langlist}")

            phones_list = []
            bert_list = []
            norm_text_list = []

            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)

            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        # Handle very short content — pad and retry
        # 处理过短的内容
        dtype = torch.float16 if self.is_half else torch.float32
        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, final=True)

        return phones, bert.to(dtype), norm_text

    def clean_text_inf(self, text, language):
        """Clean text and convert to phoneme sequence.
        清理文本并转换为音素"""
        language = language.replace("all_", "")
        if getattr(self, "offline_only", False):
            check_frontend(language, self.sovits_version)
        phones, word2ph, norm_text = clean_text(text, language, self.sovits_version)
        phones = cleaned_text_to_sequence(phones, self.sovits_version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        """Get BERT features based on language.
        根据语言获取BERT特征"""
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if self.is_half else torch.float32,
            ).to(self.device)

        return bert

    def _get_audio_sr_model(self):
        if self.sr_model is not None:
            return self.sr_model
        with self._sr_load_lock:
            if self.sr_model is None:
                if getattr(self, "offline_only", False):
                    require_files("AP-BWE", "AP_BWE_main/24kto48k", ["config.json", "g_24kto48k.zip"])
                from tools.audio_sr import AP_BWE
                self.sr_model = AP_BWE(self.device, DictToAttrRecursive)
        return self.sr_model

    def _audio_sr(self, audio, sr):
        """Audio super-resolution processing.
        音频超分辨率处理"""
        try:

            sr_model = self._get_audio_sr_model()
            return sr_model(audio, sr)
        except ImportError:
            logger.warning("未找到音频超分模块，跳过超分处理")
            return audio.cpu().detach().numpy(), sr
        except FileNotFoundError:
            logger.warning("未找到音频超分模型参数，跳过超分处理")
            return audio.cpu().detach().numpy(), sr

    def _decode_v2(self, semantic, phones, ref_audio_path, session, inp_refs, speed):
        """Decode v1/v2/Pro with aligned spectrum and speaker reference pairs."""
        references = []
        speakers = []
        for reference in inp_refs or ():
            try:
                path = os.fspath(reference) if isinstance(reference, (str, os.PathLike)) else reference.name
                spectrum = self.get_spepc(path).to(self.device)
                spectrum = spectrum.half() if self.is_half else spectrum.float()
                speaker = self.sv_model.encode(path) if self.is_v2pro else None
            except Exception as exc:
                logger.warning("Failed to load extra reference %s: %s", reference, exc)
                continue
            references.append(spectrum)
            if self.is_v2pro:
                speakers.append(speaker)

        if not references:
            spectrum = session.get("refer_spec")
            if spectrum is None:
                spectrum = self.get_spepc(ref_audio_path).to(self.device)
                spectrum = spectrum.half() if self.is_half else spectrum.float()
            references.append(spectrum)
            if self.is_v2pro:
                speaker = session.get("sv_embedding")
                if speaker is None:
                    speaker = self.sv_model.encode(ref_audio_path)
                speakers.append(speaker)

        kwargs = {"speed": speed}
        if self.is_v2pro:
            kwargs["sv_emb"] = speakers
        return self.vq_model.decode(
            semantic,
            torch.LongTensor(phones).to(self.device).unsqueeze(0),
            references,
            **kwargs,
        )[0][0]

    def get_spepc(self, filename):
        """Extract spectrogram from reference audio.
        获取频谱特征"""
        audio, sampling_rate = librosa.load(filename, sr=int(self.hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)

        audio_norm = audio.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )

        return spec

    def infer(self,
              text,
              ref_audio_path,
              prompt_text=None,
              text_language="日文",
              prompt_language="日文",
              how_to_cut="不切",
              top_k=20,
              top_p=0.6,
              temperature=0.6,
              speed=1.0,
              sample_steps=16,
              ref_free=False,
              pause_second=0.3,
              if_freeze=False,
              inp_refs=None,
              if_sr=False,
              enable_cuda_graph=None,
              enable_static_kv=True,
              max_sec_override=None,
              semantic_reference=None):
        """
        Run TTS inference (non-streaming, returns full audio).
        执行TTS推理

        Args:
            text: Target text to synthesize / 要合成的目标文本
            ref_audio_path: Path to reference audio / 参考音频路径
            prompt_text: Reference transcript. Uses ref_free mode if None / 参考文本
            text_language: Language of the target text / 目标文本的语言
            prompt_language: Language of the reference text / 参考文本的语言
            how_to_cut: Text segmentation strategy / 文本切分方式:
                "不切" (no cut), "凑四句一切", "凑50字一切",
                "按中文句号。切", "按英文句号.切", "按标点符号切"
            top_k, top_p, temperature: GPT sampling parameters / GPT采样参数
            speed: Speed multiplier / 语速控制
            sample_steps: Sampling steps for v3 models / v3模型的采样步数
            ref_free: Whether to use reference-free mode / 是否使用无参考模式
            pause_second: Pause duration between sentences / 句间停顿秒数
            if_freeze: Whether to reuse previous cache (reduces randomness) / 是否重用上次的缓存
            inp_refs: Additional reference audio paths for voice mixing / 额外的参考音频列表
            if_sr: Whether to use audio super-resolution (v3 only) / 是否使用音频超分辨率
            semantic_reference: Optional V3 Voice/dict supplying prompt/phones/BERT.
                The ordinary reference continues to supply acoustic conditions.

        Returns:
            tuple: (sample_rate, audio_data) / (采样率, 音频数据)
        """
        try:
            # Prepare input
            # 准备输入
            text = text.strip()
            if not text:
                raise ValueError("推理文本不能为空")

            logger.info(f"开始推理: '{text[:30]}...'")

            # Define v3 CFM denorm function up front to avoid undefined branches
            # 统一定义 v3 CFM 解码所需的反归一化函数，避免某些分支下未定义
            # Convert language code
            # 转换语言代码
            if text_language in self.dict_language:
                text_language_code = self.dict_language[text_language]
            else:
                text_language_code = _default_lang_code()
                logger.warning(f"未知语言: {text_language}，使用默认语言: {_TTS_OUTPUT_LANGUAGE}")

            semantic = self._normalize_semantic_reference(
                semantic_reference, prompt_text=prompt_text, prompt_language=prompt_language, ref_free=ref_free)
            reference = self._normalize_reference_prompt(
                ref_audio_path, prompt_text, prompt_language, ref_free=ref_free)
            ref_audio_path, prompt_text = reference.audio_path, reference.text
            prompt_language_code, ref_free = reference.language, reference.ref_free

            # Segment text according to selected strategy
            # 根据选择的切分方式处理文本
            logger.info(f"文本切分方式: {how_to_cut}")
            if how_to_cut == "凑四句一切":
                text = cut1(text)
            elif how_to_cut == "凑50字一切":
                text = cut2(text)
            elif how_to_cut == "按中文句号。切":
                text = cut3(text)
            elif how_to_cut == "按英文句号.切":
                text = cut4(text)
            elif how_to_cut == "按标点符号切":
                text = cut5(text)

            # Split by newlines
            # 按行切分
            while "\n\n" in text:
                text = text.replace("\n\n", "\n")
            texts = text.split("\n")
            texts = process_text(texts)

            # Initialize result buffers
            # 初始化结果
            audio_outputs = []
            sr = self.hps.data.sampling_rate if self.model_version != "v3" else 24000

            # Create silence gap between sentences
            # 创建句间停顿的静音
            zero_wav = torch.zeros(
                int(sr * pause_second),
                dtype=torch.float16 if self.is_half else torch.float32  # 根据is_half决定类型
            ).to(self.device)

            # Process reference audio (prefer session-level cache)
            # 处理参考音频（会话级缓存优先）
            sess = self._reference_session_for_request(reference, semantic)
            prompt = sess.get("prompt")

            # Extract phones and BERT features from reference audio
            # 获取参考音频的音素和BERT特征
            if not ref_free:
                if "phones1" in sess and "bert1" in sess:
                    phones1, bert1 = sess["phones1"], sess["bert1"]
                else:
                    phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_language_code)

            # Initialize inference cache
            # 初始化缓存
            cache = {}

            effective_max_sec = self._get_effective_max_sec(max_sec_override)
            # Synthesize sentence by sentence
            # 分句合成
            logger.info(f"分成 {len(texts)} 个句子进行合成")
            for i_text, text_item in enumerate(texts):
                # Skip empty sentences
                # 跳过空句
                if len(text_item.strip()) == 0:
                    continue

                # Ensure sentence ends with punctuation
                # 确保句子以标点符号结尾
                if text_item and text_item[-1] not in self.splits:
                    text_item += "。" if text_language != "英文" else "."

                logger.info(f"处理第 {i_text + 1} 句: '{text_item}'")

                # Extract phones and BERT from target text
                # 获取目标文本的音素和BERT特征
                phones2, bert2, norm_text2 = self.get_phones_and_bert(text_item, text_language_code)
                logger.info(f"处理后的目标文本: {norm_text2}")

                # Merge phones and BERT features
                # 合并音素和BERT特征
                if not ref_free:
                    bert = torch.cat([bert1, bert2], 1)
                    all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
                else:
                    bert = bert2
                    all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

                bert = bert.to(self.device).unsqueeze(0)
                all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

                # Handle cache (reuse if frozen)
                # 处理缓存
                if i_text in cache and if_freeze:
                    logger.info("使用缓存的GPT输出")
                    pred_semantic = cache[i_text]
                else:
                    # Run GPT inference
                    # GPT推理
                    logger.info("执行GPT推理...")
                    with torch.no_grad():
                        pred_semantic, idx, _semantic_attempts = self._infer_semantic_with_guard(
                            text_item=text_item,
                            target_phone_count=len(phones2),
                            all_phoneme_ids=all_phoneme_ids,
                            all_phoneme_len=all_phoneme_len,
                            prompt=None if ref_free else prompt,
                            bert=bert,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            effective_max_sec=effective_max_sec,
                            enable_cuda_graph=enable_cuda_graph,
                            enable_static_kv=enable_static_kv,
                        )
                        cache[i_text] = pred_semantic

                # Run SoVITS decoding
                # SoVITS推理
                logger.info("执行SoVITS解码...")

                if self.model_version != "v3":
                    with self._sovits_decode_lock:
                        audio = self._decode_v2(
                            pred_semantic, phones2, ref_audio_path, sess, inp_refs, speed
                        )

                    # Prevent audio clipping
                    # 防止爆音
                    max_audio = torch.abs(audio).max()
                    if max_audio > 1:
                        audio = audio / max_audio

                    # Add to output list
                    # 添加到输出列表
                    audio_outputs.append(self._trim_generated_lead(audio, sr))
                    audio_outputs.append(zero_wav)  # Inter-sentence pause / 句间停顿

                else:
                    with self._sovits_decode_lock:
                        # Decode for v3 model
                        # v3模型解码
                        import torchaudio

                        # Prefer session-cached reference spectrogram
                        # 优先使用会话缓存的参考频谱
                        refer = sess.get("refer_spec")
                        if refer is None:
                            refer = self.get_spepc(ref_audio_path).to(self.device)
                            refer = refer.half() if self.is_half else refer.float()

                        phoneme_ids0 = torch.LongTensor(phones1).to(self.device).unsqueeze(0)
                        phoneme_ids1 = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

                        # Extract reference audio features (prefer session cache)
                        # 提取参考音频特征（优先使用会话缓存）
                        fea_ref = self._clone_cached_value(sess.get("prompt_fea_ref"))
                        ge = self._clone_cached_value(sess.get("prompt_ge"))
                        if fea_ref is None or ge is None:
                            fea_ref, ge = self.vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)

                        # Load and process reference audio
                        # 加载并处理参考音频
                        mel2 = sess.get("mel2_norm")
                        if mel2 is None:
                            ref_audio, ref_sr = torchaudio.load(ref_audio_path)
                            ref_audio = ref_audio.to(self.device)
                            ref_audio = ref_audio.half() if self.is_half else ref_audio.float()
                            if ref_audio.shape[0] == 2:
                                ref_audio = ref_audio.mean(0).unsqueeze(0)
                            if ref_sr != 24000:
                                ref_audio = self._resample(ref_audio, ref_sr)
                            mel2 = _normalize_v3_spec(_v3_mel_spectrogram(ref_audio))

                        # Align lengths to T_min
                        # 调整长度
                        T_min = min(mel2.shape[2], fea_ref.shape[2])
                        mel2 = mel2[:, :, :T_min]
                        fea_ref = fea_ref[:, :, :T_min]
                        if (T_min > 468):
                            mel2 = mel2[:, :, -468:]
                            fea_ref = fea_ref[:, :, -468:]
                            T_min = 468

                        # Set chunk length
                        # 设置块长度
                        chunk_len = 934 - T_min
                        # 根据is_half决定是否使用half
                        if self.is_half:
                            mel2 = mel2.half()
                        else:
                            mel2 = mel2.float()

                        # Decode
                        # Decode target features
                        # 解码目标特征
                        fea_todo, ge = self.vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)

                        # Process in chunks
                        # 分块处理
                        cfm_resss = []
                        idx = 0
                        while True:
                            fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
                            if fea_todo_chunk.shape[-1] == 0:
                                break

                            idx += chunk_len
                            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)

                            # Run CFM inference
                            # CFM推理
                            cfm_res = self._run_cfm(
                                fea,
                                torch.LongTensor([fea.size(1)]).to(fea.device),
                                mel2,
                                sample_steps,
                            )

                            cfm_res = cfm_res[:, :, mel2.shape[2]:]
                            mel2 = cfm_res[:, :, -T_min:]
                            fea_ref = fea_todo_chunk[:, :, -T_min:]
                            cfm_resss.append(cfm_res)

                        # Concatenate chunk results
                        # 合并结果
                        cmf_res = torch.cat(cfm_resss, 2)
                        cmf_res = _denormalize_v3_spec(cmf_res)

                        # BigVGAN generates waveform / BigVGAN生成波形
                        # torch.cuda.device() ensures at::cuda::getCurrentCUDAStream()
                        # inside the fused CUDA kernel uses the correct device stream,
                        # preventing CUDNN_STATUS_MAPPING_ERROR on non-default GPUs.
                        wav_gen = self._run_bigvgan(cmf_res)
                        audio = wav_gen[0][0]

                        # Prevent audio clipping
                        # 防止爆音
                        max_audio = torch.abs(audio).max()
                        if max_audio > 1:
                            audio = audio / max_audio

                        # Add to output list
                        # 添加到输出列表
                        audio_outputs.append(self._trim_generated_lead(audio, sr))
                        audio_outputs.append(zero_wav)  # Inter-sentence pause / 句间停顿

            # Concatenate all audio segments
            # 合并所有音频片段
            if audio_outputs:
                final_audio = torch.cat(audio_outputs, 0)

                # Audio super-resolution (v3 models only)
                # 音频超分(仅v3模型支持)
                if if_sr and self.model_version == "v3":
                    try:
                        logger.info("执行音频超分...")
                        # Lazy-init super-resolution model
                        # 初始化超分模型（如果未初始化）
                        sr_model = self._get_audio_sr_model()

                        # Run super-resolution
                        # 进行音频超分
                        final_audio, sr = sr_model(final_audio.unsqueeze(0), sr)

                        # Prevent clipping again
                        # 再次防止爆音
                        max_audio = np.abs(final_audio).max()
                        if max_audio > 1:
                            final_audio = final_audio / max_audio
                    except Exception as e:
                        logger.warning(f"音频超分失败: {e}")
                        logger.warning(traceback.format_exc())
                        final_audio = final_audio.cpu().detach().numpy()
                else:
                    final_audio = final_audio.cpu().detach().numpy()

                # Ensure output is float32 (numpy arrays only, not tensors)
                # 确保音频数据是float32类型（只针对numpy数组，不处理tensor）
                if isinstance(final_audio, np.ndarray):
                    if 'float16' in str(final_audio.dtype):
                        final_audio = final_audio.astype(np.float32)

                # Return result
                # 返回结果
                logger.info(f"推理完成，生成音频长度: {len(final_audio) / sr:.2f}秒")
                return sr, final_audio
            else:
                raise ValueError("未能生成有效音频")

        except Exception as e:
            logger.error(f"推理失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def infer_stream(self,
                     text,
                     ref_audio_path,
                     prompt_text=None,
                     text_language="日文",
                     prompt_language="日文",
                     how_to_cut="按标点符号切",
                     preset: str = None,
                     top_k=20,
                     top_p=0.6,
                     temperature=0.6,
                     speed=1.0,
                     sample_steps=16,
                     ref_free=False,
                     pause_second=0.3,
                     if_freeze=False,
                     inp_refs=None,
                     if_sr=False,
                     enable_cuda_graph=None,
                     enable_static_kv=True,
                     chunk_size_seconds: float = None,
                     max_sec_override: float = None,
                     collect_t2s_stats: bool = False,
                     semantic_reference=None):
        """
        Streaming TTS inference — yields audio chunks as they are generated.
        流式执行TTS推理，逐步返回音频块

        Unlike infer(), this method is a generator that produces audio chunk-by-chunk
        for low-latency streaming playback.
        与infer函数相比，该函数是一个生成器，会逐块返回处理后的音频

        Args:
            text: Target text to synthesize / 要合成的目标文本
            ref_audio_path: Path to reference audio / 参考音频路径
            prompt_text: Reference transcript. Uses ref_free mode if None / 参考文本
            text_language: Language of the target text / 目标文本的语言
            prompt_language: Language of the reference text / 参考文本的语言
            how_to_cut: Text segmentation strategy / 文本切分方式
            preset: Quality preset — "fast", "balanced", "quality".
                Overrides top_k/top_p/temperature/speed/sample_steps defaults.
                Explicit per-argument values take precedence.
                质量预设名 — "fast", "balanced", "quality"。
                如果设置，会覆盖对应默认值。显式传入的参数仍然优先。
            top_k, top_p, temperature: GPT sampling parameters / GPT采样参数
            speed: Speed multiplier / 语速控制
            sample_steps: Sampling steps for v3 models / v3模型的采样步数
            ref_free: Whether to use reference-free mode / 是否使用无参考模式
            pause_second: Pause duration between sentences / 句间停顿秒数
            if_freeze: Whether to reuse previous cache (reduces randomness) / 是否重用上次的缓存
            inp_refs: Additional reference audio paths for voice mixing / 额外的参考音频列表
            if_sr: Whether to use audio super-resolution (v3 only) / 是否使用音频超分辨率
            chunk_size_seconds: If set, partition output into fixed-duration chunks / 按秒数分块输出
            collect_t2s_stats: Record synchronized T2S token throughput stats / 记录T2S吞吐统计
            semantic_reference: Optional V3 Voice/dict supplying prompt/phones/BERT.
                The ordinary reference continues to supply acoustic conditions.

        Yields:
            Generator of (sample_rate, audio_chunk, text_segment) tuples.
            生成器：每次生成 (采样率, 音频数据片段, 文本段落)

            The **first tuple is always (sr, None, "")** — a header yield that signals
            the sample rate before any audio is ready. Callers must skip it:
                for sr, chunk, text in tts.infer_stream(...):
                    if chunk is not None:
                        ...  # process audio
            第一个 tuple 固定为 (sr, None, "")，用于在首帧音频就绪前传递采样率。
            调用方需跳过 chunk 为 None 的条目。
        """
        # Apply generation preset — sets top_k/top_p/temperature/speed/sample_steps
        if preset is not None:
            preset_params = _apply_preset(preset)
            top_k = preset_params.get("top_k", top_k)
            top_p = preset_params.get("top_p", top_p)
            temperature = preset_params.get("temperature", temperature)
            speed = preset_params.get("speed", speed)
            sample_steps = preset_params.get("sample_steps", sample_steps)

        try:
            # Prepare input
            # 准备输入
            text = text.strip()
            if not text:
                raise ValueError("推理文本不能为空")

            logger.info(f"开始流式推理: '{text[:30]}...'")

            # Define v3 CFM denorm function up front to avoid undefined branches
            # 统一定义 v3 CFM 解码所需的反归一化函数，避免某些分支下未定义
            # Convert language code
            # 转换语言代码
            if text_language in self.dict_language:
                text_language_code = self.dict_language[text_language]
            else:
                text_language_code = _default_lang_code()
                logger.warning(f"未知语言: {text_language}，使用默认语言: {_TTS_OUTPUT_LANGUAGE}")

            semantic = self._normalize_semantic_reference(
                semantic_reference, prompt_text=prompt_text, prompt_language=prompt_language, ref_free=ref_free)
            reference = self._normalize_reference_prompt(
                ref_audio_path, prompt_text, prompt_language, ref_free=ref_free)
            ref_audio_path, prompt_text = reference.audio_path, reference.text
            prompt_language_code, ref_free = reference.language, reference.ref_free

            # Segment text according to selected strategy
            # 根据选择的切分方式处理文本
            logger.info(f"文本切分方式: {how_to_cut}")
            if how_to_cut == "凑四句一切":
                text = cut1(text)
            elif how_to_cut == "凑50字一切":
                text = cut2(text)
            elif how_to_cut == "按中文句号。切":
                text = cut3(text)
            elif how_to_cut == "按英文句号.切":
                text = cut4(text)
            elif how_to_cut == "按标点符号切":
                text = cut5(text)

            # Split by newlines
            # 按行切分
            while "\n\n" in text:
                text = text.replace("\n\n", "\n")
            texts = text.split("\n")
            texts = process_text(texts)

            # Initialize result buffers
            # 初始化结果
            sr = self.hps.data.sampling_rate if self.model_version != "v3" else 24000

            # 首先返回采样率
            yield sr, None, ""

            # Create silence gap between sentences
            # 创建句间停顿的静音
            zero_wav = torch.zeros(
                int(sr * pause_second),
                dtype=torch.float16 if self.is_half else torch.float32  # 根据is_half决定类型
            ).to(self.device)

            chunk_samples = None
            if chunk_size_seconds is not None and chunk_size_seconds > 0:
                chunk_samples = max(1, int(sr * chunk_size_seconds))
                logger.info(f"🎛️ 启用分块输出: {chunk_size_seconds:.2f}s -> {chunk_samples} samples")

            def _yield_audio_segments(audio_np: np.ndarray, text_payload: str):
                if chunk_samples is None or audio_np is None:
                    yield sr, audio_np, text_payload
                    return
                start = 0
                total_len = audio_np.shape[-1]
                first_chunk = True
                while start < total_len:
                    end = min(total_len, start + chunk_samples)
                    sub_chunk = audio_np[start:end]
                    if sub_chunk.size == 0:
                        break
                    yield sr, sub_chunk, text_payload if first_chunk else ""
                    first_chunk = False
                    start = end

            # Process reference audio (prefer session-level cache)
            # 处理参考音频（会话级缓存优先）
            sess = self._reference_session_for_request(reference, semantic)
            prompt = sess.get("prompt")

            # Extract phones and BERT features from reference audio
            # 获取参考音频的音素和BERT特征（会话级缓存优先）
            if not ref_free:
                if "phones1" in sess and "bert1" in sess:
                    phones1, bert1 = sess["phones1"], sess["bert1"]
                else:
                    phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_language_code)

            # Initialize inference cache
            # 初始化缓存
            cache = {}

            effective_max_sec = self._get_effective_max_sec(max_sec_override)
            if max_sec_override is not None:
                logger.info(f"⏱️ 应用最大时长限制: {effective_max_sec:.2f}s (原 {self.max_sec:.2f}s)")
            # 分句合成
            logger.info(f"分成 {len(texts)} 个句子进行流式合成")
            for i_text, text_item in enumerate(texts):
                # Skip empty sentences
                # 跳过空句
                if len(text_item.strip()) == 0:
                    continue

                # Ensure sentence ends with punctuation
                # 确保句子以标点符号结尾
                if text_item and text_item[-1] not in self.splits:
                    text_item += "。" if text_language != "英文" else "."

                logger.info(f"处理第 {i_text + 1} 句: '{text_item}'")

                # Extract phones and BERT from target text
                # 获取目标文本的音素和BERT特征
                phones2, bert2, norm_text2 = self.get_phones_and_bert(text_item, text_language_code)
                logger.info(f"处理后的目标文本: {norm_text2}")

                # Merge phones and BERT features
                # 合并音素和BERT特征
                if not ref_free:
                    bert = torch.cat([bert1, bert2], 1)
                    all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
                else:
                    bert = bert2
                    all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

                bert = bert.to(self.device).unsqueeze(0)
                all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

                # Handle cache (reuse if frozen)
                # 处理缓存
                if i_text in cache and if_freeze:
                    logger.info("使用缓存的GPT输出")
                    pred_semantic = cache[i_text]
                else:
                    # Run GPT inference
                    # GPT推理
                    logger.info("执行GPT推理...")
                    with torch.no_grad():
                        if collect_t2s_stats:
                            self._sync_t2s_timing()
                            t2s_start = time.perf_counter()
                        pred_semantic, idx, _semantic_attempts = self._infer_semantic_with_guard(
                            text_item=text_item,
                            target_phone_count=len(phones2),
                            all_phoneme_ids=all_phoneme_ids,
                            all_phoneme_len=all_phoneme_len,
                            prompt=None if ref_free else prompt,
                            bert=bert,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            effective_max_sec=effective_max_sec,
                            enable_cuda_graph=enable_cuda_graph,
                            enable_static_kv=enable_static_kv,
                        )
                        if collect_t2s_stats:
                            self._sync_t2s_timing()
                            self._record_t2s_stat(text_item, idx, time.perf_counter() - t2s_start)
                        cache[i_text] = pred_semantic

                # Run SoVITS decoding
                # SoVITS推理
                logger.info("执行SoVITS解码...")

                if self.model_version != "v3":
                    with self._sovits_decode_lock:
                        audio = self._decode_v2(
                            pred_semantic, phones2, ref_audio_path, sess, inp_refs, speed
                        )

                    # Prevent audio clipping
                    # 防止爆音
                    max_audio = torch.abs(audio).max()
                    if max_audio > 1:
                        audio = audio / max_audio

                    # Convert to numpy and stream back
                    # 转换为numpy并流式返回
                    audio_chunk = audio.cpu().detach().numpy()

                    # Ensure audio is float32
                    # 确保音频数据是float32类型
                    if hasattr(audio_chunk, 'dtype') and 'float16' in str(audio_chunk.dtype):
                        audio_chunk = audio_chunk.astype(np.float32)

                    # Fade out sentence ending to prevent click artifacts
                    # 句尾淡出，消除突然截断的爆音感
                    audio_chunk = self._apply_fade_out(audio_chunk, sr)
                    audio_chunk = self._trim_generated_lead(audio_chunk, sr)

                    # Stream current sentence audio + text (chunked if configured)
                    # 流式返回当前句子的音频和对应的文本（可分块）
                    for _sr, _chunk, _text in _yield_audio_segments(audio_chunk, text_item):
                        yield _sr, _chunk, _text

                    # Yield inter-sentence silence
                    # 返回句间停顿
                    pause_chunk = zero_wav.cpu().detach().numpy()
                    if hasattr(pause_chunk, 'dtype') and 'float16' in str(pause_chunk.dtype):
                        pause_chunk = pause_chunk.astype(np.float32)
                    yield sr, pause_chunk, ""  # No text for pause / 停顿不需要文本

                else:
                    # Decode for v3 model (serialized — CFM/BigVGAN share GPU resources)
                    # v3模型解码（串行化 — CFM/BigVGAN 共享 GPU 资源）
                    import torchaudio

                    def decode_chunks():
                        refer = sess.get("refer_spec")
                        if refer is None:
                            refer = self.get_spepc(ref_audio_path).to(self.device)
                            refer = refer.half() if self.is_half else refer.float()

                        phoneme_ids0 = torch.LongTensor(phones1).to(self.device).unsqueeze(0)
                        phoneme_ids1 = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

                        # Extract reference audio features (prefer session cache)
                    # 提取参考音频特征（优先使用会话缓存）
                        fea_ref = self._clone_cached_value(sess.get("prompt_fea_ref"))
                        ge = self._clone_cached_value(sess.get("prompt_ge"))
                        if fea_ref is None or ge is None:
                            fea_ref, ge = self.vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)

                        # Load and process reference audio
                    # 加载并处理参考音频
                        # Extract mel features (prefer cache) / 提取mel特征（缓存优先）
                        mel2 = sess.get("mel2_norm")
                        # Normalization parameters needed for denorm regardless of cache hit
                        # 归一化/反归一化参数与函数（无论是否命中缓存都需要）
                        if mel2 is None:
                            ref_audio, ref_sr = torchaudio.load(ref_audio_path)
                            ref_audio = ref_audio.to(self.device)
                            ref_audio = ref_audio.half() if self.is_half else ref_audio.float()
                            if ref_audio.shape[0] == 2:  # Convert to mono / 转单声道
                                ref_audio = ref_audio.mean(0).unsqueeze(0)
                            if ref_sr != 24000:
                                ref_audio = self._resample(ref_audio, ref_sr)
                            mel2 = _normalize_v3_spec(_v3_mel_spectrogram(ref_audio))

                        # Align lengths to T_min
                    # 调整长度
                        T_min = min(mel2.shape[2], fea_ref.shape[2])
                        mel2 = mel2[:, :, :T_min]
                        fea_ref = fea_ref[:, :, :T_min]
                        if (T_min > 468):
                            mel2 = mel2[:, :, -468:]
                            fea_ref = fea_ref[:, :, -468:]
                            T_min = 468

                        # Set chunk length
                    # 设置块长度
                        default_chunk_len = 934 - T_min
                        chunk_len = default_chunk_len
                        stream_v3_chunks = chunk_samples is not None and chunk_samples > 0
                        if stream_v3_chunks:
                            target_chunk_frames = max(
                                8, int(round((chunk_samples / float(sr)) * (sr / 256.0)))
                            )
                            chunk_len = max(8, min(default_chunk_len, target_chunk_frames))
                            logger.info(
                                "[v3-stream] enable true streaming: mel_chunk=%s (default=%s, target_frames=%s)"
                                % (chunk_len, default_chunk_len, target_chunk_frames)
                            )
                        # Cast to half if is_half / 根据is_half决定精度
                        if self.is_half:
                            mel2 = mel2.half()
                        else:
                            mel2 = mel2.float()

                        # Decode target features / 解码目标特征
                        fea_todo, ge = self.vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)

                        # Process in chunks / 分块处理
                        cfm_resss = []
                        _t_cfm_total = 0.0
                        idx = 0
                        total_todo_frames = fea_todo.shape[2]
                        stream_chunk_index = 0
                        lead = LeadingSilence(sr)
                        pending_text = text_item
                        while True:
                            chunk_end = min(total_todo_frames, idx + chunk_len)
                            fea_todo_chunk = fea_todo[:, :, idx:chunk_end]
                            if fea_todo_chunk.shape[-1] == 0:
                                break

                            idx = chunk_end
                            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)

                            # Run CFM inference / CFM推理
                            _t0 = time.perf_counter()
                            cfm_res = self._run_cfm(
                                fea,
                                torch.LongTensor([fea.size(1)]).to(fea.device),
                                mel2,
                                sample_steps,
                            )
                            if self._stream_sync_timing_enabled and self.device.type == "cuda":
                                torch.cuda.synchronize(self.device)
                            _t_cfm_chunk = (
                                time.perf_counter() - _t0
                                if self._stream_sync_timing_enabled
                                else None
                            )

                            cfm_res = cfm_res[:, :, mel2.shape[2]:]
                            mel2 = cfm_res[:, :, -T_min:]
                            fea_ref = fea_todo_chunk[:, :, -T_min:]
                            if _t_cfm_chunk is not None:
                                _t_cfm_total = _t_cfm_total + _t_cfm_chunk

                            if stream_v3_chunks:
                                stream_chunk_index += 1
                                is_last_stream_chunk = idx >= total_todo_frames
                                chunk_mel = _denormalize_v3_spec(cfm_res)
                                _t1 = time.perf_counter()
                                wav_gen = self._run_bigvgan(chunk_mel)
                                audio = wav_gen[0][0]
                                if self._stream_sync_timing_enabled and self.device.type == "cuda":
                                    torch.cuda.synchronize(self.device)
                                if self._stream_sync_timing_enabled:
                                    logger.info(
                                        "[v3-stream] chunk=%s cfm=%.1fms bigvgan=%.1fms mel_T=%s"
                                        % (
                                            stream_chunk_index,
                                            _t_cfm_chunk * 1000.0,
                                            (time.perf_counter() - _t1) * 1000.0,
                                            chunk_mel.shape[2],
                                        )
                                    )
                                else:
                                    logger.info(
                                        "[v3-stream] chunk=%s mel_T=%s"
                                        % (stream_chunk_index, chunk_mel.shape[2])
                                    )

                                max_audio = torch.abs(audio).max()
                                if max_audio > 1:
                                    audio = audio / max_audio

                                audio_chunk = self._finalize_stream_chunk(
                                    audio,
                                    sr,
                                    if_sr=if_sr,
                                    is_last_chunk=is_last_stream_chunk,
                                    apply_fade_in=(stream_chunk_index > 1),
                                )
                                audio_chunk = lead.feed(audio_chunk)
                                if audio_chunk is not None:
                                    yield sr, audio_chunk, pending_text
                                    pending_text = ""
                            else:
                                cfm_resss.append(cfm_res)

                        final_audio = lead.finish()
                        if final_audio is not None:
                            yield sr, final_audio, pending_text

                        if not stream_v3_chunks:
                            cmf_res = torch.cat(cfm_resss, 2)
                            cmf_res = _denormalize_v3_spec(cmf_res)

                            # Drain prior async CFM work before timing BigVGAN,
                            # otherwise the synchronize below would count CFM+BigVGAN together.
                            if self._stream_sync_timing_enabled and self.device.type == "cuda":
                                torch.cuda.synchronize(self.device)
                            _t1 = time.perf_counter()
                            wav_gen = self._run_bigvgan(cmf_res)
                            audio = wav_gen[0][0]
                            if self._stream_sync_timing_enabled and self.device.type == "cuda":
                                torch.cuda.synchronize(self.device)
                            _bigvgan_ms = (time.perf_counter() - _t1) * 1000.0
                            _cfm_ms = _t_cfm_total * 1000.0 if self._stream_sync_timing_enabled else 0.0
                            if self._stream_sync_timing_enabled:
                                logger.info("[sovits-timing] cfm=%.1fms bigvgan=%.1fms mel_T=%s",
                                            _cfm_ms, _bigvgan_ms, cmf_res.shape[2])

                            # Prevent clipping / 防止爆音
                            max_audio = torch.abs(audio).max()
                            if max_audio > 1:
                                audio = audio / max_audio

                            audio_chunk = self._finalize_stream_chunk(
                                audio,
                                sr,
                                if_sr=if_sr,
                                is_last_chunk=True,
                            )
                            audio_chunk = self._trim_generated_lead(audio_chunk, sr)

                            # Stream current sentence audio + text / 流式返回当前句子音频和文本
                            for _sr, _chunk, _text in _yield_audio_segments(audio_chunk, text_item):
                                yield _sr, _chunk, _text

                        pause_chunk = zero_wav.cpu().detach().numpy()
                        if hasattr(pause_chunk, 'dtype') and 'float16' in str(pause_chunk.dtype):
                            pause_chunk = pause_chunk.astype(np.float32)
                        yield sr, pause_chunk, ""  # No text for pause / 停顿不需要文本

                    from aquatts.inference.streaming import locked_chunks
                    yield from locked_chunks(decode_chunks(), self._sovits_decode_lock)

        except Exception as e:
            logger.error(f"流式推理失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _trim_generated_lead(self, audio, sample_rate):
        """Trim one model-produced item before adding the caller's pause."""
        pcm = audio.detach().float().cpu().numpy() if torch.is_tensor(audio) else audio
        start = leading_cut(pcm, sample_rate)
        return audio[start:] if start else audio

    def _apply_fade_out(self, audio: np.ndarray, sr: int, duration_ms: int = 15) -> np.ndarray:
        """对音频末尾做线性淡出，避免句尾突然截断产生的爆音感。
        duration_ms: 淡出持续时间（毫秒），默认 15ms
        """
        fade_samples = min(int(sr * duration_ms / 1000), len(audio) // 4)
        if fade_samples > 0:
            audio = audio.copy()
            audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        return audio

    def _apply_fade_in(self, audio: np.ndarray, sr: int, duration_ms: int = 8) -> np.ndarray:
        fade_samples = min(int(sr * duration_ms / 1000), len(audio) // 4)
        if fade_samples > 0:
            audio = audio.copy()
            audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        return audio

    def _finalize_stream_chunk(
        self,
        audio_chunk,
        sr: int,
        *,
        if_sr: bool = False,
        is_last_chunk: bool = False,
        apply_fade_in: bool = False,
    ) -> np.ndarray:
        if torch.is_tensor(audio_chunk):
            if if_sr and self.model_version == "v3":
                try:
                    logger.info("执行音频超分...")
                    sr_model = self._get_audio_sr_model()
                    audio_chunk, sr = sr_model(audio_chunk.unsqueeze(0), sr)
                    max_audio = np.abs(audio_chunk).max()
                    if max_audio > 1:
                        audio_chunk = audio_chunk / max_audio
                except Exception as e:
                    logger.warning(f"音频超分失败: {e}")
                    logger.warning(traceback.format_exc())
                    audio_chunk = audio_chunk.cpu().detach().numpy()
            else:
                audio_chunk = audio_chunk.cpu().detach().numpy()

        if hasattr(audio_chunk, 'dtype') and 'float16' in str(audio_chunk.dtype):
            audio_chunk = audio_chunk.astype(np.float32)
        elif getattr(audio_chunk, "dtype", None) != np.float32:
            audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

        if apply_fade_in:
            audio_chunk = self._apply_fade_in(audio_chunk, sr)
        if is_last_chunk:
            audio_chunk = self._apply_fade_out(audio_chunk, sr)
        return audio_chunk

    def _resample(self, audio_tensor, sr0):
        """重采样音频"""
        import torchaudio

        # 确保输入类型与权重类型匹配
        if self.is_half:
            # 如果模型是半精度，则强制将音频转为半精度
            audio_tensor = audio_tensor.half()
            # 创建半精度的重采样器
            resample_fn = torchaudio.transforms.Resample(sr0, 24000).to(self.device).half()
        else:
            # 如果模型是全精度，则强制将音频转为全精度
            audio_tensor = audio_tensor.float()
            # 创建全精度的重采样器
            resample_fn = torchaudio.transforms.Resample(sr0, 24000).to(self.device)

        return resample_fn(audio_tensor)

class DictToAttrRecursive(dict):
    """将字典转换为可属性访问的对象"""

    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def cut1(inp):
    """凑四句一切 - 每四个句子作为一个段落"""
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    """凑50字一切 - 大约每50个字符作为一个段落"""
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # 如果最后一个太短了，和前一个合一起
    if len(opts) > 1 and len(opts[-1]) < 50:
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    """按中文句号切 - 按中文句号'。'分割"""
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut4(inp):
    """按英文句号切 - 按英文句号'.'分割"""
    import re
    inp = inp.strip("\n")
    opts = re.split(r'(?<!\d)\.(?!\d)', inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut5(inp):
    """按标点符号切 - 按各种标点符号分割"""
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def split(todo_text):
    """将文本按标点符号分割成句子列表"""
    splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}

    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def process_text(texts):
    """处理文本，过滤空行并检查是否有有效内容"""
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError("请输入有效文本")
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text


def synthesize(gpt_model_path, sovits_model_path, ref_audio_path, ref_text_path, ref_language,
               target_text_path, target_language, output_path, sample_steps=16, top_p=0.6,
               temperature=0.6, speed=1.0, how_to_cut="不切", if_sr=False, pause_second=0.3,
               sv_model_path=None):
    """
    合成语音的封装函数，符合原始CLI工具的接口

    Args:
        gpt_model_path: GPT模型路径
        sovits_model_path: SoVITS模型路径
        ref_audio_path: 参考音频路径
        ref_text_path: 参考文本路径
        ref_language: 参考文本语言
        target_text_path: 目标文本路径
        target_language: 目标文本语言
        output_path: 输出路径
        sample_steps: 采样步数
        top_p: GPT采样参数
        temperature: GPT采样参数
        speed: 语速控制
        how_to_cut: 文本切分方式
        if_sr: 是否使用音频超分
        pause_second: 句间停顿秒数
        sv_model_path: Pro／Plus 的 ERes2Net 权重路径
    """
    try:
        # 初始化TTS推理器
        inferencer = TTSInferencer(
            gpt_path=gpt_model_path,
            sovits_path=sovits_model_path,
            sv_model_path=sv_model_path,
        )

        # 读取参考文本
        with open(ref_text_path, 'r', encoding='utf-8') as file:
            ref_text = file.read().strip()

        # 读取目标文本
        with open(target_text_path, 'r', encoding='utf-8') as file:
            target_text = file.read().strip()

        # 执行推理
        sampling_rate, audio_data = inferencer.infer(
            text=target_text,
            ref_audio_path=ref_audio_path,
            prompt_text=ref_text,
            text_language=target_language,
            prompt_language=ref_language,
            how_to_cut=how_to_cut,
            top_p=top_p,
            temperature=temperature,
            sample_steps=sample_steps,
            speed=speed,
            if_sr=if_sr,
            pause_second=pause_second
        )

        # 保存结果
        if hasattr(audio_data, 'dtype') and 'float16' in str(audio_data.dtype):
            # 转换为float32
            audio_data = audio_data.astype(np.float32)

            # 保存结果
        os.makedirs(output_path, exist_ok=True)
        output_wav_path = os.path.join(output_path, "output.wav")
        sf.write(output_wav_path, audio_data, sampling_rate)

        logger.info(f"音频已保存到 {output_wav_path}")
        return output_wav_path

    except Exception as e:
        logger.error(f"合成失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def main():
    """命令行入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="GPT-SoVITS TTS 推理工具")
    parser.add_argument('--gpt_model', required=True, help="GPT模型路径")
    parser.add_argument('--sovits_model', required=True, help="SoVITS模型路径")
    parser.add_argument('--sv_model', help="Pro／Plus 的 ERes2Net 权重路径")
    parser.add_argument('--ref_audio', required=True, help="参考音频路径")
    parser.add_argument('--ref_text', required=True, help="参考文本路径")
    parser.add_argument('--ref_language', required=True, choices=["中文", "英文", "日文"], help="参考音频语言")
    parser.add_argument('--target_text', required=True, help="目标文本路径")
    parser.add_argument('--target_language', required=True,
                        choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合", "粤语", "韩文",
                                 "粤英混合", "韩英混合", "多语种混合(粤语)"],
                        help="目标文本语言")
    parser.add_argument('--output_path', required=True, help="输出目录")
    parser.add_argument('--how_to_cut', default="不切",
                        choices=["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"],
                        help="文本切分方式")
    parser.add_argument('--sample_steps', type=int, default=16, help="仅V3模型：采样步数")
    parser.add_argument('--top_k', type=int, default=20, help="GPT采样参数 top_k")
    parser.add_argument('--top_p', type=float, default=0.6, help="GPT采样参数 top_p")
    parser.add_argument('--temperature', type=float, default=0.6, help="GPT采样参数 temperature")
    parser.add_argument('--speed', type=float, default=1.0, help="语速控制")
    parser.add_argument('--pause_second', type=float, default=0.3, help="句间停顿秒数")
    parser.add_argument('--if_sr', action='store_true', help="是否使用音频超分(仅V3模型)")

    args = parser.parse_args()

    synthesize(
        gpt_model_path=args.gpt_model,
        sovits_model_path=args.sovits_model,
        sv_model_path=args.sv_model,
        ref_audio_path=args.ref_audio,
        ref_text_path=args.ref_text,
        ref_language=args.ref_language,
        target_text_path=args.target_text,
        target_language=args.target_language,
        output_path=args.output_path,
        how_to_cut=args.how_to_cut,
        sample_steps=args.sample_steps,
        top_p=args.top_p,
        temperature=args.temperature,
        speed=args.speed,
        if_sr=args.if_sr,
        pause_second=args.pause_second
    )


if __name__ == "__main__":
    main()
