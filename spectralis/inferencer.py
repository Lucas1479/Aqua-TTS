# -*- coding: utf-8 -*-
import os
import sys
import time
import threading
import logging
import tempfile
import traceback
from pathlib import Path
import string
from string import punctuation

import torch
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None  # friendly error in TTSInferencer.__init__

# TTS language config (via TTS_OUTPUT_LANGUAGE env var, defaults to Japanese)
# TTS 语言配置（通过环境变量 TTS_OUTPUT_LANGUAGE 切换，默认日文）
try:
    from config.settings import TTS_OUTPUT_LANGUAGE as _TTS_OUTPUT_LANGUAGE
    from config.settings import TTS_REF_TEXT_JA as _TTS_REF_TEXT_JA
    from config.settings import TTS_REF_TEXT_EN as _TTS_REF_TEXT_EN
except ImportError:
    _TTS_OUTPUT_LANGUAGE = "日文"
    _TTS_REF_TEXT_JA = "こんにちは。今日はいい天気ですね。"
    _TTS_REF_TEXT_EN = ""

def _default_lang_code() -> str:
    """返回当前语言对应的 GPT-SoVITS 内部语言代码（用于 fallback）。"""
    return "en" if _TTS_OUTPUT_LANGUAGE == "英文" else "ja"

def _default_ref_free_prompt() -> str:
    """v3 ref_free 兜底 prompt 文本。"""
    if _TTS_OUTPUT_LANGUAGE == "英文":
        return _TTS_REF_TEXT_EN or "I see, let me think about that."
    return _TTS_REF_TEXT_JA or "こんにちは。今日はいい天気ですね。"


# Set up logging
# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tts_inference')

# Package directory for vendored-file references.
# sys.path is configured by spectralis/__init__.py before this module loads.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

def _gpt_sovits_home() -> str:
    """Return the main GPT-SoVITS repo root, or '' if not configured."""
    return os.environ.get('GPT_SOVITS_HOME', '')


# Import required modules from GPT-SoVITS
# 导入必要的模块
try:
    import librosa
    import re
    from tools.i18n import I18nAuto
    from GPT_SoVITS.text.LangSegmenter import LangSegmenter
    from GPT_SoVITS.text import cleaned_text_to_sequence
    from GPT_SoVITS.text.cleaner import clean_text
    from GPT_SoVITS.module.mel_processing import spectrogram_torch, mel_spectrogram_torch
    from tools.audio_sr import AP_BWE
    from GPT_SoVITS.text import chinese
    from GPT_SoVITS.feature_extractor import cnhubert
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS
    from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule

    """Load SoVITS model
加载SoVITS模型"""
    from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3
    from GPT_SoVITS.process_ckpt import load_sovits_new
    from peft import LoraConfig, get_peft_model

    from GPT_SoVITS.text.LangSegmenter import LangSegmenter
    from GPT_SoVITS.text import cleaned_text_to_sequence
    from GPT_SoVITS.text.cleaner import clean_text

    from GPT_SoVITS.BigVGAN import bigvgan

    from GPT_SoVITS.text import chinese

    from spectralis.inference.presets import (
        apply_cuda_graph_preset as _apply_cg_preset,
        apply_preset as _apply_preset,
    )

except ImportError as e:
    _missing = str(e)
    if "librosa" in _missing:
        logger.error("librosa is required. Install with: pip install spectralis-tts[runtime]")
    elif "peft" in _missing:
        logger.error("peft is required. Install with: pip install spectralis-tts[runtime]")
    elif "GPT_SoVITS" in _missing:
        logger.error(
            "GPT-SoVITS not found. Set GPT_SOVITS_HOME to your GPT-SoVITS repo root."
        )
    else:
        logger.error(f"Import failed: {_missing}")
    raise


class TTSInferencer:
    def __init__(self,
                 device="cuda",
                 gpt_path=None,
                 sovits_path=None,
                 bert_path=None,
                 cnhubert_path=None,
                 language="Auto",
                 cuda_graph_preset="full"):
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
            language: Default language — "Auto", "中文", "英文", "日文", etc.
                      默认语言
            cuda_graph_preset: CUDA Graph capture strategy — "full" (all buckets pre-captured),
                "minimal" (subset), "lazy" (capture on first use), "off" (static KV only)
                CUDA Graph 捕获策略 — "full" (全桶预捕获), "minimal" (最小桶集合),
                "lazy" (惰性捕获), "off" (纯 static KV)
        """
        self._cuda_graph_preset = cuda_graph_preset
        try:
            # 确定模型路径
            base_dir = _gpt_sovits_home() or _PACKAGE_DIR
            self.device = device
            device_name = str(device).lower()
            self.is_half = torch.cuda.is_available() and device_name.startswith("cuda")

            # CUDA device index — used to set the current CUDA device before custom
            # CUDA kernel calls (BigVGAN's fused anti-alias activation), which rely on
            # at::cuda::getCurrentCUDAStream() internally.  Without explicitly setting
            # the device, that function defaults to device 0 even when all tensors live
            # on a different device, causing CUDNN_STATUS_MAPPING_ERROR.
            try:
                self._tts_device_idx = int(device_name.split(":")[-1])
            except (ValueError, IndexError):
                self._tts_device_idx = 0

            # 默认模型路径
            default_gpt_path = os.path.join(base_dir, "GPT_weights_v3", "xxx-e15.ckpt")
            default_sovits_path = os.path.join(base_dir, "SoVITS_weights_v3", "xxx_e2_s174_l32.pth")
            default_sovits_pretrain_path = os.path.join(base_dir, "GPT_SoVITS", "pretrained_models", "s2Gv3.pth")
            default_bert_path = os.path.join(base_dir, "GPT_SoVITS", "pretrained_models",
                                             "chinese-roberta-wwm-ext-large")
            default_cnhubert_path = os.path.join(base_dir, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base")

            # 使用传入参数或默认路径
            self.gpt_path = gpt_path or default_gpt_path
            self.sovits_path = sovits_path or default_sovits_path
            self.bert_path = bert_path or default_bert_path
            self.cnhubert_path = cnhubert_path or default_cnhubert_path
            self.sovits_pretrain_path = default_sovits_pretrain_path

            # 检查必要文件是否存在
            for path, desc in [
                (self.gpt_path, "GPT权重"),
                (self.sovits_path, "SoVITS权重"),
                (self.sovits_pretrain_path, "SoVITS预训练权重"),
                (self.bert_path, "BERT模型"),
                (self.cnhubert_path, "CNHuBERT模型")
            ]:
                if not os.path.exists(path):
                    logger.warning(f"必要文件不存在: {path} ({desc})，请确保路径正确")

            # 初始化国际化
            self.i18n = I18nAuto(language=language)

            # 初始化语言字典
            self._init_language_dict()

            # 初始化BERT和SSL模型
            self._init_bert_model()
            self._init_ssl_model()

            # 加载GPT和SoVITS模型
            self._load_models()

            logger.info(f"⭐️ TTS推理器初始化完成")

            # 会话级缓存：按(ref_audio_path, prompt_text, prompt_language_code, model_version, is_half)键控
            self._session_cache = {}
            self._sovits_decode_lock = threading.Lock()
            # 仅在需要做性能剖析时开启；默认关闭以避免流式首句每块都强制同步 GPU。
            self._stream_sync_timing_enabled = os.environ.get("TTS_STREAM_SYNC_TIMING", "0") == "1"

        except Exception as e:
            logger.error(f"❌ 初始化TTS推理器失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise

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

    def _init_language_dict(self):
        """Initialize language dictionary mappings.
        初始化语言字典"""
        # Detect model version to select the correct language dictionary
        # 检测模型版本，根据版本确定语言字典
        self.model_version = self._detect_model_version()

        dict_language_v1 = {
            self.i18n("中文"): "all_zh",
            self.i18n("英文"): "en",
            self.i18n("日文"): "all_ja",
            self.i18n("中英混合"): "zh",
            self.i18n("日英混合"): "ja",
            self.i18n("多语种混合"): "auto",
        }

        dict_language_v2 = {
            self.i18n("中文"): "all_zh",
            self.i18n("英文"): "en",
            self.i18n("日文"): "all_ja",
            self.i18n("粤语"): "all_yue",
            self.i18n("韩文"): "all_ko",
            self.i18n("中英混合"): "zh",
            self.i18n("日英混合"): "ja",
            self.i18n("粤英混合"): "yue",
            self.i18n("韩英混合"): "ko",
            self.i18n("多语种混合"): "auto",
            self.i18n("多语种混合(粤语)"): "auto_yue",
        }

        self.dict_language = dict_language_v2 if self.model_version in ["v2", "v3"] else dict_language_v1
        self.splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}

    def _detect_model_version(self):
        """Detect model version from checkpoint path or filename.
        检测模型版本"""
        # 简单版本检测，可根据文件名或其他特征判断
        if "v3" in self.sovits_path or "v3" in self.gpt_path:
            return "v3"
        elif "v2" in self.sovits_path or "v2" in self.gpt_path:
            return "v2"
        else:
            return "v1"

    def _init_bert_model(self):
        """Initialize BERT model for Chinese text encoding.
        初始化BERT模型"""
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        logger.info(f"加载BERT模型: {self.bert_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(self.bert_path)

        if self.is_half:
            self.bert_model = self.bert_model.half().to(self.device)
        else:
            self.bert_model = self.bert_model.to(self.device)

    def _init_ssl_model(self):
        """Initialize CNHuBERT SSL model for semantic feature extraction.
        初始化SSL模型"""


        logger.info(f"加载CNHubert模型: {self.cnhubert_path}")
        cnhubert.cnhubert_base_path = self.cnhubert_path
        self.ssl_model = cnhubert.get_model()

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

        self._maybe_precapture_t2s_graph()

    def _warmup_bigvgan_shapes(self):
        """Warm BigVGAN + cuDNN for common streaming chunk mel_T sizes.

        Without this, the first real inference pays cuDNN autotune cost
        (~30-60ms per new shape).  A few warmup forward passes at common
        chunk sizes (mel_T ≈ 40, 70, 128) settle the autotune cache so that
        the first real stream chunk hits a warm kernel.
        """
        if self.bigvgan_model is None or str(self.device) == "cpu":
            return
        try:
            model = self.bigvgan_model
            dtype = next(model.parameters()).dtype
            device = next(model.parameters()).device
            num_mels = model.h.num_mels
            warmup_sizes = [40, 70, 128]
            logger.info(f"[warmup] BigVGAN shape warmup: mel_T={warmup_sizes}")
            for mel_t in warmup_sizes:
                mel = torch.randn(1, num_mels, mel_t, device=device, dtype=dtype)
                for _ in range(5):
                    _ = model(mel)
                torch.cuda.synchronize()
            logger.info("[warmup] BigVGAN shape warmup complete")
        except Exception:
            pass  # warmup is best-effort; never block startup

    def _maybe_precapture_t2s_graph(self):
        """根据 CUDA Graph preset 预捕获 T2S 阶段的 CUDA Graph。

        环境变量（ENABLE_CUDA_GRAPH, ENABLE_CUDA_GRAPH_PRECAPTURE,
        CUDA_GRAPH_PRECAPTURE_BUCKETS）可作为 preset 的 override。
        """
        try:
            # Resolve CUDA Graph preset (env vars override the named preset)
            cg = _apply_cg_preset(self._cuda_graph_preset)

            # Env var overrides (preserve backward compat)
            if os.environ.get("ENABLE_CUDA_GRAPH", "").strip() in ("0", "1"):
                cg["enable"] = os.environ["ENABLE_CUDA_GRAPH"] == "1"
            if os.environ.get("ENABLE_CUDA_GRAPH_PRECAPTURE", "").strip() in ("0", "1"):
                cg["precapture"] = os.environ["ENABLE_CUDA_GRAPH_PRECAPTURE"] == "1"
            bucket_env = os.environ.get("CUDA_GRAPH_PRECAPTURE_BUCKETS", "")
            if bucket_env.strip():
                buckets = []
                for token in bucket_env.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    try:
                        buckets.append(int(token))
                    except ValueError:
                        logger.warning(f"跳过非法桶配置: {token}")
                cg["buckets"] = buckets

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
        if 'enc_p.text_embedding.weight' not in dict_s2['weight']:
            self.hps.model.version = "v2"  # v3model,v2symbols
        elif dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            self.hps.model.version = "v1"
        else:
            self.hps.model.version = "v2"

        self.sovits_version = self.hps.model.version
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
            self.vq_model = SynthesizerTrnV3(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model
            )

        # 处理预训练模型
        if "pretrained" not in self.sovits_path:
            try:
                del self.vq_model.enc_q
            except:
                pass

        # 转换模型类型并加载到设备
        if self.is_half:
            self.vq_model = self.vq_model.half().to(self.device)
        else:
            self.vq_model = self.vq_model.to(self.device)

        self.vq_model.eval()

        # 检查是否是LoRA模型
        # 根据后缀或文件大小判断
        self.if_lora_v3 = False
        if self.model_version == "v3" and ".pth" in self.sovits_path.lower():
            file_size = os.path.getsize(self.sovits_path) / (1024 * 1024)  # MB
            if file_size < 100:  # 假设小于100MB的是LoRA权重
                self.if_lora_v3 = True
                logger.info(f"检测到LoRA模型: {self.sovits_path}")

        # 加载权重
        if not self.if_lora_v3:
            logger.info(f"加载sovits_{self.model_version}模型权重")
            self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        else:
            # 加载预训练模型和LoRA权重
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

    def _load_bigvgan_model(self):
        """Load BigVGAN vocoder model (required for v3).
        加载BigVGAN模型（v3模型需要）"""
        if self.model_version != "v3":
            self.bigvgan_model = None
            return

        try:


            bigvgan_path = os.path.join(_gpt_sovits_home(), "GPT_SoVITS", "pretrained_models",
                                        "models--nvidia--bigvgan_v2_24khz_100band_256x")
            logger.info(f"加载BigVGAN模型: {bigvgan_path}")

            # use_cuda_kernel=True：编译 anti-aliased activation 的融合 CUDA kernel，
            # 减少内存读写，对 CUDA Graph 之后的 L2 cache miss 不敏感。
            # 优先检查预编译 .pyd 缓存（无需 cl.exe/nvcc），否则尝试现场编译。
            import pathlib as _pathlib
            # Use the device index that was resolved at __init__ time.
            _tts_device_idx = self._tts_device_idx
            _cache_id = os.environ.get("BIGVGAN_CACHE_ID", "").strip()
            if _cache_id:
                import re as _re
                _cache_suffix = _re.sub(r"[^A-Za-z0-9_.-]+", "_", _cache_id.lower()).strip("_") or "unknown"
            elif torch.cuda.is_available():
                try:
                    import re as _re
                    _props = torch.cuda.get_device_properties(_tts_device_idx)
                    _name = _re.sub(r"[^A-Za-z0-9_.-]+", "_", _props.name.lower()).strip("_") or "unknown"
                    _mem_gb = int(round(_props.total_memory / (1024 ** 3)))
                    _cache_suffix = f"sm{_props.major}{_props.minor}_{_mem_gb}gb_{_name}"
                except Exception:
                    _cache_suffix = f"device{_tts_device_idx}"
            else:
                _cache_suffix = f"device{_tts_device_idx}"
            _cuda_pyd = (
                _pathlib.Path(_PACKAGE_DIR)
                / "_vendor/GPT_SoVITS/BigVGAN/alias_free_activation/cuda"
                / f"build_{_cache_suffix}"
                / "anti_alias_activation_cuda.pyd"
            )
            _use_cuda_kernel = False
            if _cuda_pyd.exists():
                _use_cuda_kernel = True
                logger.info(f"[BigVGAN] 检测到已编译 CUDA kernel 缓存，直接加载")
            else:
                try:
                    import subprocess as _sp
                    _nvcc = _sp.run(["nvcc", "--version"], capture_output=True, timeout=5)
                    if _nvcc.returncode == 0:
                        _use_cuda_kernel = True
                        logger.info("[BigVGAN] nvcc 可用，尝试编译 CUDA kernel 加速")
                except Exception:
                    logger.info("[BigVGAN] nvcc 不可用，使用 PyTorch 实现")

            kernel_override = os.environ.get("BIGVGAN_USE_CUDA_KERNEL", "").strip().lower()
            if kernel_override in {"0", "false", "off", "no"}:
                _use_cuda_kernel = False
                logger.info("[BigVGAN] BIGVGAN_USE_CUDA_KERNEL=0, forcing PyTorch path")
            elif kernel_override in {"1", "true", "on", "yes"}:
                _use_cuda_kernel = True
                logger.info("[BigVGAN] BIGVGAN_USE_CUDA_KERNEL=1, forcing CUDA kernel path")

            # activation1d.py 在首次 import 时执行模块级 load.load()，
            # 若此时无设备上下文则 CUDA kernel 内部状态绑定到 cuda:0，
            # 之后模型移到 cuda:1 会触发 CUDNN_STATUS_MAPPING_ERROR。
            # 用 torch.cuda.device() 确保 kernel 初始化在正确设备上进行。
            try:
                with torch.cuda.device(_tts_device_idx):
                    self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_path, use_cuda_kernel=_use_cuda_kernel)
            except Exception as _kernel_err:
                if _use_cuda_kernel:
                    logger.warning(f"[BigVGAN] CUDA kernel 编译失败，回退到 PyTorch 实现: {_kernel_err}")
                    with torch.cuda.device(_tts_device_idx):
                        self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_path, use_cuda_kernel=False)
                else:
                    raise
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

    def _build_session_cache(self, ref_audio_path: str, prompt_text: str, prompt_language_code: str):
        """Build a session-level reference cache (SSL, phones, BERT, mel, prompt features).
        构建会话级参考缓存。"""
        key = (ref_audio_path, prompt_text or "", prompt_language_code or "", self.model_version, self.is_half)
        if key in self._session_cache:
            return self._session_cache[key]

        cache_item = {}
        try:
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

                mel_fn = lambda x: mel_spectrogram_torch(x, **{
                    "n_fft": 1024,
                    "win_size": 1024,
                    "hop_size": 256,
                    "num_mels": 100,
                    "sampling_rate": 24000,
                    "fmin": 0,
                    "fmax": None,
                    "center": False
                })
                spec_min, spec_max = -12, 2
                norm_spec = lambda x: (x - spec_min) / (spec_max - spec_min) * 2 - 1
                mel2 = mel_fn(ref_audio)
                mel2 = norm_spec(mel2)
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

            self._session_cache[key] = cache_item
            return cache_item
        except Exception:
            logger.warning("构建会话缓存失败，回退到逐句计算。")
            logger.warning(traceback.format_exc())
            return {}

    def _clone_cached_value(self, value):
        """Clone cached tensors to avoid mutation by downstream inference paths.
        避免读取会话缓存后被后续推理路径原地复用/污染。"""
        if torch.is_tensor(value):
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
                for tmp in LangSegmenter.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegmenter.getTexts(text):
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

    def _audio_sr(self, audio, sr):
        """Audio super-resolution processing.
        音频超分辨率处理"""
        try:

            sr_model = AP_BWE(self.device, DictToAttrRecursive)
            return sr_model(audio, sr)
        except ImportError:
            logger.warning("未找到音频超分模块，跳过超分处理")
            return audio.cpu().detach().numpy(), sr
        except FileNotFoundError:
            logger.warning("未找到音频超分模型参数，跳过超分处理")
            return audio.cpu().detach().numpy(), sr

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
              enable_cuda_graph=False,
              enable_static_kv=True,
              max_sec_override=None):
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
            spec_min, spec_max = -12, 2
            denorm_spec = lambda x: (x + 1) / 2 * (spec_max - spec_min) + spec_min

            # Convert language code
            # 转换语言代码
            if text_language in self.dict_language:
                text_language_code = self.dict_language[text_language]
            else:
                text_language_code = _default_lang_code()
                logger.warning(f"未知语言: {text_language}，使用默认语言: {_TTS_OUTPUT_LANGUAGE}")

            # Use ref_free mode if no prompt text is provided
            # 如果没有提供参考文本，则使用无参考模式
            if prompt_text is None or prompt_text.strip() == "":
                ref_free = True
                logger.info("未提供参考文本，使用无参考模式")
            else:
                prompt_text = prompt_text.strip()
                # Ensure prompt text ends with punctuation
                # 确保参考文本以标点符号结尾
                if prompt_text and prompt_text[-1] not in self.splits:
                    prompt_text += "。" if prompt_language != "英文" else "."

                if prompt_language in self.dict_language:
                    prompt_language_code = self.dict_language[prompt_language]
                else:
                    prompt_language_code = _default_lang_code()

                logger.info(f"参考文本: '{prompt_text}'")

            # v3 doesn't support ref_free mode
            # v3模型不支持ref_free模式
            if self.model_version == "v3" and ref_free:
                logger.warning("v3模型不支持无参考模式，强制使用有参考模式")
                ref_free = False

                # Use default prompt text for the current language
                # 如果没有参考文本，使用当前语言的默认文本
                if not prompt_text:
                    prompt_text = _default_ref_free_prompt()
                    prompt_language_code = "en" if _TTS_OUTPUT_LANGUAGE == "英文" else "all_ja"

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
            sess_lang = prompt_language_code if not ref_free else _default_lang_code()
            sess = self._build_session_cache(ref_audio_path, prompt_text, sess_lang)
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
                        pred_semantic, idx = self.t2s_model.model.infer_panel(
                            all_phoneme_ids,
                            all_phoneme_len,
                            None if ref_free else prompt,
                            bert,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            early_stop_num=int(self.hz * effective_max_sec),
                            enable_cuda_graph=enable_cuda_graph,
                            enable_static_kv=enable_static_kv,
                        )
                        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                        cache[i_text] = pred_semantic

                # Run SoVITS decoding
                # SoVITS推理
                logger.info("执行SoVITS解码...")

                if self.model_version != "v3":
                    # Decode for v1/v2 models
                    # v1/v2模型解码
                    # Process multiple reference audios
                    # 处理多个参考音频
                    refers = []
                    if inp_refs:
                        for ref_path in inp_refs:
                            try:
                                ref_path = ref_path if isinstance(ref_path, str) else ref_path.name
                                # 根据is_half决定是否使用half
                                refer = self.get_spepc(ref_path).to(self.device)
                                if self.is_half:
                                    refer = refer.half()
                                else:
                                    refer = refer.float()
                                refers.append(refer)
                                logger.info(f"加载额外参考音频: {ref_path}")
                            except Exception as e:
                                logger.warning(f"加载额外参考音频失败: {e}")

                    # Fall back to main reference audio if no extra refs
                    # 如果没有额外参考音频，使用主参考音频
                    if len(refers) == 0:
                        refer = sess.get("refer_spec")
                        if refer is None:
                            refer = self.get_spepc(ref_audio_path).to(self.device)
                            refer = refer.half() if self.is_half else refer.float()
                        refers = [refer]

                    # Decode
                    # 解码
                    audio = self.vq_model.decode(
                        pred_semantic,
                        torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                        refers,
                        speed=speed
                    )[0][0]

                    # Prevent audio clipping
                    # 防止爆音
                    max_audio = torch.abs(audio).max()
                    if max_audio > 1:
                        audio = audio / max_audio

                    # Add to output list
                    # 添加到输出列表
                    audio_outputs.append(audio)
                    audio_outputs.append(zero_wav)  # Inter-sentence pause / 句间停顿

                else:
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
                    ref_audio, ref_sr = torchaudio.load(ref_audio_path)
                    ref_audio = ref_audio.to(self.device)
                    # 根据is_half决定是否使用half
                    if self.is_half:
                        ref_audio = ref_audio.half()
                    else:
                        ref_audio = ref_audio.float()

                    if ref_audio.shape[0] == 2:  # Convert to mono / 转单声道
                        ref_audio = ref_audio.mean(0).unsqueeze(0)

                    # Resample to 24 kHz
                    # 重采样到24kHz
                    if ref_sr != 24000:
                        ref_audio = self._resample(ref_audio, ref_sr)

                    # Extract mel features (prefer session cache)
                    # 提取mel特征（优先使用会话缓存）
                    mel2 = sess.get("mel2_norm")
                    if mel2 is None:
                        mel_fn = lambda x: mel_spectrogram_torch(x, **{
                            "n_fft": 1024,
                            "win_size": 1024,
                            "hop_size": 256,
                            "num_mels": 100,
                            "sampling_rate": 24000,
                            "fmin": 0,
                            "fmax": None,
                            "center": False
                        })
                        spec_min, spec_max = -12, 2
                        norm_spec = lambda x: (x - spec_min) / (spec_max - spec_min) * 2 - 1
                        mel2 = mel_fn(ref_audio)
                        mel2 = norm_spec(mel2)

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
                        cfm_res = self.vq_model.cfm.inference(
                            fea,
                            torch.LongTensor([fea.size(1)]).to(fea.device),
                            mel2,
                            sample_steps,
                            inference_cfg_rate=0
                        )

                        cfm_res = cfm_res[:, :, mel2.shape[2]:]
                        mel2 = cfm_res[:, :, -T_min:]
                        fea_ref = fea_todo_chunk[:, :, -T_min:]
                        cfm_resss.append(cfm_res)

                    # Concatenate chunk results
                    # 合并结果
                    cmf_res = torch.cat(cfm_resss, 2)
                    cmf_res = denorm_spec(cmf_res)

                    # BigVGAN generates waveform / BigVGAN生成波形
                    # torch.cuda.device() ensures at::cuda::getCurrentCUDAStream()
                    # inside the fused CUDA kernel uses the correct device stream,
                    # preventing CUDNN_STATUS_MAPPING_ERROR on non-default GPUs.
                    with torch.cuda.device(self._tts_device_idx):
                        with torch.inference_mode():
                            wav_gen = self.bigvgan_model(cmf_res)
                            audio = wav_gen[0][0]

                    # Prevent audio clipping
                    # 防止爆音
                    max_audio = torch.abs(audio).max()
                    if max_audio > 1:
                        audio = audio / max_audio

                    # Add to output list
                    # 添加到输出列表
                    audio_outputs.append(audio)
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
                        if not hasattr(self, 'sr_model') or self.sr_model is None:
                            self.sr_model = AP_BWE(self.device, DictToAttrRecursive)

                        # Run super-resolution
                        # 进行音频超分
                        final_audio, sr = self.sr_model(final_audio.unsqueeze(0), sr)

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
                     enable_cuda_graph=False,
                     enable_static_kv=True,
                     chunk_size_seconds: float = None,
                     max_sec_override: float = None):
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

        Yields:
            Generator of (sample_rate, audio_chunk, text_segment) tuples.
            生成器：每次生成 (采样率, 音频数据片段, 文本段落)
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
            spec_min, spec_max = -12, 2
            denorm_spec = lambda x: (x + 1) / 2 * (spec_max - spec_min) + spec_min

            # Convert language code
            # 转换语言代码
            if text_language in self.dict_language:
                text_language_code = self.dict_language[text_language]
            else:
                text_language_code = _default_lang_code()
                logger.warning(f"未知语言: {text_language}，使用默认语言: {_TTS_OUTPUT_LANGUAGE}")

            # Use ref_free mode if no prompt text is provided
            # 如果没有提供参考文本，则使用无参考模式
            if prompt_text is None or prompt_text.strip() == "":
                ref_free = True
                logger.info("未提供参考文本，使用无参考模式")
            else:
                prompt_text = prompt_text.strip()
                # Ensure prompt text ends with punctuation
                # 确保参考文本以标点符号结尾
                if prompt_text and prompt_text[-1] not in self.splits:
                    prompt_text += "。" if prompt_language != "英文" else "."

                if prompt_language in self.dict_language:
                    prompt_language_code = self.dict_language[prompt_language]
                else:
                    prompt_language_code = _default_lang_code()

                logger.info(f"参考文本: '{prompt_text}'")

            # v3 doesn't support ref_free mode
            # v3模型不支持ref_free模式
            if self.model_version == "v3" and ref_free:
                logger.warning("v3模型不支持无参考模式，强制使用有参考模式")
                ref_free = False

                # Use default prompt text for the current language
                # 如果没有参考文本，使用当前语言的默认文本
                if not prompt_text:
                    prompt_text = _default_ref_free_prompt()
                    prompt_language_code = "en" if _TTS_OUTPUT_LANGUAGE == "英文" else "all_ja"

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
            sess_lang = prompt_language_code if not ref_free else _default_lang_code()
            sess = self._build_session_cache(ref_audio_path, prompt_text, sess_lang)
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
                        pred_semantic, idx = self.t2s_model.model.infer_panel(
                            all_phoneme_ids,
                            all_phoneme_len,
                            None if ref_free else prompt,
                            bert,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            early_stop_num=int(self.hz * effective_max_sec),
                            enable_cuda_graph=enable_cuda_graph,
                            enable_static_kv=enable_static_kv,
                        )
                        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                        cache[i_text] = pred_semantic

                # Run SoVITS decoding
                # SoVITS推理
                logger.info("执行SoVITS解码...")

                if self.model_version != "v3":
                    # Decode for v1/v2 models
                    # v1/v2模型解码
                    # Process multiple reference audios
                    # 处理多个参考音频
                    refers = []
                    if inp_refs:
                        for ref_path in inp_refs:
                            try:
                                ref_path = ref_path if isinstance(ref_path, str) else ref_path.name
                                # 根据is_half决定是否使用half
                                refer = self.get_spepc(ref_path).to(self.device)
                                if self.is_half:
                                    refer = refer.half()
                                else:
                                    refer = refer.float()
                                refers.append(refer)
                                logger.info(f"加载额外参考音频: {ref_path}")
                            except Exception as e:
                                logger.warning(f"加载额外参考音频失败: {e}")

                    # Fall back to main reference audio if no extra refs
                    # 如果没有额外参考音频，使用主参考音频
                    if len(refers) == 0:
                        refer = sess.get("refer_spec")
                        if refer is None:
                            refer = self.get_spepc(ref_audio_path).to(self.device)
                            refer = refer.half() if self.is_half else refer.float()
                        refers = [refer]

                    # Decode
                    # 解码
                    audio = self.vq_model.decode(
                        pred_semantic,
                        torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                        refers,
                        speed=speed
                    )[0][0]

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

                    with self._sovits_decode_lock:
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
                        spec_min, spec_max = -12, 2
                        denorm_spec = lambda x: (x + 1) / 2 * (spec_max - spec_min) + spec_min
                        if mel2 is None:
                            ref_audio, ref_sr = torchaudio.load(ref_audio_path)
                            ref_audio = ref_audio.to(self.device)
                            ref_audio = ref_audio.half() if self.is_half else ref_audio.float()
                            if ref_audio.shape[0] == 2:  # Convert to mono / 转单声道
                                ref_audio = ref_audio.mean(0).unsqueeze(0)
                            if ref_sr != 24000:
                                ref_audio = self._resample(ref_audio, ref_sr)
                            mel_fn = lambda x: mel_spectrogram_torch(x, **{
                                "n_fft": 1024,
                                "win_size": 1024,
                                "hop_size": 256,
                                "num_mels": 100,
                                "sampling_rate": 24000,
                                "fmin": 0,
                                "fmax": None,
                                "center": False
                            })
                            norm_spec = lambda x: (x - spec_min) / (spec_max - spec_min) * 2 - 1
                            mel2 = mel_fn(ref_audio)
                            mel2 = norm_spec(mel2)

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
                        while True:
                            chunk_end = min(total_todo_frames, idx + chunk_len)
                            fea_todo_chunk = fea_todo[:, :, idx:chunk_end]
                            if fea_todo_chunk.shape[-1] == 0:
                                break

                            idx = chunk_end
                            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)

                            # Run CFM inference / CFM推理
                            _t0 = time.perf_counter()
                            cfm_res = self.vq_model.cfm.inference(
                                fea,
                                torch.LongTensor([fea.size(1)]).to(fea.device),
                                mel2,
                                sample_steps,
                                inference_cfg_rate=0
                            )
                            if self._stream_sync_timing_enabled and str(self.device) != "cpu":
                                torch.cuda.synchronize()
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
                                chunk_mel = denorm_spec(cfm_res)
                                _t1 = time.perf_counter()
                                with torch.cuda.device(self._tts_device_idx):
                                    with torch.inference_mode():
                                        wav_gen = self.bigvgan_model(chunk_mel)
                                        audio = wav_gen[0][0]
                                if self._stream_sync_timing_enabled and str(self.device) != "cpu":
                                    torch.cuda.synchronize()
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
                                yield sr, audio_chunk, text_item if stream_chunk_index == 1 else ""
                            else:
                                cfm_resss.append(cfm_res)

                        if not stream_v3_chunks:
                            cmf_res = torch.cat(cfm_resss, 2)
                            cmf_res = denorm_spec(cmf_res)

                            # Drain prior async CFM work before timing BigVGAN,
                            # otherwise the synchronize below would count CFM+BigVGAN together.
                            if str(self.device) != "cpu":
                                torch.cuda.synchronize()
                            _t1 = time.perf_counter()
                            with torch.cuda.device(self._tts_device_idx):
                                with torch.inference_mode():
                                    wav_gen = self.bigvgan_model(cmf_res)
                                    audio = wav_gen[0][0]
                            if str(self.device) != "cpu":
                                torch.cuda.synchronize()
                            _bigvgan_ms = (time.perf_counter() - _t1) * 1000.0
                            _cfm_ms = _t_cfm_total * 1000.0 if self._stream_sync_timing_enabled else 0.0
                            print("[sovits-timing] cfm=%.1fms  bigvgan=%.1fms  mel_T=%s" % (
                                _cfm_ms, _bigvgan_ms, cmf_res.shape[2],
                            ))

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

                            # Stream current sentence audio + text / 流式返回当前句子音频和文本
                            for _sr, _chunk, _text in _yield_audio_segments(audio_chunk, text_item):
                                yield _sr, _chunk, _text

                        pause_chunk = zero_wav.cpu().detach().numpy()
                        if hasattr(pause_chunk, 'dtype') and 'float16' in str(pause_chunk.dtype):
                            pause_chunk = pause_chunk.astype(np.float32)
                        yield sr, pause_chunk, ""  # No text for pause / 停顿不需要文本

        except Exception as e:
            logger.error(f"流式推理失败: {str(e)}")
            logger.error(traceback.format_exc())
            # 返回一个空音频块，避免生成器中断
            yield sr if 'sr' in locals() else 24000, np.zeros(16000, dtype=np.float32), ""

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
                    if not hasattr(self, 'sr_model') or self.sr_model is None:
                        self.sr_model = AP_BWE(self.device, DictToAttrRecursive)
                    audio_chunk, sr = self.sr_model(audio_chunk.unsqueeze(0), sr)
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
    import re
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
    punctuation = set(['!', '?', '…', ',', '.', '-', " "])

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
               temperature=0.6, speed=1.0, how_to_cut="不切", if_sr=False, pause_second=0.3):
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
    """
    try:
        # 初始化TTS推理器
        inferencer = TTSInferencer(
            gpt_path=gpt_model_path,
            sovits_path=sovits_model_path
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
