# ComfyUI-QwenASR
# ComfyUI custom nodes for Qwen3-ASR speech-to-text models.
# Models License Notice:
# - Qwen3-ASR: Apache-2.0 License (https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
# This integration script follows GPL-3.0 License.

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import folder_paths
import comfy.model_management as model_management

_CURRENT_DIR = Path(__file__).parent
_QWEN_ASR_DIR = _CURRENT_DIR / "qwen_asr"
if _QWEN_ASR_DIR.exists() and str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

try:
    from qwen_asr import Qwen3ASRModel
except Exception as _e:
    Qwen3ASRModel = None
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None

# ComfyUI model folder registration
QWEN3_ASR_ROOT = os.path.join(folder_paths.models_dir, "Qwen3-ASR")
os.makedirs(QWEN3_ASR_ROOT, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-ASR", QWEN3_ASR_ROOT)

SUPPORTED_LANGUAGES = [
    "auto",
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai",
    "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish",
    "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek",
    "Hungarian", "Macedonian", "Romanian",
]

_ASR_MODEL_CACHE = {}


def _is_model_dir(path: str) -> bool:
    """Check if a directory looks like it contains model files."""
    try:
        for f in os.listdir(path):
            if f == "config.json" or f.endswith((".safetensors", ".bin")):
                return True
    except Exception:
        pass
    return False


_DIRECT_MODEL_LABEL = "Qwen3-ASR"


def _scan_local_models():
    """Scan the Qwen3-ASR model folder for locally available model directories.

    If model files live directly inside QWEN3_ASR_ROOT (no sub-folder per model),
    return a special label so the dropdown still shows a usable entry.
    """
    found = []
    try:
        # Check if the root itself IS a model directory (files placed directly)
        if _is_model_dir(QWEN3_ASR_ROOT):
            found.append(_DIRECT_MODEL_LABEL)

        # Also scan for sub-directories (the traditional layout)
        for entry in os.scandir(QWEN3_ASR_ROOT):
            if entry.is_dir() and not entry.name.startswith("."):
                if _is_model_dir(entry.path):
                    found.append(entry.name)
    except Exception as e:
        print(f"[Qwen3ASR] Failed to scan model folder: {e}")
    found.sort()
    return found


def _get_model_choices():
    """Return list of ASR model folder names for the dropdown."""
    models = [m for m in _scan_local_models() if "Aligner" not in m]
    if not models:
        return ["(no models found – place models in models/Qwen3-ASR/)"]
    return models


def _get_aligner_choices():
    """Return list of aligner folder names for the dropdown, with 'None' option."""
    aligners = ["None"] + [m for m in _scan_local_models() if "Aligner" in m]
    return aligners


def _resolve_local_model(model_name: str) -> str:
    """Resolve a model folder name to its full path."""
    # If the user selected the direct-root entry, use QWEN3_ASR_ROOT itself
    if model_name == _DIRECT_MODEL_LABEL and _is_model_dir(QWEN3_ASR_ROOT):
        return QWEN3_ASR_ROOT

    path = os.path.join(QWEN3_ASR_ROOT, model_name)
    if os.path.isdir(path) and _is_model_dir(path):
        return path
    raise FileNotFoundError(
        f"Model '{model_name}' not found in {QWEN3_ASR_ROOT}. "
        f"Please download the model and place it in that folder."
    )

def _normalize_audio(audio) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None

    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or sample_rate is None:
        return None

    wave = waveform[0]
    if wave.ndim == 2 and wave.shape[0] > 1:
        wave = torch.mean(wave, dim=0)
    else:
        wave = wave.squeeze(0)

    return (wave.detach().cpu().numpy().astype(np.float32), int(sample_rate))


def _build_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        if device.type == "mps":
            return torch.float16
        return torch.bfloat16
    return torch.float32


def _cache_key(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attention: str,
    forced_aligner_path: str,
    max_inference_batch_size: int,
    max_new_tokens: int,
) -> tuple:
    return (
        model_path,
        str(dtype),
        str(device),
        attention,
        forced_aligner_path or "",
        int(max_inference_batch_size),
        int(max_new_tokens),
    )


def _load_cached_model(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attention: str,
    forced_aligner_path: str,
    max_inference_batch_size: int = 32,
    max_new_tokens: int = 256,
):
    key = _cache_key(model_path, dtype, device, attention, forced_aligner_path, max_inference_batch_size, max_new_tokens)
    cached = _ASR_MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    model_kwargs = {
        "dtype": dtype,
        "device_map": str(device),
        "max_inference_batch_size": int(max_inference_batch_size),
        "max_new_tokens": int(max_new_tokens),
    }
    if attention != "auto":
        model_kwargs["attn_implementation"] = attention
    if forced_aligner_path:
        model_kwargs["forced_aligner"] = forced_aligner_path
        model_kwargs["forced_aligner_kwargs"] = {
            "dtype": dtype,
            "device_map": str(device),
        }
        if attention != "auto":
            model_kwargs["forced_aligner_kwargs"]["attn_implementation"] = attention

    model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)
    _ASR_MODEL_CACHE[key] = model
    return model


def _format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_srt(time_stamps) -> str:
    if not time_stamps:
        return ""
    lines = []
    for idx, item in enumerate(time_stamps, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_srt_time(item.start_time)} --> {_format_srt_time(item.end_time)}")
        lines.append(item.text or "")
        lines.append("")
    return "\n".join(lines).strip()


def _join_tokens(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    # If either side contains CJK, join without space.
    for ch in (a[-1], b[0]):
        if "\u4e00" <= ch <= "\u9fff":
            return f"{a}{b}"
    return f"{a} {b}"


def _group_time_stamps(time_stamps, max_gap_sec: float, max_chars: int, split_mode: str):
    if not time_stamps:
        return []
    groups = []
    cur = None
    punct = ("。", "！", "？", ".", "!", "?")
    for item in time_stamps:
        text = (item.text or "").strip()
        if not text:
            continue
        if cur is None:
            cur = {
                "start": item.start_time,
                "end": item.end_time,
                "text": text,
            }
            continue

        gap = float(item.start_time) - float(cur["end"])
        too_far = gap > max_gap_sec
        too_long = max_chars > 0 and (len(cur["text"]) + len(text)) > max_chars
        end_sentence = any(cur["text"].endswith(p) for p in punct)

        split_by_punct = split_mode in ("split_by_punctuation", "split_by_punctuation_or_length", "split_by_punctuation_or_pause", "split_by_punctuation_or_pause_or_length")
        split_by_length = split_mode in ("split_by_length", "split_by_punctuation_or_length", "split_by_punctuation_or_pause_or_length")
        split_by_pause = split_mode in ("split_by_pause", "split_by_punctuation_or_pause", "split_by_punctuation_or_pause_or_length")

        should_split = False
        if split_by_punct and end_sentence:
            should_split = True
        if split_by_length and too_long:
            should_split = True
        if split_by_pause and too_far:
            should_split = True

        if should_split:
            groups.append(cur)
            cur = {
                "start": item.start_time,
                "end": item.end_time,
                "text": text,
            }
        else:
            cur["text"] = _join_tokens(cur["text"], text).strip()
            cur["end"] = item.end_time

    if cur is not None:
        groups.append(cur)
    return groups


def _build_srt_from_groups(groups) -> str:
    if not groups:
        return ""
    lines = []
    for idx, g in enumerate(groups, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_srt_time(g['start'])} --> {_format_srt_time(g['end'])}")
        lines.append(g["text"])
        lines.append("")
    return "\n".join(lines).strip()


def _default_output_dir() -> str:
    base = folder_paths.get_output_directory()
    return os.path.join(base, "ComfyUI-QwenASR")


def _is_dir_path(path: str) -> bool:
    if not path:
        return False
    if path.endswith(("/", "\\")):
        return True
    return os.path.isdir(path)


def _make_default_filename(ext: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"qwenasr_subtitle_{stamp}{ext}"


class AILab_Qwen3ASR:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = _get_model_choices()
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input to transcribe."}),
            },
            "optional": {
                "model": (model_choices, {"default": model_choices[0], "tooltip": "Choose a locally available ASR model from models/Qwen3-ASR/."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "Inference precision."}),
                "language": (SUPPORTED_LANGUAGES, {"default": "auto", "tooltip": "Force language or auto-detect."}),
                "hints": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional hints/keywords (names, terms) to improve recognition."}),
                "unload_models": ("BOOLEAN", {"default": True, "tooltip": "Unload cached model after inference."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("TEXT",)
    FUNCTION = "transcribe"
    CATEGORY = "🧪AILab/🎙️QwenASR"

    def transcribe(
        self,
        audio,
        model="Qwen3-ASR-0.6B",
        precision="bf16",
        language="auto",
        hints="",
        unload_models=True,
    ):
        if Qwen3ASRModel is None:
            raise RuntimeError(f"qwen-asr not available: {_IMPORT_ERROR}")

        device = model_management.get_torch_device()
        dtype = _build_dtype(precision, device)

        model_path = _resolve_local_model(model)

        audio_data = _normalize_audio(audio)
        if audio_data is None:
            return ("",)

        lang = None if language == "auto" else language
        ctx = hints.strip() if isinstance(hints, str) else ""

        model = _load_cached_model(model_path, dtype, device, "auto", "")
        results = model.transcribe(
            audio=audio_data,
            language=lang,
            context=ctx if ctx else None,
            return_time_stamps=False,
        )

        result = results[0]
        text = result.text or ""

        if unload_models:
            _ASR_MODEL_CACHE.clear()
            try:
                model_management.soft_empty_cache()
            except Exception:
                pass

        return (text,)


class AILab_Qwen3ASRSubtitle:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = _get_model_choices()
        aligner_choices = _get_aligner_choices()
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input to transcribe."}),
            },
            "optional": {
                "model": (model_choices, {"default": model_choices[0], "tooltip": "Choose a locally available ASR model from models/Qwen3-ASR/."}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "Inference precision."}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto", "tooltip": "Attention backend override."}),
                "forced_aligner": (aligner_choices, {"default": aligner_choices[0], "tooltip": "Forced aligner for timestamped subtitles (from models/Qwen3-ASR/)."}),
                "language": (SUPPORTED_LANGUAGES, {"default": "auto", "tooltip": "Force language or auto-detect."}),
                "hints": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional hints/keywords (names, terms) to improve recognition."}),
                "output_format": (["none", "txt", "srt"], {"default": "none", "tooltip": "File save format only (does not change subtitle output)."}),
                "output_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional output file path (relative goes to ComfyUI output)."}),
                "split_mode": (["split_by_punctuation_or_pause_or_length", "split_by_punctuation_or_pause", "split_by_punctuation_or_length", "split_by_punctuation", "split_by_pause", "split_by_length"], {"default": "split_by_punctuation_or_pause_or_length", "tooltip": "Sentence splitting strategy."}),
                "max_gap_sec": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 8.0, "step": 0.1, "tooltip": "Max silence gap to keep the same sentence."}),
                "max_chars": ("INT", {"default": 40, "min": 0, "max": 200, "tooltip": "Optional max characters per line (0 = no limit)."}),
                "max_inference_batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "tooltip": "Batch size for inference/alignment to avoid OOM."}),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 2048, "tooltip": "Max new tokens per chunk."}),
                "unload_models": ("BOOLEAN", {"default": True, "tooltip": "Unload cached model after inference."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("TEXT", "SUBTITLES", "LANUGAGE", "OUTPUT_PATH")
    FUNCTION = "transcribe"
    CATEGORY = "🧪AILab/🎙️QwenASR"

    def transcribe(
        self,
        audio,
        model="Qwen3-ASR-0.6B",
        precision="bf16",
        attention="auto",
        forced_aligner="None",
        language="auto",
        hints="",
        output_format="none",
        output_path="",
        split_mode="split_by_punctuation_or_pause_or_length",
        max_gap_sec=0.6,
        max_chars=60,
        max_inference_batch_size=32,
        max_new_tokens=256,
        unload_models=True,
    ):
        if Qwen3ASRModel is None:
            raise RuntimeError(f"qwen-asr not available: {_IMPORT_ERROR}")

        device = model_management.get_torch_device()
        dtype = _build_dtype(precision, device)

        model_path = _resolve_local_model(model)

        forced_aligner_path = ""
        if forced_aligner and forced_aligner != "None":
            forced_aligner_path = _resolve_local_model(forced_aligner)

        audio_data = _normalize_audio(audio)
        if audio_data is None:
            return ("", "", "")

        lang = None if language == "auto" else language
        ctx = hints.strip() if isinstance(hints, str) else ""

        model = _load_cached_model(
            model_path,
            dtype,
            device,
            attention,
            forced_aligner_path,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        results = model.transcribe(
            audio=audio_data,
            language=lang,
            context=ctx if ctx else None,
            return_time_stamps=True,
        )

        result = results[0]
        text = result.text or ""
        detected_lang = result.language or ""
        subtitles = ""
        file_path = ""
        time_stamps = getattr(result, "time_stamps", None)
        groups = _group_time_stamps(time_stamps, max_gap_sec=max_gap_sec, max_chars=max_chars, split_mode=split_mode)
        # Always build subtitle output
        lines = []
        for g in groups:
            lines.append(f"{g['start']:.2f}-{g['end']:.2f}: {g['text']}")
        subtitles = "\n".join(lines) if lines else ""

        # Optional file save
        if output_format != "none":
            out_path = (output_path or "").strip()
            if not os.path.isabs(out_path):
                if out_path == "":
                    out_path = _default_output_dir()
                out_path = os.path.join(folder_paths.get_output_directory(), out_path)

            if _is_dir_path(out_path):
                ext = ".srt" if output_format == "srt" else ".txt"
                out_path = os.path.join(out_path, _make_default_filename(ext))
            else:
                root, ext = os.path.splitext(out_path)
                if not ext:
                    ext = ".srt" if output_format == "srt" else ".txt"
                    out_path = root + ext
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if output_format == "srt":
                file_content = _build_srt_from_groups(groups)
            else:
                file_content = subtitles
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            file_path = out_path

        if unload_models:
            _ASR_MODEL_CACHE.clear()
            try:
                model_management.soft_empty_cache()
            except Exception:
                pass

        return (text, subtitles, detected_lang, file_path)


NODE_CLASS_MAPPINGS = {
    "AILab_Qwen3ASR": AILab_Qwen3ASR,
    "AILab_Qwen3ASRSubtitle": AILab_Qwen3ASRSubtitle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_Qwen3ASR": "ASR (QwenASR)",
    "AILab_Qwen3ASRSubtitle": "Subtitle (QwenASR)",
}
