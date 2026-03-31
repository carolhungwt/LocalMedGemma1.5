"""
model_engine.py
───────────────
MedGemma inference engine for the local radiology AI application.

Responsibilities
  • Load google/medgemma-1.5-4b-it via Hugging Face Transformers, cached for
    the lifetime of the Streamlit session.
  • Build the multimodal chat payload required by MedGemma's chat template.
  • Run inference and return a clean report string.

Hardware strategy
  • CUDA available  → bfloat16  (preferred; avoids NaN accumulation)
  • CPU only        → float16   (fallback; slower but functional)
  • device_map="auto" lets Accelerate distribute across all visible GPUs or
    fall back to CPU automatically.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st
import torch
from huggingface_hub import scan_cache_dir
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from prompts import TaskConfig

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "google/medgemma-1.5-4b-it"

# Maximum new tokens to generate; large enough for a full radiology report
MAX_NEW_TOKENS = 1024

# Repetition control — prevents the model looping on the same phrase
# repetition_penalty > 1.0 down-weights tokens already in the context window.
# no_repeat_ngram_size blocks any n-gram from appearing more than once.
REPETITION_PENALTY   = 1.3
NO_REPEAT_NGRAM_SIZE = 4

# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_hardware_info() -> dict[str, str | bool]:
    """
    Return a dictionary describing available compute resources.

    Keys
    ----
    cuda_available : bool
    device_name    : str   — GPU name or "CPU"
    dtype_label    : str   — "bfloat16" / "float16" / "float32"
    dtype          : torch.dtype
    """
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        # bfloat16 is supported on Ampere (sm_80) and newer architectures
        bf16_supported = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_supported else torch.float16
        dtype_label = "bfloat16" if bf16_supported else "float16"
    else:
        device_name = "CPU"
        # float32 on CPU avoids stability issues from low-precision ops
        dtype = torch.float32
        dtype_label = "float32"

    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "dtype_label": dtype_label,
        "dtype": dtype,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading — cached for the Streamlit session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(hf_token: Optional[str] = None) -> tuple:
    """
    Load the MedGemma processor and model, caching them for the Streamlit
    session so weights are not reloaded on every interaction.

    The function reads the HF token from (in priority order):
      1. The `hf_token` argument (entered in the sidebar).
      2. The HF_TOKEN environment variable.
      3. The HUGGING_FACE_HUB_TOKEN environment variable (legacy name).

    Parameters
    ----------
    hf_token : optional Hugging Face access token string

    Returns
    -------
    (processor, model, hardware_info)

    Raises
    ------
    EnvironmentError  if weights are not cached locally AND no token is provided.
    RuntimeError      if model loading fails for any other reason.
    """
    hw = get_hardware_info()

    # Resolve token (optional when weights are already cached locally)
    token = (
        hf_token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or None
    )

    # Check whether the model weights already exist in the local HF cache.
    # from_pretrained() needs no token at all when loading purely from disk.
    weights_cached = _is_model_cached(MODEL_ID)

    if not weights_cached and not token:
        raise EnvironmentError(
            f"'{MODEL_ID}' is not in the local cache and no Hugging Face "
            "token was provided.\n\n"
            "First-time setup: enter your token in the sidebar (or set the "
            "HF_TOKEN environment variable) to download the model weights. "
            "After the first successful download you will never need the "
            "token again on this machine."
        )

    # Pass token only when needed; avoids unnecessary network calls on
    # subsequent runs where everything is served from local disk.
    auth = {"token": token} if token else {}

    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            use_fast=True,
            trust_remote_code=True,
            **auth,
        )

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype=hw["dtype"],
            device_map="auto",
            trust_remote_code=True,
            **auth,
        )
        model.eval()

    except Exception as exc:
        # Surface a clear message distinguishing auth vs. other failures
        err = str(exc).lower()
        if any(k in err for k in ("401", "403", "gated", "access", "forbidden", "unauthorized")):
            raise EnvironmentError(
                f"Authentication failed for '{MODEL_ID}'.\n\n"
                "  • Make sure your token is valid and not revoked.\n"
                "  • Confirm you have accepted the model license at "
                "huggingface.co/google/medgemma-1.5-4b-it."
            ) from exc
        raise RuntimeError(
            f"Failed to load '{MODEL_ID}': {exc}\n\n"
            "Possible causes:\n"
            "  • Insufficient disk space or a corrupted cache "
            "(try deleting ~/.cache/huggingface/hub/models--google--medgemma-1.5-4b-it).\n"
            "  • Network error during a partial download."
        ) from exc

    return processor, model, hw


def _is_model_cached(model_id: str) -> bool:
    """Return True if *any* revision of model_id exists in the local HF cache."""
    try:
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == model_id:
                return True
    except Exception:
        pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_multimodal_messages(
    images: list[Image.Image],
    task_config: TaskConfig,
) -> list[dict]:
    """
    Build the `messages` list for `processor.apply_chat_template`.

    Gemma 3 / MedGemma 1.5 constraints
    ────────────────────────────────────
    • The chat template does NOT support a "system" role — the system prompt
      is prepended to the user text content instead.
    • Image tokens must appear AFTER the text token in the content list,
      matching the high-dimensional CT notebook convention.

    Resulting structure
    -------------------
    [
      { "role": "user",
        "content": [
          {"type": "text",  "text": "<system_prompt>\\n\\n<user_prompt>"},
          {"type": "image"},   # one entry per image slice
          ...
        ]
      }
    ]
    """
    combined_text = f"{task_config.system_prompt}\n\n{task_config.user_prompt}"
    image_tokens  = [{"type": "image"} for _ in images]

    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": combined_text}] + image_tokens,
        }
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def generate_inference(
    images      : list[Image.Image],
    task_config : TaskConfig,
    processor,
    model,
    hw          : dict,
) -> str:
    """
    Run a single multimodal inference pass through MedGemma and return the
    cleaned output text.

    Two-step processor call (required by Gemma3Processor)
    ──────────────────────────────────────────────────────
    Step 1 — apply_chat_template(..., tokenize=False)
        Returns a plain formatted string.  Gemma3Processor does NOT accept
        images=, return_tensors=, or return_dict= here.

    Step 2 — processor(text=..., images=..., return_tensors="pt")
        Tokenises the text and encodes the images together into model-ready
        tensors.

    Parameters
    ----------
    images      : list of RGB PIL Images (one per slice/frame)
    task_config : TaskConfig from prompts.build_task_config()
    processor   : loaded AutoProcessor (Gemma3Processor)
    model       : loaded AutoModelForImageTextToText
    hw          : hardware info dict from get_hardware_info()

    Returns
    -------
    str  — the model's generated radiology report (trimmed of special tokens)

    Raises
    ------
    torch.cuda.OutOfMemoryError  re-raised with a user-friendly message.
    RuntimeError                 for any other inference failure.
    """
    if not images:
        raise ValueError("At least one image is required for inference.")

    messages = _build_multimodal_messages(images, task_config)

    # ── Step 1: format the chat into a plain string ──────────────────────────
    try:
        prompt_text: str = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,         # returns str — images are NOT passed here
        )
    except Exception as exc:
        raise RuntimeError(
            f"apply_chat_template failed: {exc}"
        ) from exc

    # ── Step 2: tokenise text + encode images into tensors ───────────────────
    try:
        inputs = processor(
            text=prompt_text,
            images=images,
            return_tensors="pt",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Processor failed to encode inputs: {exc}\n"
            "Check that the number of <image> placeholders in the prompt "
            "matches len(images)."
        ) from exc

    # Move all tensors to the model's device; cast floating-point tensors only
    model_device = next(model.parameters()).device
    model_dtype  = next(model.parameters()).dtype

    inputs = {
        k: (v.to(device=model_device, dtype=model_dtype)
            if v.is_floating_point() else v.to(device=model_device))
        for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[-1]

    # ── Step 3: generate ─────────────────────────────────────────────────────
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,              # greedy decoding for reproducibility
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            )
    except torch.cuda.OutOfMemoryError as oom:
        torch.cuda.empty_cache()
        raise torch.cuda.OutOfMemoryError(
            "GPU ran out of memory during inference.\n\n"
            "Suggested remedies:\n"
            "  • Reduce the number of video frames (try 8 instead of 16).\n"
            "  • Upload fewer DICOM slices (sub-sample the series).\n"
            "  • Close other GPU-intensive applications.\n"
            f"Original error: {oom}"
        ) from oom
    except Exception as exc:
        raise RuntimeError(f"Model generation failed: {exc}") from exc

    # Decode only the newly generated tokens (exclude the echoed prompt)
    generated_ids = output_ids[0][input_len:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)

    return raw_text.strip()
