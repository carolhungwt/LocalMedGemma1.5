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
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from prompts import TaskConfig

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "google/medgemma-1.5-4b-it"

# Maximum new tokens to generate; large enough for a full radiology report
MAX_NEW_TOKENS = 1024

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
    EnvironmentError  if no token is found and the model requires authentication.
    RuntimeError      if model loading fails for any other reason.
    """
    hw = get_hardware_info()

    # Resolve authentication token
    token = (
        hf_token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise EnvironmentError(
            "A Hugging Face access token is required to download "
            f"'{MODEL_ID}' (gated model). "
            "Provide it in the sidebar or set the HF_TOKEN environment variable."
        )

    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            token=token,
            trust_remote_code=True,
        )

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            token=token,
            torch_dtype=hw["dtype"],
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load '{MODEL_ID}': {exc}\n\n"
            "Possible causes:\n"
            "  • Invalid or expired Hugging Face token.\n"
            "  • You have not accepted the model's license on huggingface.co.\n"
            "  • Insufficient disk space or network error during download."
        ) from exc

    return processor, model, hw


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_multimodal_messages(
    images: list[Image.Image],
    task_config: TaskConfig,
) -> list[dict]:
    """
    Build the `messages` list consumed by `processor.apply_chat_template`.

    Structure
    ---------
    [
      { "role": "system",  "content": [{"type": "text", "text": SYSTEM_PROMPT}] },
      { "role": "user",
        "content": [
          {"type": "text",  "text": <user prompt>},
          {"type": "image"},   # repeated once per image slice
          ...
        ]
      },
    ]

    Placing the text token before the image tokens matches the MedGemma
    high-dimensional CT notebook convention.
    """
    image_tokens = [{"type": "image"} for _ in images]

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": task_config.system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": task_config.user_prompt}] + image_tokens,
        },
    ]
    return messages


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

    Parameters
    ----------
    images      : list of RGB PIL Images (one per slice/frame)
    task_config : TaskConfig from prompts.build_task_config()
    processor   : loaded AutoProcessor
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

    # apply_chat_template returns a fully formatted prompt string; tokenise it
    # along with the image list into model inputs.
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images=images,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Processor failed to build inputs: {exc}\n"
            "Ensure the number of {{type: image}} tokens matches len(images)."
        ) from exc

    # Move all tensors to the model's device and correct dtype
    model_device = next(model.parameters()).device
    model_dtype  = next(model.parameters()).dtype

    inputs = {
        k: (v.to(device=model_device, dtype=model_dtype)
            if v.is_floating_point() else v.to(device=model_device))
        for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[-1]

    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # greedy decoding for reproducibility
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

    # Decode only the newly generated tokens (skip the prompt echo)
    generated_ids = output_ids[0][input_len:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)

    return raw_text.strip()
