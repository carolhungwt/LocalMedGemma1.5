"""
prompts.py
──────────
Defines all task identifiers, system-level radiologist persona, and
user-facing prompt templates for MedGemma inference.

Design notes
  • SYSTEM_PROMPT  — injected once as the "system" role in the chat template
    to ground the model as a board-certified radiologist.
  • TASK_PROMPTS   — user-visible task keys mapped to their template strings.
    Templates that contain {disease} require string .format() substitution
    before use.
  • TaskConfig     — lightweight dataclass carrying everything needed to build
    the final multimodal message list passed to `apply_chat_template`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ─────────────────────────────────────────────────────────────────────────────
# Task identifiers
# ─────────────────────────────────────────────────────────────────────────────

class Task(str, Enum):
    GENERAL_DESCRIPTION  = "General Description"
    DISEASE_CLASSIFICATION = "Disease Classification"
    ANATOMICAL_LOCATION  = "Anatomical Location"
    CUSTOM_PROMPT        = "Custom Prompt"


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — injected as the "system" role
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = (
    "You are an expert radiologist with board certification and over 20 years of "
    "clinical experience interpreting medical imaging studies across all modalities "
    "(X-ray, CT, MRI, ultrasound, nuclear medicine). "
    "Your responses must be structured, precise, and use correct radiological "
    "terminology. Always organise your findings under clearly labelled sections: "
    "TECHNIQUE, FINDINGS, and IMPRESSION. "
    "Do not provide a final diagnosis or recommend treatment — limit your output "
    "to radiological observations and differential considerations. "
    "If image quality is insufficient for interpretation, state this clearly."
)

# ─────────────────────────────────────────────────────────────────────────────
# Task prompt templates
# ─────────────────────────────────────────────────────────────────────────────

TASK_PROMPTS: dict[Task, str] = {
    Task.GENERAL_DESCRIPTION: (
        "Analyze this medical imaging study. "
        "Provide a comprehensive structured report covering:\n"
        "  • TECHNIQUE: Describe the imaging modality, plane(s), and any contrast use "
        "    visible from the images.\n"
        "  • FINDINGS: Systematically describe the appearance of all visible anatomical "
        "    structures — report both normal and abnormal findings. Note size, "
        "    morphology, density/signal, and distribution of any lesion or abnormality.\n"
        "  • IMPRESSION: Summarise the most clinically significant findings and "
        "    provide a prioritised differential diagnosis list."
    ),

    Task.DISEASE_CLASSIFICATION: (
        "Analyze this medical imaging study specifically for the presence or absence "
        "of **{disease}**.\n\n"
        "Structure your response as follows:\n"
        "  • TECHNIQUE: Briefly note the apparent modality and acquisition parameters.\n"
        "  • TARGETED FINDINGS: Focus exclusively on radiological signs that confirm "
        "    or argue against {disease}. Describe each relevant imaging feature "
        "    (e.g., density, margins, distribution, associated findings) with "
        "    precise anatomical references.\n"
        "  • IMPRESSION: State whether the imaging findings are CONSISTENT WITH, "
        "    SUSPICIOUS FOR, or ARGUE AGAINST {disease}. Assign a confidence level "
        "    (low / moderate / high) and list key supporting evidence."
    ),

    Task.ANATOMICAL_LOCATION: (
        "Identify and precisely localise the primary abnormality visible in this "
        "medical imaging study.\n\n"
        "Structure your response as follows:\n"
        "  • TECHNIQUE: Note the apparent modality and any relevant acquisition details.\n"
        "  • FINDINGS: Describe the dominant abnormality in full. Include:\n"
        "      – Exact anatomical location (organ, lobe, segment, zone, or layer).\n"
        "      – Spatial relationships to adjacent named structures (vessels, "
        "        nerves, organ boundaries).\n"
        "      – Dimensions (if estimable) and morphological characteristics.\n"
        "      – Any secondary or ancillary findings that assist localisation.\n"
        "  • IMPRESSION: Provide a concise anatomical summary suitable for "
        "    surgical or interventional planning."
    ),
}

# Placeholder text shown in the UI for the custom prompt textarea
CUSTOM_PROMPT_PLACEHOLDER: str = (
    "Enter your custom radiology prompt here...\n\n"
    "Example: 'Evaluate the left lower lobe for consolidation and describe "
    "any associated pleural effusion or mediastinal shift.'"
)


# ─────────────────────────────────────────────────────────────────────────────
# TaskConfig — bundles everything needed for a single inference call
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskConfig:
    """
    Encapsulates the resolved prompt and metadata for one inference request.

    Attributes
    ----------
    task         : The Task enum value selected by the user.
    user_prompt  : The final, fully-resolved prompt string (disease substituted,
                   custom text included, etc.)
    system_prompt: The radiologist persona system prompt.
    """
    task         : Task
    user_prompt  : str
    system_prompt: str = field(default_factory=lambda: SYSTEM_PROMPT)


def build_task_config(
    task         : Task,
    disease      : str  = "",
    custom_prompt: str  = "",
) -> TaskConfig:
    """
    Resolve the correct prompt template and return a ready-to-use TaskConfig.

    Parameters
    ----------
    task          : Selected Task enum value.
    disease       : Required when task == Task.DISEASE_CLASSIFICATION.
    custom_prompt : Required when task == Task.CUSTOM_PROMPT.

    Returns
    -------
    TaskConfig

    Raises
    ------
    ValueError  if required supplementary inputs are missing.
    """
    if task == Task.DISEASE_CLASSIFICATION:
        if not disease.strip():
            raise ValueError(
                "A disease or condition name must be provided for the "
                "Disease Classification task."
            )
        prompt = TASK_PROMPTS[task].format(disease=disease.strip())

    elif task == Task.CUSTOM_PROMPT:
        if not custom_prompt.strip():
            raise ValueError("A custom prompt must be entered.")
        prompt = custom_prompt.strip()

    else:
        prompt = TASK_PROMPTS[task]

    return TaskConfig(task=task, user_prompt=prompt)
