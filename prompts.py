"""
prompts.py
──────────
Defines all task identifiers, modality context, system-level radiologist
persona, and user-facing prompt templates for MedGemma inference.

Design notes
  • Modality       — enum of supported imaging modalities; drives the modality
                     context string prepended to every prompt so the model does
                     not have to guess the acquisition type from pixel data.
  • SYSTEM_PROMPT  — radiologist persona grounding; merged into the user turn
                     because Gemma 3 has no "system" role.
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
# Imaging modality
# ─────────────────────────────────────────────────────────────────────────────

class Modality(str, Enum):
    """
    Supported imaging modalities.

    Values match human-readable labels shown in the UI.
    DICOM tag (0008,0060) values are mapped to these in data_utils.py.
    AUTO is used when no modality is specified or detectable.
    """
    AUTO          = "Auto-detect"
    XRAY          = "X-Ray"
    CT            = "CT (Computed Tomography)"
    MRI           = "MRI (Magnetic Resonance Imaging)"
    ULTRASOUND    = "Ultrasound"
    PET           = "PET / Nuclear Medicine"
    FLUOROSCOPY   = "Fluoroscopy / Angiography"
    MAMMOGRAPHY   = "Mammography"
    OTHER         = "Other / Unknown"


# Human-readable short labels used in the UI badge
MODALITY_SHORT: dict[Modality, str] = {
    Modality.AUTO        : "Auto",
    Modality.XRAY        : "X-Ray",
    Modality.CT          : "CT",
    Modality.MRI         : "MRI",
    Modality.ULTRASOUND  : "US",
    Modality.PET         : "PET/NM",
    Modality.FLUOROSCOPY : "XA/Fluoro",
    Modality.MAMMOGRAPHY : "Mammo",
    Modality.OTHER       : "Other",
}

# Per-modality context injected at the START of every prompt so the model
# receives explicit acquisition context rather than inferring it from pixels.
# This is particularly important for normalised/windowed images that can look
# similar across modalities after uint8 conversion.
MODALITY_CONTEXT: dict[Modality, str] = {
    Modality.AUTO: (
        "The imaging modality has not been specified. Infer it from the image "
        "appearance and state your interpretation in the TECHNIQUE section."
    ),
    Modality.XRAY: (
        "IMAGING MODALITY: Plain radiograph (X-Ray).\n"
        "Interpret this study using standard radiographic conventions. "
        "Assess for radio-opaque and radio-lucent abnormalities. "
        "Reference standard projections (PA/AP, lateral) where relevant."
    ),
    Modality.CT: (
        "IMAGING MODALITY: Computed Tomography (CT).\n"
        "Apply Hounsfield Unit (HU) conventions where applicable. "
        "Comment on window settings inferred from the image appearance "
        "(lung, mediastinal, bone, or soft-tissue window). "
        "If contrast enhancement is visible, note the phase (arterial, venous, delayed)."
    ),
    Modality.MRI: (
        "IMAGING MODALITY: Magnetic Resonance Imaging (MRI).\n"
        "Identify the likely pulse sequence (T1, T2, FLAIR, DWI, GRE, etc.) "
        "from signal characteristics and describe findings accordingly. "
        "Note signal intensity relative to reference tissues (e.g., CSF, fat, muscle)."
    ),
    Modality.ULTRASOUND: (
        "IMAGING MODALITY: Ultrasound (US).\n"
        "Describe echogenicity (anechoic, hypoechoic, isoechoic, hyperechoic), "
        "through-transmission, and posterior acoustic features. "
        "Comment on vascularity if Doppler information is visible."
    ),
    Modality.PET: (
        "IMAGING MODALITY: PET / Nuclear Medicine.\n"
        "Describe the distribution and intensity of radiotracer uptake. "
        "Report focal areas of abnormal uptake using SUV terminology where visible. "
        "Correlate with any co-registered anatomical imaging if present."
    ),
    Modality.FLUOROSCOPY: (
        "IMAGING MODALITY: Fluoroscopy / Digital Subtraction Angiography (DSA).\n"
        "Describe vessel opacification, luminal calibre, filling defects, "
        "extravasation, or dynamic flow abnormalities visible in the frames provided."
    ),
    Modality.MAMMOGRAPHY: (
        "IMAGING MODALITY: Mammography.\n"
        "Use ACR BI-RADS terminology. Describe breast composition, masses "
        "(shape, margin, density), calcifications (morphology, distribution), "
        "architectural distortion, and asymmetries."
    ),
    Modality.OTHER: (
        "IMAGING MODALITY: Not specified or non-standard.\n"
        "Describe the imaging study as presented, noting any technical features "
        "that help identify the acquisition method."
    ),
}

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
                   custom text included, modality context prepended).
    system_prompt: The radiologist persona system prompt.
    modality     : The imaging modality, used for display and prompt injection.
    """
    task         : Task
    user_prompt  : str
    system_prompt: str      = field(default_factory=lambda: SYSTEM_PROMPT)
    modality     : Modality = Modality.AUTO


def build_task_config(
    task         : Task,
    disease      : str      = "",
    custom_prompt: str      = "",
    modality     : Modality = Modality.AUTO,
) -> TaskConfig:
    """
    Resolve the correct prompt template and return a ready-to-use TaskConfig.

    The modality context string is always prepended to the task prompt so the
    model receives explicit acquisition context before the task instructions.

    Parameters
    ----------
    task          : Selected Task enum value.
    disease       : Required when task == Task.DISEASE_CLASSIFICATION.
    custom_prompt : Required when task == Task.CUSTOM_PROMPT.
    modality      : Imaging modality — drives the context prefix injected into
                    the prompt. Defaults to AUTO (model infers from pixels).

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
        task_text = TASK_PROMPTS[task].format(disease=disease.strip())

    elif task == Task.CUSTOM_PROMPT:
        if not custom_prompt.strip():
            raise ValueError("A custom prompt must be entered.")
        task_text = custom_prompt.strip()

    else:
        task_text = TASK_PROMPTS[task]

    # Prepend the modality context so it is always the first thing the model reads
    modality_prefix = MODALITY_CONTEXT.get(modality, MODALITY_CONTEXT[Modality.AUTO])
    full_prompt = f"{modality_prefix}\n\n{task_text}"

    return TaskConfig(task=task, user_prompt=full_prompt, modality=modality)
