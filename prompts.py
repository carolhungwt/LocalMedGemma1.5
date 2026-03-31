"""
prompts.py
──────────
Defines all task identifiers, modality context, anatomical region context,
X-Ray view context, system-level radiologist persona, and user-facing prompt
templates for MedGemma inference.

Design notes
  • Modality       — enum of supported imaging modalities; drives the modality
                     context block prepended to every prompt.
  • BodyRegion     — enum of anatomical regions; filtered per modality in the
                     UI via MODALITY_REGIONS.  Adds a focused region line to
                     the prompt so the model scopes its findings correctly.
  • XRayView       — enum of standard radiographic projections, only surfaced
                     in the UI when modality == XRAY.
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
# Anatomical body region
# ─────────────────────────────────────────────────────────────────────────────

class BodyRegion(str, Enum):
    """
    Anatomical regions surfaced in the UI.

    UNSPECIFIED means the user chose not to narrow the region; the model
    describes whatever anatomy is visible.  All other values inject a
    targeted region line into the prompt.
    """
    UNSPECIFIED      = "Not specified"
    HEAD_BRAIN       = "Head / Brain"
    NECK             = "Neck"
    CHEST            = "Chest / Thorax"
    ABDOMEN          = "Abdomen"
    PELVIS           = "Pelvis"
    ABDOMEN_PELVIS   = "Abdomen & Pelvis"
    CAP              = "Chest / Abdomen / Pelvis"
    SPINE_CERVICAL   = "Spine — Cervical"
    SPINE_THORACIC   = "Spine — Thoracic"
    SPINE_LUMBAR     = "Spine — Lumbar / Sacral"
    SHOULDER         = "Shoulder"
    ELBOW            = "Elbow"
    WRIST_HAND       = "Wrist / Hand"
    HIP              = "Hip"
    KNEE             = "Knee"
    ANKLE_FOOT       = "Ankle / Foot"
    BREAST           = "Breast"
    CARDIAC          = "Cardiac"
    WHOLE_BODY       = "Whole Body"


# Which regions are clinically relevant for each modality.
# The UI selectbox is filtered to this list.  UNSPECIFIED is always
# prepended by the UI so the user can opt out.
MODALITY_REGIONS: dict[Modality, list[BodyRegion]] = {
    Modality.AUTO: list(BodyRegion),   # show all when modality is unknown
    Modality.XRAY: [
        BodyRegion.CHEST,
        BodyRegion.ABDOMEN,
        BodyRegion.PELVIS,
        BodyRegion.SPINE_CERVICAL,
        BodyRegion.SPINE_THORACIC,
        BodyRegion.SPINE_LUMBAR,
        BodyRegion.HEAD_BRAIN,
        BodyRegion.SHOULDER,
        BodyRegion.ELBOW,
        BodyRegion.WRIST_HAND,
        BodyRegion.HIP,
        BodyRegion.KNEE,
        BodyRegion.ANKLE_FOOT,
        BodyRegion.WHOLE_BODY,
    ],
    Modality.CT: [
        BodyRegion.HEAD_BRAIN,
        BodyRegion.NECK,
        BodyRegion.CHEST,
        BodyRegion.ABDOMEN,
        BodyRegion.PELVIS,
        BodyRegion.ABDOMEN_PELVIS,
        BodyRegion.CAP,
        BodyRegion.SPINE_CERVICAL,
        BodyRegion.SPINE_THORACIC,
        BodyRegion.SPINE_LUMBAR,
        BodyRegion.CARDIAC,
        BodyRegion.WHOLE_BODY,
    ],
    Modality.MRI: [
        BodyRegion.HEAD_BRAIN,
        BodyRegion.NECK,
        BodyRegion.SPINE_CERVICAL,
        BodyRegion.SPINE_THORACIC,
        BodyRegion.SPINE_LUMBAR,
        BodyRegion.CHEST,
        BodyRegion.ABDOMEN,
        BodyRegion.PELVIS,
        BodyRegion.CARDIAC,
        BodyRegion.BREAST,
        BodyRegion.SHOULDER,
        BodyRegion.ELBOW,
        BodyRegion.WRIST_HAND,
        BodyRegion.HIP,
        BodyRegion.KNEE,
        BodyRegion.ANKLE_FOOT,
    ],
    Modality.ULTRASOUND: [
        BodyRegion.ABDOMEN,
        BodyRegion.PELVIS,
        BodyRegion.NECK,
        BodyRegion.BREAST,
        BodyRegion.CARDIAC,
        BodyRegion.SHOULDER,
        BodyRegion.HIP,
        BodyRegion.WHOLE_BODY,
    ],
    Modality.PET: [
        BodyRegion.WHOLE_BODY,
        BodyRegion.HEAD_BRAIN,
        BodyRegion.NECK,
        BodyRegion.CHEST,
        BodyRegion.ABDOMEN_PELVIS,
    ],
    Modality.FLUOROSCOPY: [
        BodyRegion.CHEST,
        BodyRegion.ABDOMEN,
        BodyRegion.PELVIS,
        BodyRegion.SPINE_CERVICAL,
        BodyRegion.SPINE_THORACIC,
        BodyRegion.SPINE_LUMBAR,
        BodyRegion.CARDIAC,
        BodyRegion.HEAD_BRAIN,
        BodyRegion.WHOLE_BODY,
    ],
    Modality.MAMMOGRAPHY: [
        BodyRegion.BREAST,
    ],
    Modality.OTHER: list(BodyRegion),
}


# ─────────────────────────────────────────────────────────────────────────────
# X-Ray projection / view
# ─────────────────────────────────────────────────────────────────────────────

class XRayView(str, Enum):
    """
    Standard radiographic projections.

    Only surfaced in the UI when Modality == XRAY.
    CUSTOM triggers a free-text input so the user can specify any non-standard
    projection (e.g. Judet, Sunrise, tunnel view).
    """
    UNSPECIFIED  = "Not specified"
    PA           = "PA (Posteroanterior)"
    AP           = "AP (Anteroposterior)"
    LATERAL      = "Lateral"
    PA_LATERAL   = "PA & Lateral"
    AP_LATERAL   = "AP & Lateral"
    OBLIQUE      = "Oblique"
    LORDOTIC     = "Apical Lordotic"
    DECUBITUS    = "Decubitus"
    SWIMMERS     = "Swimmer's (C7-T1)"
    CUSTOM       = "Custom view…"


# ─────────────────────────────────────────────────────────────────────────────
# Context builders — produce the lines inserted between modality block and task
# ─────────────────────────────────────────────────────────────────────────────

def _build_region_line(region: BodyRegion) -> str:
    """Return a one-line region context string, or empty string if unspecified."""
    if region == BodyRegion.UNSPECIFIED:
        return ""
    return (
        f"ANATOMICAL REGION: {region.value}.\n"
        "Focus your systematic review on this region. Describe adjacent "
        "structures only where relevant to the primary findings."
    )


def _build_view_line(view: XRayView, custom_view: str = "") -> str:
    """Return a one-line projection context string, or empty string if unspecified."""
    if view == XRayView.UNSPECIFIED:
        return ""
    label = custom_view.strip() if view == XRayView.CUSTOM and custom_view.strip() else view.value
    return f"RADIOGRAPHIC VIEW / PROJECTION: {label}."


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
    user_prompt  : Fully assembled prompt string:
                     [modality block] + [region line] + [view line] + [task]
    system_prompt: The radiologist persona system prompt.
    modality     : Imaging modality used for display and prompt injection.
    region       : Anatomical region focus (BodyRegion.UNSPECIFIED = no focus).
    xray_view    : Radiographic projection; only meaningful when modality=XRAY.
    """
    task         : Task
    user_prompt  : str
    system_prompt: str        = field(default_factory=lambda: SYSTEM_PROMPT)
    modality     : Modality   = Modality.AUTO
    region       : BodyRegion = BodyRegion.UNSPECIFIED
    xray_view    : XRayView   = XRayView.UNSPECIFIED


def build_task_config(
    task         : Task,
    disease      : str        = "",
    custom_prompt: str        = "",
    modality     : Modality   = Modality.AUTO,
    region       : BodyRegion = BodyRegion.UNSPECIFIED,
    xray_view    : XRayView   = XRayView.UNSPECIFIED,
    custom_view  : str        = "",
) -> TaskConfig:
    """
    Resolve the correct prompt template and return a ready-to-use TaskConfig.

    Prompt assembly order
    ─────────────────────
    1. Modality context block  (always present)
    2. Anatomical region line  (omitted when BodyRegion.UNSPECIFIED)
    3. X-Ray view / projection (omitted unless modality=XRAY and view set)
    4. Task instructions

    Parameters
    ----------
    task          : Selected Task enum value.
    disease       : Required when task == Task.DISEASE_CLASSIFICATION.
    custom_prompt : Required when task == Task.CUSTOM_PROMPT.
    modality      : Imaging modality. Defaults to AUTO.
    region        : Anatomical region. Defaults to UNSPECIFIED (no focus).
    xray_view     : Radiographic projection. Defaults to UNSPECIFIED.
    custom_view   : Free-text projection label used when xray_view=CUSTOM.

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

    # ── Assemble context blocks ───────────────────────────────────────────────
    modality_block = MODALITY_CONTEXT.get(modality, MODALITY_CONTEXT[Modality.AUTO])
    region_line    = _build_region_line(region)
    view_line      = (
        _build_view_line(xray_view, custom_view)
        if modality == Modality.XRAY
        else ""
    )

    # Join non-empty sections with blank lines for readability
    context_parts = [modality_block] + [s for s in (region_line, view_line) if s]
    full_prompt   = "\n\n".join(context_parts) + "\n\n" + task_text

    return TaskConfig(
        task=task,
        user_prompt=full_prompt,
        modality=modality,
        region=region,
        xray_view=xray_view,
    )
