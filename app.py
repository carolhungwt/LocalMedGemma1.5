"""
app.py
──────
Streamlit entry-point for the MedGemma Local Radiology AI application.

Run with:
    streamlit run app.py

Layout overview
  ┌────────────────┬──────────────────────────────────────────────────┐
  │   SIDEBAR      │  MAIN AREA                                       │
  │                │  ┌───────────────────┬──────────────────────────┐│
  │  • HF Token    │  │  LEFT COLUMN      │  RIGHT COLUMN            ││
  │  • HW Status   │  │  File Uploader    │  Analyze button          ││
  │  • Task Picker │  │  Image Preview    │  Spinner + Report        ││
  │                │  └───────────────────┴──────────────────────────┘│
  └────────────────┴──────────────────────────────────────────────────┘
"""

from __future__ import annotations

import traceback

import streamlit as st
import torch
from PIL import Image

import data_utils as du
import model_engine as me
from prompts import (
    Task, Modality, MODALITY_SHORT, MODALITY_REGIONS,
    BodyRegion, XRayView,
    build_task_config, CUSTOM_PROMPT_PLACEHOLDER,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedGemma Radiology AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — clean clinical aesthetic
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
        /* Main background */
        .stApp { background-color: #0e1117; }

        /* Report card */
        .report-card {
            background-color: #1a1f2e;
            border: 1px solid #2d3a5c;
            border-radius: 10px;
            padding: 1.5rem 2rem;
            font-family: 'Courier New', monospace;
            font-size: 0.88rem;
            line-height: 1.7;
            color: #d4e0f7;
            white-space: pre-wrap;
        }
        /* Section headers inside report */
        .report-card strong { color: #7eb8f7; }

        /* Sidebar headings */
        .sidebar-section {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #8899bb;
            margin-top: 1.2rem;
            margin-bottom: 0.3rem;
        }

        /* Status badge */
        .hw-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .hw-badge-gpu  { background:#1a3a2a; color:#4ade80; border:1px solid #4ade80; }
        .hw-badge-cpu  { background:#3a2a1a; color:#fb923c; border:1px solid #fb923c; }

        /* Divider */
        hr { border-color: #2d3a5c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png",
        width=40,
    )
    st.title("MedGemma AI")
    st.caption("Local Radiology Report Generator")
    st.divider()

    # ── Authentication ───────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Hugging Face Access</p>', unsafe_allow_html=True)

    weights_cached = me._is_model_cached(me.MODEL_ID)
    if weights_cached:
        st.success("Model weights cached locally — no token needed.", icon="✅")

    hf_token = st.text_input(
        "HF Token" if not weights_cached else "HF Token (optional)",
        type="password",
        placeholder="hf_..." if not weights_cached else "Not required — weights are local",
        help=(
            "Only needed for the **first download** of the gated MedGemma model. "
            "Once weights are cached locally this field can be left blank. "
            "Get a token at huggingface.co/settings/tokens"
        ),
    )

    # ── Hardware status ──────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Hardware Status</p>', unsafe_allow_html=True)
    hw = me.get_hardware_info()
    if hw["cuda_available"]:
        st.markdown(
            f'<span class="hw-badge hw-badge-gpu">GPU · {hw["device_name"]}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Precision: {hw['dtype_label']}")
    else:
        st.markdown(
            '<span class="hw-badge hw-badge-cpu">CPU only — inference will be slow</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Precision: {hw['dtype_label']}")

    # ── Task selector ────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Analysis Task</p>', unsafe_allow_html=True)
    task_label = st.selectbox(
        "Task",
        options=[t.value for t in Task],
        label_visibility="collapsed",
    )
    selected_task = Task(task_label)

    # Conditional supplementary inputs
    disease_name   = ""
    custom_prompt  = ""
    num_vid_slices = du.DEFAULT_VIDEO_SLICES

    if selected_task == Task.DISEASE_CLASSIFICATION:
        disease_name = st.text_input(
            "Pathology / Disease",
            placeholder="e.g. pulmonary embolism",
            help="The model will focus exclusively on this condition.",
        )

    if selected_task == Task.CUSTOM_PROMPT:
        custom_prompt = st.text_area(
            "Custom Prompt",
            height=140,
            placeholder=CUSTOM_PROMPT_PLACEHOLDER,
        )

    # ── Imaging modality selector ────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">Imaging Modality</p>', unsafe_allow_html=True)

    auto_detected: Modality = st.session_state.get("detected_modality", Modality.AUTO)
    modality_options = [m.value for m in Modality]
    default_idx = modality_options.index(auto_detected.value)

    selected_modality_val = st.selectbox(
        "Modality",
        options=modality_options,
        index=default_idx,
        label_visibility="collapsed",
        help=(
            "For DICOM files this is auto-detected from the DICOM Modality tag "
            "(0008,0060). For images and videos, select the correct modality "
            "manually — this injects acquisition-specific context into the prompt "
            "so the model reasons with the right imaging vocabulary."
        ),
    )
    selected_modality = Modality(selected_modality_val)

    if auto_detected not in (Modality.AUTO, Modality.OTHER):
        st.caption(f"Auto-detected from DICOM tag: **{MODALITY_SHORT[auto_detected]}**")
    elif auto_detected == Modality.OTHER:
        st.caption("DICOM tag present but modality unrecognised — please select manually.")

    # ── Anatomical region selector ───────────────────────────────────────────
    # Shown for every modality so the user can scope the report.
    # Options are filtered to the clinically relevant subset for the chosen
    # modality; UNSPECIFIED is always the first / default option.
    st.markdown('<p class="sidebar-section">Anatomical Region</p>', unsafe_allow_html=True)

    applicable_regions = MODALITY_REGIONS.get(selected_modality, list(BodyRegion))
    # Ensure UNSPECIFIED is always first even if missing from the filtered list
    region_options = [BodyRegion.UNSPECIFIED] + [
        r for r in applicable_regions if r != BodyRegion.UNSPECIFIED
    ]

    # Reset region selection when modality changes so a stale value is not kept
    if "last_modality" not in st.session_state:
        st.session_state.last_modality = selected_modality
    if st.session_state.last_modality != selected_modality:
        st.session_state.last_modality  = selected_modality
        st.session_state.selected_region = BodyRegion.UNSPECIFIED

    region_values = [r.value for r in region_options]
    saved_region  = st.session_state.get("selected_region", BodyRegion.UNSPECIFIED)
    region_idx    = (
        region_values.index(saved_region.value)
        if saved_region.value in region_values
        else 0
    )

    selected_region_val = st.selectbox(
        "Region",
        options=region_values,
        index=region_idx,
        label_visibility="collapsed",
        help=(
            "Narrows the report to the selected anatomical area. "
            "Choose 'Not specified' to let the model describe all visible anatomy."
        ),
    )
    selected_region = BodyRegion(selected_region_val)
    st.session_state.selected_region = selected_region

    # ── X-Ray view / projection selector ────────────────────────────────────
    # Only shown when the modality is X-Ray.
    selected_xray_view  = XRayView.UNSPECIFIED
    custom_view_text    = ""

    if selected_modality == Modality.XRAY:
        st.markdown('<p class="sidebar-section">Radiographic View</p>', unsafe_allow_html=True)

        view_options = [v.value for v in XRayView]
        selected_view_val = st.selectbox(
            "View",
            options=view_options,
            index=0,
            label_visibility="collapsed",
            help=(
                "Specifies the projection used for acquisition. This is injected "
                "directly into the prompt so the model applies the correct "
                "anatomical orientation (e.g. PA vs AP heart size conventions)."
            ),
        )
        selected_xray_view = XRayView(selected_view_val)

        if selected_xray_view == XRayView.CUSTOM:
            custom_view_text = st.text_input(
                "Custom projection name",
                placeholder="e.g. Judet, Sunrise, Tunnel, Mortise…",
                help="Enter any non-standard projection name.",
            )

    st.divider()

    # ── Advanced settings (collapsible) ─────────────────────────────────────
    with st.expander("Advanced Settings", expanded=False):
        num_vid_slices = st.slider(
            "Video / volume frames",
            min_value=4,
            max_value=32,
            value=du.DEFAULT_VIDEO_SLICES,
            step=4,
            help="Number of uniformly spaced frames extracted from video files.",
        )
        st.caption(
            "Reducing this value lowers GPU memory usage for video inputs."
        )

    st.divider()
    st.caption(
        "⚠ For research / educational use only. "
        "This tool does not provide medical advice."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main area header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 🩻 MedGemma Local Radiology AI")
st.markdown(
    "Upload a medical image, DICOM series, or video to generate a structured "
    "radiology report powered by **google/medgemma-1.5-4b-it**."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Two-column layout
# ─────────────────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([1, 1], gap="large")

# ── LEFT: File uploader + preview ────────────────────────────────────────────
with left_col:
    st.subheader("📂 Input")

    accept_multiple = st.checkbox(
        "Multiple files (DICOM series / volume)",
        value=False,
        help="Enable to upload multiple .dcm files forming a 3-D volume.",
    )

    uploaded_files = st.file_uploader(
        "Upload medical image, video, or DICOM file(s)",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "mp4", "mov", "avi", "dcm"],
        accept_multiple_files=accept_multiple,
        label_visibility="collapsed",
    )

    # Normalise to list
    if uploaded_files is None:
        uploaded_files = []
    elif not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    # Session state for parsed images so the preview persists
    if "parsed_images"      not in st.session_state:
        st.session_state.parsed_images      = []
    if "modality_label"     not in st.session_state:
        st.session_state.modality_label     = ""
    if "detected_modality"  not in st.session_state:
        st.session_state.detected_modality  = Modality.AUTO
    if "parse_error"        not in st.session_state:
        st.session_state.parse_error        = ""

    if uploaded_files:
        try:
            images, file_label, detected_mod = du.process_uploaded_files(
                uploaded_files,
                num_video_slices=num_vid_slices,
            )
            st.session_state.parsed_images     = images
            st.session_state.modality_label    = file_label
            st.session_state.detected_modality = detected_mod
            st.session_state.parse_error       = ""
        except Exception as exc:
            st.session_state.parsed_images     = []
            st.session_state.modality_label    = ""
            st.session_state.detected_modality = Modality.AUTO
            st.session_state.parse_error       = str(exc)

    # ── Preview ──────────────────────────────────────────────────────────────
    if st.session_state.parse_error:
        st.error(f"File error: {st.session_state.parse_error}")

    elif st.session_state.parsed_images:
        images       = st.session_state.parsed_images
        modality_lbl = st.session_state.modality_label

        det_mod  = st.session_state.detected_modality
        mod_badge = (
            f"  ·  Modality: **{MODALITY_SHORT[det_mod]}**"
            if det_mod not in (Modality.AUTO, Modality.OTHER)
            else ""
        )
        st.success(f"Loaded: **{modality_lbl}**  |  {len(images)} frame(s){mod_badge}")

        if len(images) == 1:
            st.image(images[0], use_container_width=True, caption="Input image")
        else:
            slice_idx = st.slider(
                "Browse slices / frames",
                min_value=1,
                max_value=len(images),
                value=len(images) // 2 + 1,
                format="Slice %d",
            )
            st.image(
                images[slice_idx - 1],
                use_container_width=True,
                caption=f"Slice {slice_idx} / {len(images)}",
            )
    else:
        st.info("Upload a file above to see a preview here.")

# ── RIGHT: Analysis button + report ──────────────────────────────────────────
with right_col:
    st.subheader("📋 Report")

    if "report" not in st.session_state:
        st.session_state.report = ""
    if "report_error" not in st.session_state:
        st.session_state.report_error = ""

    # Token only required when weights aren't cached yet
    token_ok     = bool(hf_token) or me._is_model_cached(me.MODEL_ID)
    analyse_ready = bool(st.session_state.parsed_images) and token_ok

    if not token_ok:
        st.warning(
            "First-time setup: enter your Hugging Face token in the sidebar "
            "to download the model. You won't need it again after that."
        )
    elif not st.session_state.parsed_images:
        st.info("Upload a medical image to enable the Analyse button.")

    analyse_clicked = st.button(
        "🔬 Analyse",
        disabled=not analyse_ready,
        use_container_width=True,
        type="primary",
    )

    if analyse_clicked and analyse_ready:
        st.session_state.report       = ""
        st.session_state.report_error = ""

        # Build task config first — surface prompt errors before model load
        try:
            task_config = build_task_config(
                task=selected_task,
                disease=disease_name,
                custom_prompt=custom_prompt,
                modality=selected_modality,
                region=selected_region,
                xray_view=selected_xray_view,
                custom_view=custom_view_text,
            )
        except ValueError as ve:
            st.session_state.report_error = str(ve)
            st.error(str(ve))
            st.stop()

        # Load model (cached — first call may take several minutes)
        with st.spinner("Loading MedGemma model (first run downloads weights)…"):
            try:
                processor, model, hw_loaded = me.load_model(hf_token=hf_token)
            except (EnvironmentError, RuntimeError) as exc:
                st.session_state.report_error = str(exc)
                st.error(str(exc))
                st.stop()

        # Run inference
        with st.spinner(
            f"Analysing {st.session_state.modality_label} "
            f"({len(st.session_state.parsed_images)} frame(s))…"
        ):
            try:
                report = me.generate_inference(
                    images=st.session_state.parsed_images,
                    task_config=task_config,
                    processor=processor,
                    model=model,
                    hw=hw_loaded,
                )
                st.session_state.report = report
            except torch.cuda.OutOfMemoryError as oom:
                st.session_state.report_error = str(oom)
            except Exception as exc:
                st.session_state.report_error = (
                    f"Inference failed: {exc}\n\n"
                    f"```\n{traceback.format_exc()}\n```"
                )

    # ── Display report / error ────────────────────────────────────────────────
    if st.session_state.report_error:
        st.error(st.session_state.report_error)

    elif st.session_state.report:
        report_text = st.session_state.report

        # Download button
        st.download_button(
            label="⬇ Download Report (.txt)",
            data=report_text,
            file_name="radiology_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Rendered report card
        # Bold the section headings (TECHNIQUE / FINDINGS / IMPRESSION)
        import re
        highlighted = re.sub(
            r'\b(TECHNIQUE|FINDINGS?|IMPRESSION|TARGETED FINDINGS?'
            r'|ANATOMICAL SUMMARY|LOCALISATION)\b',
            r'<strong>\1</strong>',
            report_text,
        )
        st.markdown(
            f'<div class="report-card">{highlighted}</div>',
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div style="
                border:1px dashed #2d3a5c;
                border-radius:10px;
                padding:2rem;
                text-align:center;
                color:#4a5a7a;
                font-size:0.9rem;
            ">
            The generated radiology report will appear here after you
            click <b>Analyse</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<p style='text-align:center; color:#4a5a7a; font-size:0.78rem;'>"
    "MedGemma Local · Non-commercial research use only · "
    "Not a substitute for professional medical advice"
    "</p>",
    unsafe_allow_html=True,
)
