"""
debug_prompt.py
───────────────
Prints the fully assembled MedGemma prompt to stdout for inspection.

Usage (PowerShell):
    python debug_prompt.py

    # Override any option via flags:
    python debug_prompt.py --modality CT --region "Chest / Thorax" --task general
    python debug_prompt.py --modality "X-Ray" --region "Chest / Thorax" --view "PA & Lateral"
    python debug_prompt.py --modality MRI --region "Head / Brain" --task disease --disease "glioblastoma"
    python debug_prompt.py --task custom --custom "Describe the right hilum in detail."
    python debug_prompt.py --list-options
"""

from __future__ import annotations

import argparse
import sys
import textwrap

# Force UTF-8 output on Windows so the script works in both cp1252 PowerShell
# terminals and modern Windows Terminal (which is UTF-8 by default).
if sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from prompts import (
    Task, Modality, BodyRegion, XRayView,
    MODALITY_REGIONS,
    build_task_config,
)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Print the assembled MedGemma prompt for debugging.",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--modality", default="Auto-detect",
    help=f"Modality value. Default: 'Auto-detect'\nChoices: {[m.value for m in Modality]}",
)
parser.add_argument(
    "--region", default="Not specified",
    help="Body region. Default: 'Not specified'",
)
parser.add_argument(
    "--task", default="general",
    choices=["general", "disease", "anatomy", "custom"],
    help="Task shorthand. Default: general",
)
parser.add_argument("--disease",     default="pulmonary embolism", help="Disease name (task=disease)")
parser.add_argument("--custom",      default="",                   help="Custom prompt text (task=custom)")
parser.add_argument("--view",        default="Not specified",       help="X-Ray view (modality=X-Ray only)")
parser.add_argument("--custom-view", default="",                   help="Free-text view when --view 'Custom view…'")
parser.add_argument(
    "--list-options", action="store_true",
    help="Print all valid enum values and exit.",
)
args = parser.parse_args()

# ── List options mode ─────────────────────────────────────────────────────────

if args.list_options:
    print("\n=== Modality values ===")
    for m in Modality:
        print(f"  {m.value!r}")

    print("\n=== BodyRegion values ===")
    for r in BodyRegion:
        print(f"  {r.value!r}")

    print("\n=== XRayView values ===")
    for v in XRayView:
        print(f"  {v.value!r}")

    print("\n=== Task shortcuts ===")
    print("  general | disease | anatomy | custom")

    print("\n=== MODALITY_REGIONS (filtered regions per modality) ===")
    for mod, regions in MODALITY_REGIONS.items():
        names = [r.value for r in regions if r != BodyRegion.UNSPECIFIED]
        print(f"  {mod.value!r:40s} -> {names}")
    raise SystemExit(0)

# ── Resolve enums ─────────────────────────────────────────────────────────────

try:
    modality = Modality(args.modality)
except ValueError:
    print(f"[ERROR] Unknown modality: {args.modality!r}")
    print(f"        Valid values: {[m.value for m in Modality]}")
    raise SystemExit(1)

try:
    region = BodyRegion(args.region)
except ValueError:
    print(f"[ERROR] Unknown region: {args.region!r}")
    print(f"        Valid values: {[r.value for r in BodyRegion]}")
    raise SystemExit(1)

try:
    xray_view = XRayView(args.view)
except ValueError:
    print(f"[ERROR] Unknown X-Ray view: {args.view!r}")
    print(f"        Valid values: {[v.value for v in XRayView]}")
    raise SystemExit(1)

task_map = {
    "general" : Task.GENERAL_DESCRIPTION,
    "disease"  : Task.DISEASE_CLASSIFICATION,
    "anatomy"  : Task.ANATOMICAL_LOCATION,
    "custom"   : Task.CUSTOM_PROMPT,
}
task = task_map[args.task]

# ── Build config ──────────────────────────────────────────────────────────────

try:
    cfg = build_task_config(
        task=task,
        disease=args.disease,
        custom_prompt=args.custom,
        modality=modality,
        region=region,
        xray_view=xray_view,
        custom_view=args.custom_view,
    )
except ValueError as exc:
    print(f"[ERROR] {exc}")
    raise SystemExit(1)

# ── Print ─────────────────────────────────────────────────────────────────────

SEP  = "=" * 72
SEP2 = "-" * 72

def section(title: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)

def block(label: str, text: str) -> None:
    print(f"\n[{label}]")
    for line in text.splitlines():
        print(f"  {line}")

print(f"\n{SEP}")
print("  MEDGEMMA PROMPT DEBUG")
print(SEP)
print(f"  Task       : {cfg.task.value}")
print(f"  Modality   : {cfg.modality.value}")
print(f"  Region     : {cfg.region.value}")
print(f"  XRay View  : {cfg.xray_view.value}")
print(SEP)

block("SYSTEM PROMPT", cfg.system_prompt)
block("USER PROMPT (sent to model)", cfg.user_prompt)

# Show combined string as it arrives in model_engine._build_multimodal_messages
combined = f"{cfg.system_prompt}\n\n{cfg.user_prompt}"
block("COMBINED TEXT (system + user, as injected into chat)", combined)

print()
print(SEP2)
print(f"  Total characters : {len(combined)}")
print(f"  Approx tokens    : ~{len(combined) // 4}  (rough 4-char/token estimate)")
print(SEP2 + "\n")
