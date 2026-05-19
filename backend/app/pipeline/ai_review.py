"""OpenRouter vision AI review of the alignment overlay."""
from __future__ import annotations

import base64
import json

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


async def review_alignment(
    overlay_png_path: str,
    confidence: float,
    residual_mm: float,
    api_key: str,
    model: str = "anthropic/claude-sonnet-4-6",
) -> dict:
    """Call OpenRouter vision model to sanity-check the alignment overlay.

    Returns a dict with `summary`, `concerns` (list), and `model_confidence` (0-1).
    This is purely advisory — it never affects the alignment result.
    """
    if not api_key:
        return {
            "summary": "AI review unavailable — no API key configured.",
            "concerns": [],
            "model_confidence": None,
        }

    try:
        with open(overlay_png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return {
            "summary": "AI review unavailable — overlay image not found.",
            "concerns": [],
            "model_confidence": None,
        }

    prompt = f"""You are reviewing an automated alignment of a LiDAR scan to a building floor plan.

The deterministic pipeline reports:
- Confidence: {confidence:.0%}
- Residual error: {residual_mm:.1f}mm

The image shows the scan outline (red) overlaid on the plan (black) after alignment.

Respond with ONLY a JSON object — no markdown, no preamble:
{{
  "summary": "one sentence describing the quality of the overlay",
  "concerns": ["list", "of", "specific", "mismatches"],
  "model_confidence": 0.0
}}

model_confidence should be a float from 0.0 to 1.0 reflecting your visual assessment of alignment quality.
Your role is advisory only. Do not output measurements or numeric values other than model_confidence."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "http://localhost:5173",
                    "X-Title": "AlignAI MVP",
                },
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            }},
                        ],
                    }],
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception as e:
        return {
            "summary": f"AI review failed: {str(e)[:120]}",
            "concerns": [],
            "model_confidence": None,
        }


def render_overlay_png(
    scan_pts_2d: "np.ndarray",
    plan_lines_2d: "np.ndarray",
    output_path: str,
    dpi: int = 150,
) -> str:
    """Render a top-down overlay PNG for the AI review step.

    scan_pts_2d: (N, 2) scan footprint in plan coordinates
    plan_lines_2d: (M, 2, 2) plan wall segments
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    # Plan lines — black
    if len(plan_lines_2d) > 0:
        segs = [[(s[0][0], s[0][1]), (s[1][0], s[1][1])] for s in plan_lines_2d]
        lc = mc.LineCollection(segs, colors="#111827", linewidths=1.0)
        ax.add_collection(lc)

    # Scan outline — red, semi-transparent
    if len(scan_pts_2d) > 0:
        ax.scatter(
            scan_pts_2d[:, 0], scan_pts_2d[:, 1],
            c="#e11d48", s=0.3, alpha=0.4, linewidths=0
        )

    # Auto-scale
    if len(plan_lines_2d) > 0:
        pts = plan_lines_2d.reshape(-1, 2)
        pad = max(pts.max(axis=0) - pts.min(axis=0)) * 0.05
        ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
        ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)

    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
