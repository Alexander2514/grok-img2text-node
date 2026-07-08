"""
Utilidades compartidas para los nodos GrokVision.
"""

import base64
import io
import requests
import numpy as np
from PIL import Image


def tensor_to_base64(image_tensor) -> str:
    """Convierte tensor de ComfyUI (B,H,W,C float32 0-1) a JPEG base64."""
    img_np = (image_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode="RGB")
    max_dim = 1536
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_xai(headers: dict, payload: dict):
    """
    Llama a https://api.x.ai/v1/chat/completions.
    Devuelve (result_str, None) en exito o (None, error_str) en fallo.
    """
    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        result = (
            resp.json()["choices"][0]["message"]["content"].strip().strip("`").strip()
        )
        return result, None
    except requests.exceptions.HTTPError:
        try:
            body = resp.json().get("error", {}).get("message", resp.text[:300])
        except Exception:
            body = resp.text[:300]
        return None, f"[GrokVision ERROR] HTTP {resp.status_code}: {body}"
    except requests.exceptions.Timeout:
        return None, "[GrokVision ERROR] Timeout — modelo tardo mas de 120s"
    except Exception as e:
        return None, f"[GrokVision ERROR] {type(e).__name__}: {str(e)}"
