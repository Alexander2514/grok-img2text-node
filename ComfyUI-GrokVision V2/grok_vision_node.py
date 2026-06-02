"""
ComfyUI-GrokVision  v2.0
────────────────────────
Nodo para generar prompts fotorrealistas enviando una imagen a Grok Vision.
Incluye prompts fórmula predefinidos + modo custom.
"""

import base64
import io
import json
import requests
import numpy as np
from PIL import Image

# ── Modelos disponibles ────────────────────────────────────────────────
MODELS = [
    "grok-4.20-multi-agent-0309",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
]

# ── Prompts fórmula predefinidos ───────────────────────────────────────
FORMULA_PROMPTS = {
    "🎯 Custom (escribe el tuyo)": "",

    "📸 Pose Reference → Sasha": (
        "Analyze this image and extract ONLY the pose, body position, camera angle, "
        "shot type, and scene setting. Then build a complete photorealistic prompt "
        "replacing the subject with Sasha using her fixed traits from the system prompt. "
        "Keep the exact pose, angle and environment but swap the character entirely."
    ),

    "🌅 Scene + Outfit Description": (
        "Describe this image as a detailed photorealistic prompt. Focus on: "
        "shot type, camera angle and lens, lighting direction and quality, "
        "background/environment details, outfit (every garment with fabric, cut, color, fit), "
        "pose and expression. Output the prompt only, no preamble."
    ),

    "💡 Lighting & Mood Extraction": (
        "Extract only the lighting setup and mood from this image and describe it "
        "as a technical photography lighting prompt. Include: light source type, "
        "direction, quality (hard/soft), color temperature, shadows, atmosphere. "
        "Output as a comma-separated prompt segment to append to an existing prompt."
    ),

    "👗 Outfit Detail Extractor": (
        "Describe every garment and accessory visible in this image with maximum detail: "
        "fabric type, texture, color, cut, fit, how it interacts with the body. "
        "Output as a prompt segment only, starting directly with the clothing description."
    ),

    "🎬 Full Cinematic Prompt": (
        "Analyze this image and generate a complete cinematic photorealistic prompt. "
        "Structure: [shot type + lens + angle], [subject appearance], [pose + expression], "
        "[outfit details], [lighting], [background], [atmosphere + style]. "
        "No markdown, no preamble, output the prompt only."
    ),

    "🔄 img2img Variation": (
        "Describe this image as a detailed prompt that could recreate it with slight "
        "variations. Focus on composition, lighting, color palette, mood, and subject "
        "details. Make it suitable for img2img generation. Output the prompt only."
    ),
}

FORMULA_KEYS = list(FORMULA_PROMPTS.keys())

# ── System prompt por defecto (Sasha) ─────────────────────────────────
DEFAULT_SYSTEM = """# Role: Photorealistic Prompt Architect for Sasha

## Fixed Character Profile (ALWAYS include verbatim)
Character name: sasha
Physical traits: light greenish-gray eyes, black chin-length wavy bob haircut, vitiligo patches around mouth, extremely pale cool porcelain skin tone, flawless snow-white fair complexion with cool pinkish undertones, pure porcelain skin with no warmth or tan, full heavy breasts, narrow waist, wide hips, thick thighs and a round firm ass.

## Output Structure (always follow this exact order)
1. Character tag line: "sasha, sasha, [fixed physical traits], [scene setting]."
2. Shot: camera angle, lens (e.g. 50mm), focal point, depth of field.
3. Pose & action: body position, hands, expression, eye contact.
4. Outfit: fabric, cut, color, fit, how it interacts with her body.
5. Skin: always end with "Highly realistic pale skin texture with visible subtle pores and soft natural highlights."
6. Lighting & atmosphere.
7. Technical tail: style, grain, detail focus areas.

## Rules
- ALWAYS include ALL fixed character traits — never skip them.
- English only. No markdown in output. No preamble. Prompt only."""


# ── Helper: tensor → base64 JPEG ──────────────────────────────────────
def tensor_to_base64(image_tensor) -> str:
    img_np = (image_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode="RGB")
    # Resize si es muy grande (max 1536px)
    max_dim = 1536
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Nodo principal ─────────────────────────────────────────────────────
class GrokVisionPrompt:
    CATEGORY     = "GrokVision"
    FUNCTION     = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {
                    "default": "xai-tu-key-aqui",
                    "multiline": False,
                }),
                "model": (MODELS, {
                    "default": MODELS[2],  # non-reasoning por defecto (más rápido)
                }),
                "formula": (FORMULA_KEYS, {
                    "default": FORMULA_KEYS[1],  # Pose Reference → Sasha
                }),
            },
            "optional": {
                "custom_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Solo se usa si seleccionas '🎯 Custom' en formula",
                }),
                "system_prompt": ("STRING", {
                    "default": DEFAULT_SYSTEM,
                    "multiline": True,
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 4096,
                    "step": 128,
                }),
            },
        }

    def generate(
        self,
        image,
        api_key: str,
        model: str,
        formula: str,
        custom_instruction: str = "",
        system_prompt: str = DEFAULT_SYSTEM,
        max_tokens: int = 1024,
    ):
        # ── elegir instrucción ────────────────────────────────────────
        if formula == FORMULA_KEYS[0]:  # Custom
            instruction = custom_instruction.strip()
            if not instruction:
                instruction = "Describe this image as a detailed photorealistic prompt."
        else:
            instruction = FORMULA_PROMPTS[formula]

        # ── codificar imagen ──────────────────────────────────────────
        b64 = tensor_to_base64(image)

        # ── payload ───────────────────────────────────────────────────
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt.strip(),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": instruction,
                        },
                    ],
                },
            ],
        }

        # ── llamada API ───────────────────────────────────────────────
        try:
            resp = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data   = resp.json()
            result = data["choices"][0]["message"]["content"].strip()
            # Limpiar posibles backticks si el modelo los incluye
            result = result.strip("`").strip()

        except requests.exceptions.HTTPError as e:
            body = ""
            try:
                body = resp.json().get("error", {}).get("message", resp.text[:300])
            except Exception:
                body = resp.text[:300]
            result = f"[GrokVision ERROR] HTTP {resp.status_code}: {body}"

        except requests.exceptions.Timeout:
            result = "[GrokVision ERROR] Timeout — el modelo tardó más de 120s"

        except Exception as e:
            result = f"[GrokVision ERROR] {type(e).__name__}: {str(e)}"

        print(f"\n[GrokVision] modelo={model} | fórmula={formula}")
        print(f"[GrokVision] OUTPUT:\n{result[:200]}...\n")

        return (result,)


# ── Registro ───────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "GrokVisionPrompt": GrokVisionPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokVisionPrompt": "🔭 Grok Vision → Prompt",
}
