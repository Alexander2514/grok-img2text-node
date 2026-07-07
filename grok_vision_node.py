"""
ComfyUI-GrokVision  v3.1
────────────────────────
Nodo para generar prompts fotorrealistas enviando:
  - Modo IMAGEN  → analiza una imagen y genera el prompt
  - Modo BOOSTER → recibe un prompt simple y lo potencia para Flux + LoRA Sasha
"""

import base64
import io
import requests
import numpy as np
from PIL import Image

# ── Modelos disponibles ────────────────────────────────────────────────
MODELS = [
    "grok-2-vision-latest",
    "grok-2-vision-1212",
    "grok-3-mini-beta",
    "grok-3-beta",
]

# ── Modos del nodo ─────────────────────────────────────────────────────
MODES = [
    "Image → Prompt",
    "Prompt Booster",
]

# ── Fórmulas para modo IMAGEN ──────────────────────────────────────────
FORMULA_PROMPTS = {
    "Custom (escribe el tuyo)": "",
    "Pose Reference -> Sasha": (
        "Analyze this image and extract ONLY the pose, body position, camera angle, "
        "shot type, and scene setting. Then build a complete photorealistic prompt "
        "replacing the subject with Sasha using her fixed traits from the system prompt. "
        "Keep the exact pose, angle and environment but swap the character entirely."
    ),
    "Scene + Outfit Description": (
        "Describe this image as a detailed photorealistic prompt. Focus on: "
        "shot type, camera angle and lens, lighting direction and quality, "
        "background/environment details, outfit (every garment with fabric, cut, color, fit), "
        "pose and expression. Output the prompt only, no preamble."
    ),
    "Lighting & Mood Extraction": (
        "Extract only the lighting setup and mood from this image and describe it "
        "as a technical photography lighting prompt. Include: light source type, "
        "direction, quality (hard/soft), color temperature, shadows, atmosphere. "
        "Output as a comma-separated prompt segment to append to an existing prompt."
    ),
    "Outfit Detail Extractor": (
        "Describe every garment and accessory visible in this image with maximum detail: "
        "fabric type, texture, color, cut, fit, how it interacts with the body. "
        "Output as a prompt segment only, starting directly with the clothing description."
    ),
    "Full Cinematic Prompt": (
        "Analyze this image and generate a complete cinematic photorealistic prompt. "
        "Structure: [shot type + lens + angle], [subject appearance], [pose + expression], "
        "[outfit details], [lighting], [background], [atmosphere + style]. "
        "No markdown, no preamble, output the prompt only."
    ),
    "img2img Variation": (
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

# ── System prompt para Booster ─────────────────────────────────────────
BOOSTER_SYSTEM = """# Role: Flux2 LoRA Prompt Booster for Sasha

## Your Job
You receive a short, simple, or rough prompt. Your task is to expand and restructure it
into a powerful, detailed photorealistic prompt that maximizes LoRA activation and
visual quality for the Flux2 model with the Sasha LoRA.

## Fixed Character Profile (MANDATORY — always inject ALL traits verbatim)
LoRA trigger words (must appear at the start): sasha, sasha
Physical traits: light greenish-gray eyes, black chin-length wavy bob haircut,
vitiligo patches around mouth, extremely pale cool porcelain skin tone,
flawless snow-white fair complexion with cool pinkish undertones,
full heavy breasts, narrow waist, wide hips, thick thighs and a round firm ass.

## Output Structure (strict order, no skipping)
1. LoRA triggers + character anchor:
   "sasha, sasha, [all fixed physical traits],"
2. Scene & setting: where is she, what is the environment, time of day.
3. Shot type & camera: focal length (e.g. 85mm portrait), angle, framing, depth of field.
4. Pose & action: exact body position, hands, gaze, expression.
5. Outfit: every garment — fabric, texture, color, cut, how it fits her body specifically.
6. Skin detail line (always include verbatim):
   "Highly realistic pale skin texture with visible subtle pores and soft natural highlights."
7. Lighting: source type, direction, quality (hard/soft), color temperature, shadows.
8. Atmosphere & style: mood, grain/film style, color palette, cinematic references if relevant.
9. Quality anchors (always end with):
   "ultra-detailed, sharp focus, photorealistic, 8k, raw photo."

## Expansion Rules
- Take whatever the user gives (even 3 words) and build a full cinematic scene.
- Infer missing details intelligently — if they say "at the beach", choose a time of day,
  lighting condition, outfit, and pose that make sense visually.
- The Sasha character traits are NON-NEGOTIABLE and must always appear in full.
- Never use markdown, headers, or bullet points in the output.
- Output the prompt only — no preamble, no explanation, no quotes around it."""


# ── Helper: tensor → base64 JPEG ──────────────────────────────────────
def tensor_to_base64(image_tensor) -> str:
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


# ── Nodo principal ─────────────────────────────────────────────────────
class GrokVisionPrompt:
    CATEGORY = "GrokVision"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "xai-tu-key-aqui",
                        "multiline": False,
                    },
                ),
                "mode": (
                    MODES,
                    {
                        "default": MODES[0],
                    },
                ),
                "model": (
                    MODELS,
                    {
                        "default": MODELS[0],
                    },
                ),
                "formula": (
                    FORMULA_KEYS,
                    {
                        "default": FORMULA_KEYS[1],
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 128,
                        "max": 4096,
                        "step": 128,
                    },
                ),
            },
            "optional": {
                # ── Modo IMAGEN ──────────────────────────────────────
                "image": ("IMAGE",),
                # ── Modo BOOSTER ─────────────────────────────────────
                "input_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Prompt simple para el booster (ej: sasha en la playa al atardecer)",
                    },
                ),
                "booster_extra": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Instrucciones extra opcionales (ej: enfocarse en el outfit)",
                    },
                ),
                # ── Compartidos ───────────────────────────────────────
                "custom_instruction": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Solo se usa si formula = Custom",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": DEFAULT_SYSTEM,
                        "multiline": True,
                    },
                ),
            },
        }

    def generate(
        self,
        api_key: str,
        mode: str,
        model: str,
        formula: str,
        max_tokens: int,
        image=None,
        input_prompt: str = "",
        booster_extra: str = "",
        custom_instruction: str = "",
        system_prompt: str = DEFAULT_SYSTEM,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }

        # ══════════════════════════════════════════════════════════════
        # MODO BOOSTER — prompt simple → prompt potenciado
        # ══════════════════════════════════════════════════════════════
        if mode == MODES[1]:  # "Prompt Booster"
            if not input_prompt.strip():
                return (
                    "[GrokVision ERROR] Modo Booster: conecta un Text Multiline al input 'input_prompt'.",
                )

            # Booster no necesita imagen ni visión — usa modelo de texto puro
            booster_model = model
            if "vision" in model.lower():
                # Preferir modelo de texto si el seleccionado es vision-only
                booster_model = "grok-3-mini-beta"

            user_text = f"Input prompt: {input_prompt.strip()}"
            if booster_extra.strip():
                user_text += f"\n\nAdditional instructions: {booster_extra.strip()}"
            user_text += (
                "\n\nExpand this into a full, powerful Flux2 prompt following your system rules. "
                "Output the prompt only."
            )

            payload = {
                "model": booster_model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": BOOSTER_SYSTEM},
                    {"role": "user", "content": user_text},
                ],
            }

        # ══════════════════════════════════════════════════════════════
        # MODO IMAGEN — imagen → prompt
        # ══════════════════════════════════════════════════════════════
        else:  # "Image → Prompt"
            if image is None:
                return (
                    "[GrokVision ERROR] Modo Image: conecta un LoadImage al input 'image'.",
                )

            # Modo imagen necesita modelo con visión
            vision_model = model
            if "vision" not in model.lower():
                vision_model = "grok-2-vision-latest"

            if formula == FORMULA_KEYS[0]:  # Custom
                instruction = custom_instruction.strip() or (
                    "Describe this image as a detailed photorealistic prompt."
                )
            else:
                instruction = FORMULA_PROMPTS[formula]

            b64 = tensor_to_base64(image)

            payload = {
                "model": vision_model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt.strip()},
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
                            {"type": "text", "text": instruction},
                        ],
                    },
                ],
            }

        # ── Llamada API ────────────────────────────────────────────────
        try:
            resp = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            result = data["choices"][0]["message"]["content"].strip()
            result = result.strip("`").strip()

        except requests.exceptions.HTTPError:
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

        print(f"\n[GrokVision] modo={mode} | modelo={model}")
        print(f"[GrokVision] OUTPUT:\n{result[:300]}...\n")

        return (result,)


# ── Registro ───────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "GrokVisionPrompt": GrokVisionPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokVisionPrompt": "Grok Vision -> Prompt",
}
