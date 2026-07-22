"""
ComfyUI-GrokVision  v4.2
────────────────────────
Modo IMAGEN  → imagen + formula → prompt fotorrealista para Flux/Sasha
Modo BOOSTER → prompt simple   → prompt potenciado para Flux/Sasha
"""

from .utils import tensor_to_base64, call_xai

# ── Modelos (fuente: docs.x.ai — julio 2026) ─────────────────────────────────
# Modelos con soporte de IMAGEN (vision) — julio 2026
# grok-2-vision-* retirados el 15/05/2026, ahora grok-4.5 soporta imagen
VISION_MODELS = [
    "grok-4.5",
    "grok-4.5-latest",
]
TEXT_MODELS = [
    "grok-4.3",
    "grok-4",
    "grok-4-fast",
    "grok-4-1-fast-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-3",
    "grok-3-fast",
    "grok-3-mini",
    "grok-3-mini-fast",
    "grok-build-0.1",
]
# Desplegable completo — el nodo elige automaticamente segun el modo
# Si el modo necesita vision y elegiste un modelo de texto -> usa grok-4.5
# Si el modo es solo texto -> usa el modelo que elegiste
ALL_MODELS = [
    "grok-4.5",              # vision + texto | RECOMENDADO para modo imagen
    "grok-4.5-latest",       # alias a la version mas reciente de 4.5
    "grok-4.3",              # texto/razonamiento profundo
    "grok-4",                # Grok 4 base
    "grok-4-fast",           # Grok 4 rapido, 2M ctx
    "grok-4-1-fast-reasoning",      # razonamiento rapido
    "grok-4-1-fast-non-reasoning",  # maxima velocidad sin razonamiento
    "grok-3",                # solido, menor costo
    "grok-3-fast",
    "grok-3-mini",           # economico, bueno para pruebas
    "grok-3-mini-fast",
    "grok-build-0.1",        # especializado en codigo
]

MODES = [
    "Image -> Prompt",
    "Prompt Booster",
]

FORMULA_PROMPTS = {
    "Custom": "",
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

DEFAULT_SYSTEM = """# Role: Photorealistic Prompt Architect for Sasha

## Fixed Character Profile (ALWAYS include verbatim)
Character name: sasha
Physical traits: light greenish-gray eyes, black chin-length wavy bob haircut,
vitiligo patches around mouth, extremely pale cool porcelain skin tone,
flawless snow-white fair complexion with cool pinkish undertones,
pure porcelain skin with no warmth or tan, full heavy breasts, narrow waist,
wide hips, thick thighs and a round firm ass.

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

BOOSTER_SYSTEM = """# Role: Flux2 LoRA Prompt Booster for Sasha

## Your Job
Receive a short or rough prompt, expand it into a powerful photorealistic prompt
that maximizes LoRA activation and visual quality for Flux2 + Sasha LoRA.

## Fixed Character Profile (MANDATORY)
LoRA triggers (start of prompt): sasha, sasha
Traits: light greenish-gray eyes, black chin-length wavy bob haircut,
vitiligo patches around mouth, extremely pale cool porcelain skin tone,
flawless snow-white fair complexion with cool pinkish undertones,
full heavy breasts, narrow waist, wide hips, thick thighs and a round firm ass.

## Output Structure (strict order)
1. "sasha, sasha, [all fixed physical traits],"
2. Scene & setting: environment, time of day.
3. Shot type & camera: focal length, angle, framing, depth of field.
4. Pose & action: body position, hands, gaze, expression.
5. Outfit: fabric, texture, color, cut, fit on her body.
6. "Highly realistic pale skin texture with visible subtle pores and soft natural highlights."
7. Lighting: source, direction, quality, color temperature, shadows.
8. Atmosphere & style: mood, grain, color palette, cinematic feel.
9. End with: "ultra-detailed, sharp focus, photorealistic, 8k, raw photo."

## Rules
- Build a full cinematic scene even from 3 words.
- Sasha traits NON-NEGOTIABLE, always full.
- No markdown, no preamble. Output the prompt only."""


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
                    {"default": "xai-tu-key-aqui", "multiline": False},
                ),
                "mode": (MODES, {"default": MODES[0]}),
                "model": (ALL_MODELS, {"default": "grok-4.5"}),
                "formula": (FORMULA_KEYS, {"default": FORMULA_KEYS[1]}),
                "max_tokens": (
                    "INT",
                    {"default": 1024, "min": 128, "max": 4096, "step": 128},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "input_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Prompt simple para el Booster",
                    },
                ),
                "booster_extra": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Instrucciones extra opcionales",
                    },
                ),
                "custom_instruction": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Instruccion custom (solo si formula=Custom)",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {"default": DEFAULT_SYSTEM, "multiline": True},
                ),
            },
        }

    def generate(
        self,
        api_key,
        mode,
        model,
        formula,
        max_tokens,
        image=None,
        input_prompt="",
        booster_extra="",
        custom_instruction="",
        system_prompt=DEFAULT_SYSTEM,
    ):

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }

        if mode == MODES[1]:  # Prompt Booster
            if not input_prompt.strip():
                return (
                    "[GrokVision] Booster: conecta un Text Multiline al input 'input_prompt'.",
                )
            active_model = model if model not in VISION_MODELS else "grok-4.3"  # vision no sirve para texto puro
            user_text = f"Input prompt: {input_prompt.strip()}"
            if booster_extra.strip():
                user_text += f"\n\nAdditional instructions: {booster_extra.strip()}"
            user_text += "\n\nExpand into a full Flux2 prompt. Output the prompt only."
            payload = {
                "model": active_model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": BOOSTER_SYSTEM},
                    {"role": "user", "content": user_text},
                ],
            }

        else:  # Image -> Prompt
            if image is None:
                return (
                    "[GrokVision] Image mode: conecta un LoadImage al input 'image'.",
                )
            # Si el modelo elegido no soporta vision, usa grok-4.5 automaticamente
            active_model = model if model in VISION_MODELS else "grok-4.5"
            instruction = (
                custom_instruction.strip()
                if formula == FORMULA_KEYS[0]
                else FORMULA_PROMPTS[formula]
            ) or "Describe this image as a detailed photorealistic prompt."
            payload = {
                "model": active_model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt.strip()},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{tensor_to_base64(image)}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": instruction},
                        ],
                    },
                ],
            }

        result, error = call_xai(headers, payload)
        if error:
            print(error)
            return (error,)

        print(f"\n[GrokVision] modo={mode} | modelo={active_model}")
        print(f"[GrokVision] OUTPUT:\n{result[:300]}...\n")
        return (result,)


NODE_CLASS_MAPPINGS = {"GrokVisionPrompt": GrokVisionPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"GrokVisionPrompt": "Grok Vision -> Prompt v4"}