"""
ComfyUI-GrokKrea2Prompt  v1.1
──────────────────────────────
Nodo para generar prompts optimizados para Krea2.
Recibe texto o imagen, razona con Grok y devuelve un prompt
ultra-preciso listo para pegar en Krea2.
"""

from .utils import tensor_to_base64, call_xai

# ── Modelos ───────────────────────────────────────────────────────────────────
VISION_MODELS = [
    "grok-2-vision-latest",
    "grok-2-vision-1212",
]
TEXT_MODELS = [
    "grok-4.3",
    "grok-4",
    "grok-4-fast",
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-3",
    "grok-3-fast",
    "grok-3-mini",
    "grok-3-mini-fast",
]
ALL_MODELS = VISION_MODELS + TEXT_MODELS

INPUT_MODES = [
    "Text -> Krea2 Prompt",
    "Image -> Krea2 Prompt",
]

KREA2_STYLES = [
    "Photorealistic",
    "Cinematic Film",
    "Editorial Fashion",
    "Analog Film",
    "Oil Painting",
    "Digital Art",
    "Concept Art",
    "Watercolor",
    "Dark Fantasy",
    "Neon Cyberpunk",
    "None (solo descripcion)",
]

KREA2_SYSTEM = """# Role: Krea2 Prompt Engineer

## About Krea2
Krea2 is a generative AI platform (different from Flux/Stable Diffusion).
It does NOT use LoRA trigger words. It responds best to:
- Dense, concrete visual descriptions in natural language sentences
- Clear structure: Subject -> Composition -> Lighting -> Style
- Specific material textures, fabric details, skin tones, hair qualities
- Cinematic or artistic references when relevant
- Technical photography terms for photorealistic outputs
- Avoiding vague adjectives like "beautiful" or "stunning"

## Your Task
You receive either a short text prompt or an image description.
Reason carefully about the scene, then output ONE single Krea2-optimized prompt.

## Output Structure (always follow this order, no headers in output)
1. SUBJECT: Who/what, physical appearance, expression, key features.
2. ACTION & POSE: What are they doing, body position, hands, gaze direction.
3. OUTFIT & PROPS: Every garment and accessory with material, color, fit.
4. ENVIRONMENT: Location, time of day, background details.
5. COMPOSITION: Shot type (close-up/medium/wide), camera angle, focal length.
6. LIGHTING: Source, direction, quality (hard/soft), color temperature, shadows.
7. STYLE TAG: A short tail of 3-6 comma-separated style/quality descriptors.

## Critical Rules
- Output the prompt ONLY. No explanation, no markdown, no preamble.
- Write in flowing descriptive sentences for sections 1-6, comma-separated tags for 7.
- Be specific and concrete.
- Total length: 80-180 words.
- Never use LoRA trigger words (this is NOT Flux)."""


class GrokKrea2Prompt:
    CATEGORY = "GrokVision"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("krea2_prompt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {"default": "xai-tu-key-aqui", "multiline": False},
                ),
                "input_mode": (INPUT_MODES, {"default": INPUT_MODES[0]}),
                "model": (ALL_MODELS, {"default": "grok-4.3"}),
                "krea2_style": (KREA2_STYLES, {"default": KREA2_STYLES[0]}),
                "max_tokens": (
                    "INT",
                    {"default": 1024, "min": 128, "max": 4096, "step": 128},
                ),
            },
            "optional": {
                "input_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Describe la escena o personaje (ej: mujer en la playa al atardecer)",
                    },
                ),
                "image": ("IMAGE",),
                "extra_instructions": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Instrucciones extra opcionales (ej: enfocar en outfit, agregar lluvia)",
                    },
                ),
                "custom_system": (
                    "STRING",
                    {"default": KREA2_SYSTEM, "multiline": True},
                ),
            },
        }

    def generate(
        self,
        api_key,
        input_mode,
        model,
        krea2_style,
        max_tokens,
        input_text="",
        image=None,
        extra_instructions="",
        custom_system=KREA2_SYSTEM,
    ):

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }

        style_suffix = ""
        if krea2_style != "None (solo descripcion)":
            style_suffix = (
                f"\n\nThe style tag at the end MUST reflect: {krea2_style}. "
                f"Choose the most relevant descriptors for {krea2_style}."
            )

        if input_mode == INPUT_MODES[0]:  # Text -> Krea2
            if not input_text.strip():
                return ("[GrokKrea2] Conecta un Text Multiline al input 'input_text'.",)
            active_model = model if model not in VISION_MODELS else "grok-4.3"
            user_content = f"Input idea: {input_text.strip()}\n\nCreate an optimized Krea2 prompt.{style_suffix}"
            if extra_instructions.strip():
                user_content += f"\n\nExtra: {extra_instructions.strip()}"
            payload = {
                "model": active_model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": custom_system.strip()},
                    {"role": "user", "content": user_content},
                ],
            }

        else:  # Image -> Krea2
            if image is None:
                return ("[GrokKrea2] Conecta un LoadImage al input 'image'.",)
            active_model = model if model in VISION_MODELS else "grok-2-vision-latest"
            user_content_text = (
                "Analyze this image carefully. Generate an optimized Krea2 prompt "
                "that could recreate or reimagine this scene with maximum fidelity."
                f"{style_suffix}"
            )
            if extra_instructions.strip():
                user_content_text += f"\n\nExtra: {extra_instructions.strip()}"
            payload = {
                "model": active_model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": custom_system.strip()},
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
                            {"type": "text", "text": user_content_text},
                        ],
                    },
                ],
            }

        result, error = call_xai(headers, payload)
        if error:
            print(error)
            return (error,)

        print(
            f"\n[GrokKrea2] modo={input_mode} | modelo={active_model} | estilo={krea2_style}"
        )
        print(f"[GrokKrea2] OUTPUT:\n{result[:300]}...\n")
        return (result,)


NODE_CLASS_MAPPINGS = {"GrokKrea2Prompt": GrokKrea2Prompt}
NODE_DISPLAY_NAME_MAPPINGS = {"GrokKrea2Prompt": "Grok -> Krea2 Prompt"}
