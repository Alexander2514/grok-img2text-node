# ComfyUI-GrokVision

Custom node que envía una imagen + instrucción a **xAI Grok Vision**
y devuelve un prompt fotorrealista listo para usar en CLIPTextEncode.

---

## Instalación en RunPod

```bash
cd /workspace/ComfyUI/custom_nodes
git clone <tu-repo> ComfyUI-GrokVision
# o copia la carpeta manualmente y luego:
pip install -r ComfyUI-GrokVision/requirements.txt --break-system-packages
```

Reinicia ComfyUI.

---

## Nodo: 🔭 Grok Vision → Prompt

### Entradas

| Campo | Tipo | Descripción |
|---|---|---|
| `image` | IMAGE | Imagen de referencia (de LoadImage u otro nodo) |
| `api_key` | STRING | Tu key de xAI (`xai-...`) |
| `instruction` | STRING | Qué quieres que Grok describa o enfoque |
| `system_prompt` | STRING | Rasgos fijos del personaje / estilo |
| `model` | STRING | Modelo Grok (default: `grok-2-vision-1212`) |
| `max_tokens` | INT | Longitud máxima del prompt (default: 1024) |

### Salida

| Campo | Tipo | Descripción |
|---|---|---|
| `prompt` | STRING | Prompt listo para CLIPTextEncode |

---

## Flujo básico

```
[LoadImage] ──────────────────────────────────────┐
                                                   ▼
[Text Multiline: system_prompt con Sasha] ──▶ [🔭 Grok Vision → Prompt]
                                                   │
                                                   ▼
                                          [CLIPTextEncode]
                                                   │
                                                   ▼
                                             [Sampler]
```

---

## System prompt recomendado para Sasha

Pega esto en el campo `system_prompt`:

```
# Role: Photorealistic Prompt Architect for Sasha

## Fixed Character Profile (ALWAYS include verbatim)
Character name: sasha
Physical traits: light greenish-gray eyes, black chin-length wavy bob haircut,
vitiligo patches around mouth, extremely pale cool porcelain skin tone,
flawless snow-white fair complexion with cool pinkish undertones,
pure porcelain skin with no warmth or tan, full heavy breasts, narrow waist,
wide hips, thick thighs and a round firm ass.

## Output Structure
1. Character tag line: "sasha, sasha, [fixed physical traits], [scene setting]."
2. Shot: camera angle, lens, focal point, depth of field.
3. Pose & action: body position, hands, expression, eye contact.
4. Outfit: fabric, cut, color, fit, interaction with body.
5. Skin: always end with "Highly realistic pale skin texture with visible subtle pores."
6. Lighting & atmosphere.
7. Technical tail: style, grain, detail areas.

## Rules
- ALWAYS include ALL fixed character traits — never skip them.
- English only. No markdown in output. No preamble. Prompt only.
```

---

## Instrucciones de ejemplo

- `"Describe the pose, outfit and setting. Integrate Sasha's fixed traits."`
- `"Generate a prompt based on this reference image, keeping Sasha's character."`
- `"Use this image as a pose reference. Keep Sasha's physical traits intact."`
