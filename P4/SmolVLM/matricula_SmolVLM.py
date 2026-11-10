"""Reconocimiento de matrículas con HuggingFaceTB/SmolVLM-500M-Instruct."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


MODELO = "HuggingFaceTB/SmolVLM-500M-Instruct"
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
_processor = None
_model = None


def cargar_modelo():
    """Carga una vez el procesador y el modelo y los reutiliza en cada lectura."""
    global _processor, _model
    if _model is None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        _processor = AutoProcessor.from_pretrained(MODELO)
        _model = AutoModelForImageTextToText.from_pretrained(MODELO).to(DISPOSITIVO)
        _model.eval()
    return _processor, _model


def leer_matricula(img: np.ndarray) -> tuple[str, None, bool]:
    if img is None or img.size == 0:
        return "", None, False
    if img.ndim == 3 and img.shape[2] == 3:
        imagen = Image.fromarray(img[:, :, ::-1])
    else:
        imagen = Image.fromarray(img)

    processor, model = cargar_modelo()
    mensajes = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Read the license plate in this image. Return only the plate characters, without explanations."},
        ],
    }]
    prompt = processor.apply_chat_template(mensajes, add_generation_prompt=True)
    entradas = processor(text=prompt, images=imagen, return_tensors="pt").to(DISPOSITIVO)
    with torch.inference_mode():
        salida = model.generate(**entradas, max_new_tokens=32, do_sample=False)
    respuesta = processor.batch_decode(
        salida[:, entradas["input_ids"].shape[1]:], skip_special_tokens=True
    )[0]
    texto = "".join(caracter for caracter in respuesta.upper() if caracter.isalnum())
    return texto, None, len(texto) > 4
