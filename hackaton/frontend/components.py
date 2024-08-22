import base64
from io import BytesIO
from pathlib import Path

import gradio as gr

from PIL import Image

import hackaton
from hackaton.backend.model import Model
from hackaton.utils.constants import CSS


def get_component(model: Model) -> gr.Blocks:
    """
    Crea el componente de gradio para el modelo
    """

    img = Image.open(Path(hackaton.__file__).parent.parent / "abocato.png")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    with gr.Blocks(css=CSS) as demo:
        gr.Interface(
            fn=model.input_processor,
            inputs=gr.Textbox(label="Pregunta", lines=5),
            outputs=gr.Textbox(label="Respuesta", lines=5),
            title="Hackathon Accenture - Solución del Equipo",
            description="Modelo de Lenguaje Natural que responde preguntas sobre IA generativa, ya sea sobre el reglamento de la UE, la política nacional chilena o los modelos de lenguaje en sí.",
        )
        gr.HTML(
            f'<div class="center"><img src="data:image/png;base64,{img_str}" class="right" width="200" height="200"/></div>'
        )
        return demo
