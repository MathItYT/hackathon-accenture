# Proyecto Hackathon Accenture
Aquí se encuentra nuestra solución al proyecto.

![Logo](abocato.png)

## ¿Cómo instalar el programa?
1. Clonar el repositorio
```bash
git clone https://github.com/MathItYT/hackathon-accenture.git
```

2. Instalar el **código y sus dependencias**
```bash
pip install .
```

3. Ir a la carpeta `hackaton-accenture` y crear un archivo `.env` con la siguiente variable de entorno:

```bash
OPENAI_API_KEY=<API_KEY>
```

Donde `<API_KEY>` es la clave de la API de OpenAI.

## ¿Cómo correr el programa?
```bash
python -m hackaton
```

## Notas
- El programa se demora en iniciar y tira muchos prints, pero funciona. 😁

- El programa es un chatbot que corre en el navegador, por la dirección indicada en la consola cuando termina de cargar.

Por ejemplo:
```bash
Running on local URL: http://127.0.0.1:<PORT>
```

Donde `<PORT>` es el puerto en el que corre el servidor.
