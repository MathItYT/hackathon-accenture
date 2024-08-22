import os

from dotenv import load_dotenv

from hackaton.utils.constants import DOTENV_FILE


def load_env():
    """
    Carga las variables de entorno del archivo .env
    """
    load_dotenv(str(DOTENV_FILE))


def getenv(key: str) -> str:
    """
    Obtiene el valor de una variable de entorno

    Args:
        key (str): Nombre de la variable de entorno

    Returns:
        str: Valor de la variable de entorno
    """
    return os.getenv(key)
