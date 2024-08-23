# Importaciones estándar
from operator import itemgetter
from typing import Literal

# Importaciones de terceros
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Importaciones locales
from hackaton.utils import constants, environment


environment.load_env()


class Clasificador(BaseModel):
    """
    Clasifica el tipo de consulta para el uso especifico de texto
    (Y optimización de recursos)
    """

    type: Literal[
        "reglamento_ue", "politica_nacional_chilena_ia", "modelos_lenguaje"
    ] = Field(
        description="Dada la consulta recibida, clasifica si se trata de reglamento_ue o politica_nacional_chilena_ia o modelos_lenguaje",
    )


llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=environment.getenv("OPENAI_API_KEY"),
)

structured_llm = llm.with_structured_output(Clasificador)


def create_final_chain():
    """
    Crea la chain final para el modelo
    """

    # Preguntas clave para la clasificación de texto de manera exaustiva

    reglamento_ue_resume = ResumeTopic(
        constants.DATA_FOLDER / "reglamento_ue.txt",
        [
            "Responde en forma corta en no más de 30 palabras, ¿en qué enfoca legislativamente la UE al reglamentar las LLM?",
            "Responde brevemente (50 palabras) ¿Cuál es la importancia de la transparencia en los sistemas de IA?",
            "Responde brevemente (50 palabras) ¿cuales son los puntos mas importantes del reglamento UE 2024/1689 en cuanto a IA?",
            "Responde brevemente (50 palabras) ¿Que implicancias e politicas públicas tiene la IA y, es acorde a éstas mismas la reglamentacion de la UE?",
        ],
    )

    reglamento_ue_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"Recibirás una consulta sobre el Reglamento (UE) 2024/1689, responderás de forma cordial. Basate en el siguiente resumen para responder: {reglamento_ue_resume.resume()}",
            ),
            ("human", "{question}"),
        ]
    )

    plan_accion_ia_resume = ResumeTopic(
        constants.DATA_FOLDER / "plan_accion_ia.txt",
        [
            "Da una respuesta breve (50 palabras) ¿Qué implicancias en políticas públicas puede tener la IA con respecto a Chile, y cómo el plan de acción podría aplacar sus consecuencias?",
            "Responde brevemente (50 palabras) ¿Qué prácticas podrían pretender una reforma en dicho plan de acción?"
            "Responde brevemente (50 palabras), A rasgos generales ¿Qué busca hacer el plan de acción?",
            "Responde brevemente (en 50 palabras o menos): ¿Cómo aborda la Política Nacional de IA de Chile los impactos laborales de la automatización y cuál es el rol de la IA en la lucha contra la crisis climática?"
            "Responde brevemente (en 50 palabras o menos): ¿Qué medidas contempla la Política Nacional de IA de Chile para fortalecer la infraestructura tecnológica y cómo aborda la inclusión y no discriminación en la implementación de sistemas de inteligencia artificial?",
        ],
    )

    politica_nacional_ia_resume = ResumeTopic(
        constants.DATA_FOLDER / "politica_nacional_ia.txt",
        [
            "Da una respuesta breve (50 palabras), ¿Qué implicancias tiene la IA en las politicas públicas según el texto entregado?",
            "Responde brevemente (50 palabras) ¿Se puede extrapolar la IA para mejorar el ordenamiento juridico según lo entregado por el texto?",
            "Responde brevemente (50 palabras) ¿De qué manera la Política Nacional de IA de Chile aborda los impactos laborales que surgen de la automatización?",
            "Responde brevemente (en 50 palabras o menos): ¿Cómo aborda la Política Nacional de IA de Chile los impactos laborales de la automatización y cuál es el rol de la IA en la lucha contra la crisis climática?"
            "Responde brevemente (en 50 palabras o menos): ¿Qué medidas contempla la Política Nacional de IA de Chile para fortalecer la infraestructura tecnológica y cómo aborda la inclusión y no discriminación en la implementación de sistemas de inteligencia artificial?",
        ],
    )

    politica_nacional_ia_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"Recibirás una consulta sobre Política Nacional de IA de Chile, responderás de forma cordial. Basate en el siguiente resumen para responder:\n{politica_nacional_ia_resume.resume()}\n\n\n{plan_accion_ia_resume.resume()}",
            ),
            ("human", "{question}"),
        ]
    )

    modelos_lenguaje_resume = ResumeTopic(
        constants.DATA_FOLDER / "modelos_lenguaje.txt",
        [
            "Responde brevemente (en 50 palabras o menos): ¿Qué es un transformer en el contexto de los LLMs y cuáles son algunas de sus limitaciones?"
            "Responde brevemente (en 50 palabras o menos): ¿Qué es un sistema de Generación Aumentada por Recuperación (RAG) y qué técnicas se pueden utilizar para representar documentos y consultas en un RAG?"
            "Responde brevemente (en 50 palabras o menos): ¿Qué métodos existen para manejar consultas ambiguas en un sistema RAG y cómo se puede mejorar la coherencia entre la información recuperada y la generación de texto?"
            "Responde brevemente (en 50 palabras o menos): ¿Cómo se puede mitigar el sesgo en los LLMs y cómo se evalúa la calidad de respuesta en tareas de generación de texto?"
        ],
    )

    modelos_lenguaje_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"Recibirás una consulta sobre Modelos de Lenguaje, responderás de forma cordial. Basate en el siguiente resumen para responder:\n{modelos_lenguaje_resume.resume()}",
            ),
            ("human", "{question}"),
        ]
    )

    reglamento_ue_chain = reglamento_ue_prompt_template | llm | StrOutputParser()

    politica_nacional_ia_chain = (
        politica_nacional_ia_prompt_template | llm | StrOutputParser()
    )

    modelos_lenguaje_prompt_chain = (
        modelos_lenguaje_prompt_template | llm | StrOutputParser()
    )

    final_chain = RunnablePassthrough.assign(
        classification=(itemgetter("question") | structured_llm)
    ) | RunnablePassthrough.assign(  # Clasifica la pregunta
        output_text=(
            lambda x: (
                reglamento_ue_chain
                if x["classification"].type == "reglamento_ue"
                else (
                    politica_nacional_ia_chain
                    if x["classification"].type == "politica_nacional_chilena_ia"
                    else modelos_lenguaje_prompt_chain
                )
            )
        )
    )

    return final_chain


class ResumeTopic:
    def __init__(self, file_path, questions) -> None:
        self.questions = questions
        # Carga del documento
        loader = TextLoader(file_path)
        documents = loader.load()

        # Generación de Docs y Split
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Generación de embeddings
        embeddings = OpenAIEmbeddings(api_key=environment.getenv("OPENAI_API_KEY"))

        # Creación del vectorstore
        vectorstore = FAISS.from_documents(texts, embeddings)

        # Generate Retrieval
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def resume(self) -> str:
        resume_info = []
        for question in self.questions:
            docs = self.retriever.invoke(question)
            resume_info += [doc.page_content for doc in docs]
        return "\n\n".join(resume_info)


class Model:
    def __init__(self) -> None:
        self.final_chain = create_final_chain()

    def input_processor(self, input_text: str) -> str:
        """
        Será llamada por un método a decidir, procesará el input que entregue el usuario y entregará una respuesta
        """
        response = self.final_chain.invoke({"question": input_text})
        return response["output_text"]
