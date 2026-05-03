import chainlit as cl
import ollama
import chromadb
import os, uuid
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

## FASE 1 - Lettura Files e Chunking

documents_dir = "resumes"

documents = []
metadatas = []
ids = []

for filename in os.listdir(documents_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(documents_dir, filename), "r") as file:
            chunks = file.read().replace("\n", ".").split("### ")
            for chunk in chunks:
                if not chunk.isspace() and not chunk == "":
                    documents.append(chunk)
                    metadatas.append({"source": filename})
                    guid = str(uuid.uuid4())
                    ids.append(guid)

## FASE 2 - Embeddings e inserimento nel DB Vettoriale

openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="CVs", embedding_function=openai_ef
)

collection.add(documents=documents, metadatas=metadatas, ids=ids)

## FASE 3 - CHAT

@cl.on_chat_start
def on_chat_start():
    cl.user_session.set(
        "messages",
        [
            {
                "role": "system",
                "content": """
                Sei un assistente specializzato nel mondo HR, rispondi in modo professionale, sintetico e pragmatico. Il tuo ruolo è individuare il candidato ideale rispetto alle richieste dell'utente.
                """,
            }
        ],
    )


@cl.on_message
async def handle_message(message: cl.Message):
    user_question = message.content

    results = collection.query(query_texts=[user_question], n_results=1)

    def leggi_prime_100_righe(file_path):
        with open(file_path, "r") as file:
            righe = []
            for i, riga in enumerate(file):
                if i < 100:
                    righe.append(riga.strip())
                else:
                    break
        return righe

    filename = results["metadatas"][0][0]["source"]
    context_nome_candidato = leggi_prime_100_righe(
        os.path.join(documents_dir, filename)
    )

    nome = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": f"""
                Dato il seguente contesto individua nome e cognome del candidato e ritorna solo nome e cognome del candidato. Quello che sto per fornirti è il curriculum vitae del candidato: {context_nome_candidato}
                """,
            }
        ],
    )

    nome = nome["message"]["content"]

    context = f"CONTESTO: nome file {results['metadatas'][0][0]['source']} ecco il paragrafo più significativo: {results['documents'][0][0]}"

    prompt = f"""
        Dato il seguente contesto: 
        [[[
        {context}
        ]]].
        Rispondi alla domanda dell'utente: [[[ {user_question} ]]].
        Spiega che nel file individuato c'è il profilo più adatto. 
        Assicurati di nominare il nome del file.
        Assicurati di indicare il nome del candidato: [[[ {nome} ]]].
        Argomenta la scelta utilizzando il contenuto del testo individuato nel contesto.
        Se non trovi corrispondenza in nessun cv non inventare."""

    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": prompt})

    response_message = cl.Message(content="")
    await response_message.send()

    try:
        stream = ollama.chat(model="llama3.2", messages=messages, stream=True)

        for chunk in stream:
            await response_message.stream_token(chunk["message"]["content"])

        messages.append({"role": "assistant", "content": response_message.content})
        await response_message.update()
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        cl.Message(content=error_message).send()
        print(error_message)

    cl.user_session.set("messages", messages)


@cl.on_chat_end
def on_chat_end():
    cl.Message(
        content="""
        Grazie per aver utilizzato il nostro assistente. 
        Buona giornata!
        """
    ).send()