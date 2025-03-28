import os
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline


# Función para cargar y dividir documentos
def load_and_split_documents(folder_path):
    print('En load_and_split_documents')
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, file_name))
            documents.extend(text_splitter.split_documents(loader.load()))
    return documents

# Crear embeddings y almacenar en ChromaDB (modelo más potente)
def create_chroma_index(docs):
    model_name = "sentence-transformers/all-mpnet-base-v2"  # Modelo de embeddings más potente
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return Chroma.from_documents(docs, embedding=embeddings)

def setup_llm():
    print('Cargando modelo en local...')
    return pipeline("text2text-generation", model="google/flan-t5-base")

def ask_rag(query, vector_store, llm):
    print('En ask_rag')
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})
    docs = retriever.invoke(query)

    # Limitar el contexto para evitar entradas muy largas
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    print('-----------')
    print("Prompt enviado al modelo:", prompt)  # Depuración
    print('-----------')
    # Ejecutar el modelo en local
    respuesta = llm(prompt, max_new_tokens=240, temperature=0.5, do_sample=True)
    print(f'Respuesta RAW LLM: {respuesta}')
    return respuesta[0]['generated_text']

if __name__ == "__main__":
    print('Ejecutando MAIN')
    # Cargar documentos de la carpeta 'documentos'
    documents = load_and_split_documents("documentos")

    if not documents:
        raise ValueError("No se encontraron documentos para procesar.")

    # Crear índice ChromaDB
    vector_store = create_chroma_index(documents)

    # Configurar el modelo LLM
    llm = setup_llm()

    print("RAG configurado y listo para consultas")

    # Ejemplos de uso del RAG
    preguntas = [
        "What happened during the French Revolution?",
        "What is the main equation of relativity?",
        "What is artificial intelligence?",
        "What are the principles of existentialism?",
        "What are the effects of climate change?",
        "Who is Alain Sanchez Gonzalez?",
        "What do you know about Denia Peña Estilismo?"
    ]

    for pregunta in preguntas:
        print('************************************************')
        print(f'Pregunta: {pregunta}')
        respuesta = ask_rag(pregunta, vector_store, llm)
        print(f'Respuesta: {respuesta}')
        print('************************************************')