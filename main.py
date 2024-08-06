import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain

def generate_response(
    uploaded_file,
    openai_api_key,
    query_text,
    response_text
):
    #formato del archivo cargado
    documents = [uploaded_file.read().decode()]
    
    #partirlo en trozos pequeños
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    texts = text_splitter.create_documents(documents)
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key
    )
    
    # crear un vectorstore y almacenar allí los textos
    db = FAISS.from_documents(texts, embeddings)
    
    # crear una interfaz recuperadora
    retriever = db.as_retriever()
    
    # crear un verdadero diccionario de control de calidad
    real_qa = [
        {
            "question": query_text,
            "answer": response_text
        }
    ]
    
    # cadena regular de control de calidad
    qachain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=retriever,
        input_key="question"
    )
    
    # predicciones
    predictions = qachain.apply(real_qa)
    
    # crear una cadena eval
    eval_chain = QAEvalChain.from_llm(
        llm=OpenAI(openai_api_key=openai_api_key)
    )
    # que se autocalifique
    graded_outputs = eval_chain.evaluate(
        real_qa,
        predictions,
        question_key="question",
        prediction_key="result",
        answer_key="answer"
    )
    
    response = {
        "predictions": predictions,
        "graded_outputs": graded_outputs
    }
    
    return response

st.set_page_config(
    page_title="Evaluar una aplicación RAG"
)
st.title("Evaluar una aplicación RAG")

st.write("Contacte con [Matias Toro Labra](https://www.linkedin.com/in/luis-matias-toro-labra-b4074121b/) para construir sus proyectos de IA")

with st.expander("Evaluar la calidad de un RAG APP"):
    st.write("""
        Para evaluar la calidad de una aplicación GAR, haremos lo siguiente
        preguntas para las que ya conocemos las
        respuestas reales.
        
        De esta manera podemos ver si la aplicación está produciendo
        las respuestas correctas o si está alucinando.
    """)

uploaded_file = st.file_uploader(
    "Cargar un documento .txt",
    type="txt"
)

query_text = st.text_input(
    "Introduzca una pregunta que ya haya comprobado:",
    placeholder="Escriba aquí su pregunta",
    disabled=not uploaded_file
)

response_text = st.text_input(
    "Introduzca la respuesta real a la pregunta:",
    placeholder="Escriba aquí la respuesta confirmada",
    disabled=not uploaded_file
)

result = []
with st.form(
    "myform",
    clear_on_submit=True
):
    openai_api_key = st.text_input(
        "OpenAI API Key:",
        type="password",
        disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text)
    )
    if submitted and openai_api_key.startswith("sk-"):
        with st.spinner(
            "Espera, por favor. Estoy trabajando en ello..."
            ):
            response = generate_response(
                uploaded_file,
                openai_api_key,
                query_text,
                response_text
            )
            result.append(response)
            del openai_api_key
            
if len(result):
    st.write("Pregunta")
    st.info(response["predictions"][0]["question"])
    st.write("Respuesta real")
    st.info(response["predictions"][0]["answer"])
    st.write("Respuesta proporcionada por la aplicación AI")
    st.info(response["predictions"][0]["result"])
    st.write("Por lo tanto, la respuesta de la AI App fue")
    st.info(response["graded_outputs"][0]["results"])

