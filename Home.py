# SQLite edge case
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Required Libraries
import sqlite3
import openai
import os
import tempfile
import streamlit as st
import pandas as pd
import chromadb
import streamlit.components.v1 as components
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
import time

# Initialize vars
prompt_tokens = ""
completion_tokens = ""
total_tokens = ""
formatted_prompt = ""
execution_time = ""
docs = ""
kb1_db = None
kb2_db = None

# Initialize the Chroma DB and create the Collections
client = chromadb.Client()
kb1 = client.get_or_create_collection(name="kb1") # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
kb2 = client.get_or_create_collection(name="kb2")

# Check if the session_state has the knowledge_base_1 attribute. If not, initialize it.
if "knowledge_base_1" not in st.session_state:
    st.session_state.knowledge_base_1 = None

# Check if the session_state has kb1_db attribute. If not, initialize it.
if "kb1_db" not in st.session_state:
    st.session_state.kb1_db = None

# Check if the session_state has kb2_db attribute. If not, initialize it.
if "kb2_db" not in st.session_state:
    st.session_state.kb2_db = None

def load_data(sheets_url):
    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    return pd.read_csv(csv_url)

# Function to call OpenAI
def whole_openai_call(FormattedPrompt, model="gpt-4", temperature=1):
    try:
        start_time = time.time()  # Start the timer

        # Try to call the OpenAI API with the given model and temperature
        openai_response = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "user", "content": FormattedPrompt}
            ]
        )

        end_time = time.time()  # Stop the timer

        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        formatted_time = f"{minutes}m:{seconds}s"

        # Extract the message content and token usage from the OpenAI response
        message_content = openai_response['choices'][0]['message']['content']
        prompt_tokens = openai_response['usage']['prompt_tokens']
        completion_tokens = openai_response['usage']['completion_tokens']
        total_tokens = openai_response['usage']['total_tokens']

        return {
            "message_content": message_content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "execution_time": formatted_time  # Add the execution time to the returned dictionary
        }

    except openai.error.OpenAIError as e:
        print(f"An error occurred: {e}")
        return None

# Function to stringify Vector DB results
def combine_page_contents(documents):
    return "\n".join(doc.page_content for doc in documents)

def assemble_prompt(prompt_text, input_param_1_value=None, input_param_2_value=None, input_param_3_value=None, kb1_db=None, kb2_db=None, num_results=5):

    formatted_prompt = prompt_text

    # Add input parameters if they exist
    params = [input_param_1_value, input_param_2_value, input_param_3_value]
    for index, param in enumerate(params, start=1):
        if param:
            formatted_prompt += '\n\n[Param {}]\n\n{}'.format(index, param)

    # If there's anything in kb1, fetch the relevant context and append to the prompt
    if kb1_db:
        kb1_docs = kb1_db.similarity_search(query=prompt_text, k=num_results)
        stringified_kb1_docs = combine_page_contents(kb1_docs)
        formatted_prompt += '\n\n[Doc_Link_1]\n\n' + stringified_kb1_docs

    # Similarly, fetch context from kb2 if it's available
    if kb2_db:
        kb2_docs = kb2_db.similarity_search(query=prompt_text, k=num_results)
        stringified_kb2_docs = combine_page_contents(kb2_docs)
        formatted_prompt += '\n\n[Doc_Link_2]\n\n' + stringified_kb2_docs

    return formatted_prompt






# Main Streamlit App

# Image on sidebar
image = Image.open('fulllogo_nobuffer.jpg')
st.sidebar.image(image)

# Load Google Sheet data
df = load_data(st.secrets["public_gsheets_url"])

# Callback functions for Next and Previous
def onClickNext():
    if st.session_state.current_question_index < len(questions) - 1:
        st.session_state.current_question_index += 1

def onClickPrev():
    if st.session_state.current_question_index > 0:
        st.session_state.current_question_index -= 1

questions = df["Prompt_ID"].tolist()

# Initialize the session state
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0

# Select box
selected_question = st.sidebar.selectbox("Questions", questions, index=st.session_state.current_question_index)

# Sync the current_question_index with the selected value
st.session_state.current_question_index = questions.index(selected_question)

# Buttons
if st.session_state.current_question_index > 0:
    st.sidebar.button("Previous", on_click=onClickPrev)

if st.session_state.current_question_index < len(questions) - 1:
    st.sidebar.button("Next", on_click=onClickNext)

st.sidebar.write('---')




# PDF Uploader and Splitter
uploaded_pdf = st.sidebar.file_uploader("Knowledge Base 1") 
st.sidebar.write('---')
st.sidebar.write("PDF Splitting Params")
chunk_size = st.sidebar.number_input("Chunk Size", value=1000)
chunk_overlap = st.sidebar.number_input("Chunk Overlap", value=200)
num_results = st.sidebar.number_input("Num Results", value=4)

# Check if the uploaded_pdf exists and is different from the one in session_state
if uploaded_pdf and (st.session_state.knowledge_base_1 is None or uploaded_pdf.getvalue() != st.session_state.knowledge_base_1.getvalue()):
    with st.spinner('Processing the PDF...'):
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(uploaded_pdf.getvalue())
            
            # Delete anything currently in the collection
            client.delete_collection(name="kb1")
            kb1 = client.get_or_create_collection(name="kb1")
            st.session_state.kb1_db = None
            
            # Use the temporary file's path for PyPDFLoader
            loader = PyPDFLoader(tfile.name)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            documents = text_splitter.split_documents(pages)
            kb1_db = Chroma.from_documents(
                documents,
                embedding=OpenAIEmbeddings(),
                collection_name="kb1",
            )
            st.session_state.kb1_db = kb1_db

    # Update st.session_state.knowledge_base_1 after processing
    st.session_state.knowledge_base_1 = uploaded_pdf

# Retrieve kb1_db from session_state for use
kb1_db = st.session_state.kb1_db

# Prompt Name as Heading
matching_prompt_name_text = df[df["Prompt_ID"] == selected_question]["Prompt_ID"].iloc[0]
st.subheader(matching_prompt_name_text)

# Display Selected Question
matching_question_text = df[df["Prompt_ID"] == selected_question]["Question"].iloc[0]
st.subheader("Question")
st.markdown(f'<div style="text-align: right; font-size:28px; line-height:32px; margin-top:12px; margin-bottom:36px;">{matching_question_text}</div>', unsafe_allow_html=True)

# Display Prompt Text
selected_prompt_text = df[df["Prompt_ID"] == selected_question]["Prompt_text"].iloc[0]
st.subheader("Prompt")
prompt_text = st.text_area("Full Prompt", selected_prompt_text, height=400)

# Grab more info for the final dataframe
selected_prompt_index = df[df["Prompt_ID"] == selected_question]["Prompt Index"].iloc[0]

# Parameters
st.subheader("Parameters")
model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"], index=1)
temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, step=0.01, value=0.5)

# Additional Input Parameters
input_param_1_value = df[df["Prompt_ID"] == selected_question]["Input_Param 1"].iloc[0]
input_param_2_value = df[df["Prompt_ID"] == selected_question]["Input_Param 2"].iloc[0]
input_param_3_value = df[df["Prompt_ID"] == selected_question]["Input_Param 3"].iloc[0]
optional_input_1 = ""
optional_input_2 = ""
optional_input_3 = ""

# Check and display st.text_area for each Input_Param if it exists
if pd.notna(input_param_1_value):
    optional_input_1 = st.text_area("Input_Param 1", value=input_param_1_value)
    
if pd.notna(input_param_2_value):
    optional_input_2 = st.text_area("Input_Param 2", value=input_param_2_value)
    
if pd.notna(input_param_3_value):
    optional_input_3 = st.text_area("Input_Param 3", value=input_param_3_value)

secondary_pdf = st.file_uploader("Knowledge Base 2")

# Handling PDF processing after it's uploaded
if secondary_pdf:
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(secondary_pdf.getvalue())
        
        with st.spinner('Processing the PDF...'):
            
            # Delete anything currently in the collection
            client.delete_collection(name="kb2")
            kb1 = client.get_or_create_collection(name="kb2")
            st.session_state.kb2_db = None
            
            # Use the temporary file's path for PyPDFLoader
            loader = PyPDFLoader(tfile.name)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            documents = text_splitter.split_documents(pages)
            kb2_db = Chroma.from_documents(
              documents,
              embedding=OpenAIEmbeddings(),
              collection_name="kb2",
            )
            st.session_state.kb2_db = kb2_db

# Retrieve kb2_db from session_state for use
kb2_db = st.session_state.kb2_db




# Post button click logic

answer = ""
# Querying will be done when "Run Prompt" button is clicked
if st.button("Run Prompt"):
    formatted_prompt = assemble_prompt(prompt_text, optional_input_1, optional_input_2, optional_input_3, kb1_db, kb2_db)
    openai_response = whole_openai_call(formatted_prompt, model, temperature)
    if openai_response:
        answer = openai_response['message_content']
        prompt_tokens = openai_response['prompt_tokens']
        completion_tokens = openai_response['completion_tokens']
        total_tokens = openai_response['total_tokens']
        execution_time = openai_response['execution_time']

# Displaying the Result
st.subheader("Output")
output_text = st.text_area("Output", answer, height=400)

# Compute the word count for the "Output" and display the output dataframe
output_word_count = len(str(answer).split())

data = {
    "Prompt Index": [selected_prompt_index],
    "Prompt_ID": [matching_prompt_name_text],
    "Question": [matching_question_text],
    "Model": [model],
    "Temperature": [temperature],
    "Input Param 1": [optional_input_1],
    "Input Param 2": [optional_input_2],
    "Input Param 3": [optional_input_3],
    "Full Prompt Tokens": [total_tokens],
    "Output": [answer],
    "Output Tokens": [completion_tokens],
    "Output Characters": [len(str(answer))],
    "Output Word Count": [output_word_count]    # 2. Add the computed word count
}

output_data = pd.DataFrame(data)
st.subheader("Output Row For Copying")
output_2_row_df = st.dataframe(data)

# Counts
st.subheader("Counts and Full Prompt")
word_count = len(output_text.split())
st.write(f"Word Count: {word_count}")
st.write(f"Prompt Tokens: {prompt_tokens}")
st.write(f"Completion Tokens: {completion_tokens}")
st.write(f"Total Tokens: {total_tokens}")
st.write(f"Full Prompt: {formatted_prompt}")
st.write(f"Execution Time: {execution_time}")
