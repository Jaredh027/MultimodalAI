# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import os
import base64
import datetime
import argparse
import pandas as pd
from PIL import Image
from io import BytesIO

import streamlit as st
import streamlit_analytics
from streamlit_feedback import streamlit_feedback

from bot_config.utils import get_config
from utils.memory import init_memory, get_summary, add_history_to_memory
from guardrails.fact_check import fact_check
from llm.llm_client import LLMClient
from retriever.embedder import NVIDIAEmbedders, HuggingFaceEmbeders
from retriever.vector import MilvusVectorClient, QdrantClient
from retriever.retriever import Retriever
from utils.feedback import feedback_kwargs

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

llm_client = LLMClient("llama2_code_34b")

# get the config from the command line, or set a default
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help = "Provide a chatbot config to run the deployment")

st.set_page_config(
        page_title = "Multimodal RAG Assistant",
        page_icon = ":speech_balloon:",
        layout = "wide",
)

@st.cache_data()
def load_config(cfg_arg):
    try:
        config = get_config(os.path.join("bot_config", cfg_arg + ".config"))
        return config
    except Exception as e:
        print("Error loading config:", e)
        return None

args = vars(parser.parse_args())
cfg_arg = args["config"]

# Page title
st.header("Multimodal AI Assistant")
st.markdown("Upload an image of a product webpage and specify which company you want to taylor the product page to.")


# Initialize session state variables if not already present

if 'prompt_value' not in st.session_state:
    st.session_state['prompt_value'] = None

if cfg_arg and "config" not in st.session_state:
    st.session_state.config = load_config(cfg_arg)

if "config" not in st.session_state:
    st.session_state.config = load_config("multimodal")
    print(st.session_state.config)

if "messages" not in st.session_state:
    st.session_state.messages = [
            {"role": "assistant", "content": "Which company would you like to use for customization?"}
        ]

if "sources" not in st.session_state:
    st.session_state.sources = []

if "image_query" not in st.session_state:
    st.session_state.image_query = ""

if "queried" not in st.session_state:
    st.session_state.queried = False

if "memory" not in st.session_state:
    st.session_state.memory = init_memory(llm_client.llm, st.session_state.config['summary_prompt'])
memory = st.session_state.memory

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    prev_cfg = st.session_state.config
    try:
        defaultidx = [["multimodal"]].index(st.session_state.config["name"].lower())
    except:
        defaultidx = 0
    st.header("Bot Configuration")
    cfg_name = st.selectbox("Select a configuration/type of bot.", (["multimodal"]), index=defaultidx)
    st.session_state.config = get_config(os.path.join("bot_config", cfg_name+".config"))
    config = get_config(os.path.join("bot_config", cfg_name+".config"))
    if st.session_state.config != prev_cfg:
        st.experimental_rerun()

    st.success("Select an experience above.")

    st.header("Image Input Query")

    uploaded_file = st.file_uploader("Upload an image (JPG/JPEG/PNG) along with a text input:", accept_multiple_files=False)

    if uploaded_file:
        # Read the uploaded file into memory
        bytes_data = uploaded_file.getvalue()
        
        # Convert bytes data to a PIL Image
        image = Image.open(BytesIO(bytes_data))

        # Resize the image - example, reduce dimensions
        image = image.resize((image.width // 2, image.height // 2))

        # Reduce quality - re-encode the image to a lower quality JPEG
        mime_type = uploaded_file.type  # Example: "image/jpeg"
        img_format = mime_type[6:].upper()
        buffered = BytesIO()
        image.save(buffered, format=img_format, quality=20)
        smaller_bytes_data = buffered.getvalue()

        # Now encode this smaller image to base64
        b64_string = base64.b64encode(smaller_bytes_data).decode("utf-8")
        
        st.success("Image loaded for multimodal RAG Q&A.")
    
        with st.spinner("Getting image description using Fuyu"):
            fuyu = ChatNVIDIA(model="playground_fuyu_8b")
            if len(b64_string) % 4:
                b64_string += '=' * (4 - len(b64_string) % 4)

            base64_with_mime_type = f"data:{mime_type};base64,{b64_string}"

            #Description of webpage with products 
            res = fuyu.invoke(f'Describe the content as an html company product page with as much detail as possible. Include style components in detail. \n<img src="{base64_with_mime_type}" />')

            st.session_state.image_query = res.content
            st.write(st.session_state.image_query)
    if not uploaded_file:
        st.session_state.image_query = ""



from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader

# contains external document of company product information
loader = UnstructuredFileLoader("output.txt")
data = loader.load()

with open("output.txt") as f:
    external_document = f.read()

t_chunk_separators = ['\n\n', '\n']
t_chunk_size = 1000
t_chunk_overlap = 25
text_splitter = RecursiveCharacterTextSplitter(
    separators = t_chunk_separators,
    chunk_size=t_chunk_size, 
    chunk_overlap=t_chunk_overlap,
    length_function = len
    )

# splitting document to array of text
pages = text_splitter.split_text(external_document)

# creating split documents
texts = text_splitter.create_documents(pages)

embeddings = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="passage")

# FAISS vector db storage
db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()

# Check if the topic has changed
if st.session_state['prompt_value'] == None:
    prompt_value = ""
    st.session_state["prompt_value"] = prompt_value

colx, coly = st.columns([1,20])

prompt = st.chat_input("Hi there! Enter comapny name here.")

if prompt and st.session_state.image_query:
    

    user_query = f"Customize this webpage to use {prompt}'s products"

    #fetch the relevant documents from the vector db using the user_query
    with st.spinner("Obtaining relevant documents..."):
        docs = retriever.invoke(f"Get only {prompt}'s product information")

    prompt = f"\nWrite an html code replacing what is requested in the question: {user_query}" + " using this information: " + docs[0].page_content 
    transformed_query = {"text": prompt}

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = llm_client.chat_with_prompt("You are a helpful AI coding assistant", prompt)
        full_response = ""
        for chunk in response:
            full_response += chunk
            #message_placeholder.markdown(full_response + "▌")
        #message_placeholder.markdown(full_response)

    with st.chat_message("assistant"):
        st.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
elif prompt and not st.session_state.image_query:
    with st.chat_message("error"):
        st.markdown("You must upload and image first.")

