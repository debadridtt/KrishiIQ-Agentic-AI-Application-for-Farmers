from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import plotly.express as px
from IPython.display import Markdown

import requests
import pandas as pd
import io


st.set_page_config(
        page_title="KrishiIQ | Crop Insights",
        page_icon="🌱",layout='wide', initial_sidebar_state="auto"
)

if st.button("Back to Home"):
    st.switch_page("main_app.py")
    
def hide_pages():
    st.markdown("""
        <style>
            section[data-testid="stSidebar"][aria-expanded="true"]{
                display: none;
            }
        </style>
        """, unsafe_allow_html=True)

hide_pages()

st.header('KrishiIQ🌱 Real-Time Market Analytics of Crops')

data_gov_api_key = "579b464db66ec23bdd000001170231baf36041b14075ed20651fb971"
data_gov_resource_id = "35985678-0d79-46b4-9ed6-6f13308a1d24"
data_gov_base_url = "https://api.data.gov.in/resource/"

col1, col2 = st.columns(2)

with col1:
    state_option = st.selectbox(
    "Please select State:",
    ("Maharashtra", "West Bengal", "Punjab"),
)

with col2:
    crop_option = st.selectbox(
    "Please select Crop:",
    ("Cotton", "Potato", "Rice"),
)

submit = st.button("Generate KrishiIQ🌱 Insights")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-pro",google_api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-pro")

def fetch_commodity_data(state, crop, data_gov_base_url=data_gov_base_url, data_gov_resource_id=data_gov_resource_id, data_gov_api_key=data_gov_api_key):
    api_url = f"{data_gov_base_url}{data_gov_resource_id}?api-key={data_gov_api_key}&format=csv&&filters[State]={state}&filters[Commodity]={crop}&limit=100000"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        print('Fetched ', data.shape[0], 'rows of data')

        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except pd.errors.EmptyDataError:
        print("No data to parse or CSV is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def map_role(role):
    if role == "model":
        return "assistant"
    elif role == "assistant":
        return "model"
    else:
        return role

def fetch_gemini_response(user_query):
    # Use the session's model to generate a response
    response = st.session_state.chat_session.model.generate_content(user_query)
    print(f"Gemini's Response: {response}")
    return response.parts[0].text

if submit:
    with st.spinner("Generating KrishiIQ🌱 insights..."):
        df = fetch_commodity_data(state_option, crop_option)
        print(df.shape)

        df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y')
        df = df.sort_values(by=['Arrival_Date'])
        df.reset_index(drop=True, inplace=True)
        df = df.drop_duplicates(subset=['Arrival_Date'])
        df.reset_index(drop=True, inplace=True)
        df = df[['State', 'Commodity', 'Arrival_Date', 'Min_Price', 'Max_Price', 'Modal_Price']]

        st.plotly_chart(px.line(df, x="Arrival_Date", y="Max_Price", title='Price of Potato'), use_container_width = True)

        details = model1.invoke(f"Given the above dataframe {df} can you forecast the Max_Price, Min_Price, Modal_Price for next 30 days?. Return only the forecasted numbers in a table format and suggest if the crop should be sold now or at a later time? Please don't share any Python code")
        readable_output = details.content
        st.dataframe(data=df.tail(5), use_container_width=True)
        st.write(readable_output)
        formatted_history = [
            {
                "role": "user",
                "parts": [readable_output]
            }
        ]

def role_to_streamlit(role):
  if role == "model":
    return "assistant"
  else:
    return role

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history = [])
     
for message in st.session_state.chat.history:
    with st.chat_message(role_to_streamlit(message.role)):
        st.markdown(Markdown(message.parts[0].text).data)

if prompt := st.chat_input("I possess knowledge related to real-estate market about Mumbai. What would you like to know?"):
    st.chat_message("user").markdown(prompt)
    response = st.session_state.chat.send_message(prompt) 
    with st.chat_message("assistant"):
        st.markdown(response.text)
