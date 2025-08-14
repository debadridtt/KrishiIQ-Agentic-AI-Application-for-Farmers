from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import plotly.express as px
import requests
import pandas as pd
import io
import json # Import json for parsing LLM output
import re # Import regex for fallback parsing

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool # Standard tool decorator
from langchain_community.tools import DuckDuckGoSearchRun # For general web search knowledge

# Import for Speech-to-Text and Text-to-Speech
from google.cloud import texttospeech as tts
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment
from st_audiorec import st_audiorec
import hashlib # For audio hashing
from langdetect import detect # For language detection
from deep_translator import GoogleTranslator # For text translation

st.set_page_config(
    page_title="KrishiIQ | Crop Insights",
    page_icon="🌱", layout='wide', initial_sidebar_state="auto"
)

# Move these button/page switches to a more appropriate place if they interfere with main flow
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

data_gov_api_key = os.getenv("DATA_GOV_API_KEY")
data_gov_resource_id = os.getenv("DATA_GOV_RESOURCE_KEY")
data_gov_base_url = "https://api.data.gov.in/resource/"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

main_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.2)
model_for_forecasting = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
llm_for_entity_extraction = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1)


# --- Helper Functions for Speech and Translation ---

def text_to_speech_vertex(text, lang_code="en-US"):
    """Converts text to speech using Google Cloud Text-to-Speech API."""
    client = tts.TextToSpeechClient()
    synthesis_input = tts.SynthesisInput(text=text)
    
    voice = None # Initialize voice to None
    
    if lang_code == 'hi':
        tts_lang_code = 'hi-IN'
        voice = tts.VoiceSelectionParams(language_code=tts_lang_code, name='hi-IN-Neural2-A', ssml_gender=tts.SsmlVoiceGender.FEMALE) 
    elif lang_code == 'bn':
        tts_lang_code = 'bn-IN'
        voice = tts.VoiceSelectionParams(language_code=tts_lang_code, name='bn-IN-Neural2-A', ssml_gender=tts.SsmlVoiceGender.FEMALE) 
    elif lang_code == 'en':
        tts_lang_code = 'en-IN' 
        voice = tts.VoiceSelectionParams(language_code=tts_lang_code, name='en-IN-Wavenet-D', ssml_gender=tts.SsmlVoiceGender.FEMALE) 
    else:
        tts_lang_code = f"{lang_code}-{lang_code.upper()}" if len(lang_code) == 2 else 'en-US' 
        try:
            voice = tts.VoiceSelectionParams(language_code=tts_lang_code, ssml_gender=tts.SsmlVoiceGender.FEMALE)
        except Exception:
            voice = tts.VoiceSelectionParams(language_code=tts_lang_code, ssml_gender=tts.SsmlVoiceGender.NEUTRAL)
            
        if lang_code == 'es': voice = tts.VoiceSelectionParams(language_code='es-ES', ssml_gender=tts.SsmlVoiceGender.FEMALE)
        elif lang_code == 'fr': voice = tts.VoiceSelectionParams(language_code='fr-FR', ssml_gender=tts.SsmlVoiceGender.FEMALE)
        elif lang_code == 'de': voice = tts.VoiceSelectionParams(language_code='de-DE', ssml_gender=tts.SsmlVoiceGender.FEMALE)

    if voice is None:
        voice = tts.VoiceSelectionParams(language_code='en-US', name='en-US-Neural2-C', ssml_gender=tts.SsmlVoiceGender.FEMALE)


    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)
    
    try:
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        st.error(f"Text-to-Speech error for language {lang_code} (voice setup: {voice}): {e}. Trying with default English voice as fallback.")
        voice = tts.VoiceSelectionParams(language_code='en-US', name='en-US-Neural2-C', ssml_gender=tts.SsmlVoiceGender.FEMALE) 
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content


def speech_to_text_vertex(audio_bytes, primary_speech_lang_code="en-IN"):
    """Converts speech audio bytes to text using Google Cloud Speech-to-Text API.
    `primary_speech_lang_code` guides the script of the transcription.
    """
    client = speech.SpeechClient()

    audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1) 
    if audio_segment.frame_rate != 16000: 
        audio_segment = audio_segment.set_frame_rate(16000)

    processed_audio_bytes_io = io.BytesIO()
    audio_segment.export(processed_audio_bytes_io, format="wav") 
    processed_audio_bytes = processed_audio_bytes_io.getvalue()

    audio = speech.RecognitionAudio(content=processed_audio_bytes)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, 
        sample_rate_hertz=16000,
        language_code=primary_speech_lang_code, 
        enable_automatic_punctuation=True,
        alternative_language_codes=['hi-IN', 'bn-IN', 'en-IN', 'en-US'] 
    )
    
    with st.spinner(f"Transcribing your speech... (Primary language code: {primary_speech_lang_code})"):
        response = client.recognize(config=config, audio=audio)
    
    if response.results:
        return response.results[0].alternatives[0].transcript
    return "Sorry, could not understand speech."

def detect_language(text):
    """Detects the language of a given text, with a fallback."""
    if not text:
        return 'en'
    try:
        return detect(text)
    except:
        return 'en' # Default to English if detection fails

def translate_text(text, source_lang, target_lang):
    """Translates text from source to target language using GoogleTranslator."""
    if not text:
        return ""
    if source_lang == target_lang:
        return text
    
    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated
    except Exception as e:
        st.warning(f"Translation error from {source_lang} to {target_lang}: {e}. Returning original text.")
        return text

@st.cache_data(ttl=3600)
def fetch_commodity_data(state, crop, data_gov_base_url=data_gov_base_url, data_gov_resource_id=data_gov_resource_id, data_gov_api_key=data_gov_api_key):
    """Fetches commodity data from Data.gov.in API."""
    api_url = f"{data_gov_base_url}{data_gov_resource_id}?api-key={data_gov_api_key}&format=csv&&filters[State]={state}&filters[Commodity]={crop}&limit=100000"

    try:
        response = requests.get(api_url)
        response.raise_for_status() 
        data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        return data

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}. Please check your API key and internet connection.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.warning("No data to parse or CSV is empty for the selected criteria. Please try different selections.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching or parsing data: {e}")
        return pd.DataFrame()

@tool
def general_knowledge_query(query: str) -> str:
    """Answers general knowledge questions that are NOT related to the loaded crop market data.
    Use this tool when the user's question is about common facts, definitions, or anything
    that doesn't require analyzing the specific dataframe.
    """
    try:
        response = main_llm.invoke(f"Answer the following question: {query}")
        return response.content
    except Exception as e:
        return f"Error answering general knowledge question: {e}"

search = DuckDuckGoSearchRun()
@tool
def web_search_tool(query: str) -> str:
    """Performs a web search to find information. Use this tool for questions that require
    up-to-date information, specific facts, or external data that is not present
    in the loaded crop market data or general knowledge.
    Examples: 'What is the current price of rice?', 'News about agriculture policies in India'.
    """
    try:
        return search.run(query)
    except Exception as e:
        return f"Error performing web search: {e}"


# --- Initialize session state variables for chatbot and data ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df_data" not in st.session_state:
    st.session_state.df_data = pd.DataFrame()
if "main_agent_executor" not in st.session_state:
    st.session_state.main_agent_executor = None
if "chart_data" not in st.session_state:
    st.session_state.chart_data = None
if "forecast_text" not in st.session_state:
    st.session_state.forecast_text = ""
if "tail_df_display" not in st.session_state:
    st.session_state.tail_df_display = pd.DataFrame()
if "current_state" not in st.session_state:
    st.session_state.current_state = "West Bengal" 
if "current_crop" not in st.session_state:
    st.session_state.current_crop = "Cotton" 
if "main_agent_memory" not in st.session_state: 
    st.session_state.main_agent_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
if 'user_input_mode' not in st.session_state:
    st.session_state.user_input_mode = "Text" 
if 'user_input_lang_code' not in st.session_state:
    st.session_state.user_input_lang_code = 'en' 
if 'dropdowns_populated' not in st.session_state:
    st.session_state.dropdowns_populated = False 
if 'dropdown_input_lang_code' not in st.session_state:
    st.session_state.dropdown_input_lang_code = 'en' 
if 'auto_trigger_pending' not in st.session_state:
    st.session_state.auto_trigger_pending = False
if 'state_crop_input_mode' not in st.session_state:
    st.session_state.state_crop_input_mode = "Speech" # Default to Speech

# --- New session state variables for chat input persistence ---
if 'user_chat_input_text' not in st.session_state:
    st.session_state.user_chat_input_text = ""
if 'user_chat_input_audio_bytes' not in st.session_state:
    st.session_state.user_chat_input_audio_bytes = None
if 'process_chat_input' not in st.session_state:
    st.session_state.process_chat_input = False


def initialize_main_conversational_agent(df_to_analyze, llm, memory):
    pandas_agent = create_pandas_dataframe_agent(
        llm,
        df_to_analyze,
        verbose=True,
        agent_type="tool-calling",
        allow_dangerous_code=True,
    )

    @tool
    def analyze_crop_data(query: str) -> str:
        """Use this tool ONLY for questions that require analyzing or extracting information
        from the LOADED CROP MARKET DATA.
        Examples:
        - 'What is the average Max_Price?'
        - 'Show me the data for January 2024.'
        - 'What is the range of Modal_Price?'
        - 'Find the highest Min_Price and its date.'
        - 'Summarize the price trends.'
        """
        try:
            response = pandas_agent.invoke({"input": query})
            return response['output']
        except Exception as e:
            return f"Error analyzing crop data: {e}. The dataframe might not contain information relevant to your question or the query format was unclear for data analysis."

    tools = [analyze_crop_data, general_knowledge_query, web_search_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are KrishiIQ🌱, an AI assistant specializing in crop market analytics. "
                         "You have access to specific tools to answer questions. "
                         "**Prioritize using the 'analyze_crop_data' tool for any questions related to the loaded crop market data.** "
                         "Use 'general_knowledge_query' for common facts or definitions. "
                         "Use 'web_search_tool' for current events, specific facts not in your general knowledge, or external information. "
                         "Always try to be helpful and provide concise answers. If a question is about the loaded data, ensure the 'analyze_crop_data' tool is called."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory, 
        handle_parsing_errors=True
    )
    return agent_executor

# --- Function to handle data generation and insights (auto-triggered or by button) ---
def generate_krishiiq_insights(state, crop, input_lang_code):
    """
    Fetches data, generates charts, forecasts, and initializes the chatbot agent.
    This function can be called automatically after speech input or by the manual button.
    """
    # CLEAR specific session state variables when NEW data is requested
    st.session_state.messages = [] 
    st.session_state.chart_data = None
    st.session_state.forecast_text = ""
    st.session_state.tail_df_display = pd.DataFrame()
    st.session_state.df_data = pd.DataFrame() 
    st.session_state.main_agent_executor = None 
    st.session_state.main_agent_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
    
    st.session_state.user_input_lang_code = input_lang_code


    with st.spinner(f"Generating KrishiIQ🌱 insights for {crop} in {state}..."):
        df = fetch_commodity_data(state, crop)

        if not df.empty:
            df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y', errors='coerce')
            df.dropna(subset=['Arrival_Date'], inplace=True)
            df = df.sort_values(by=['Arrival_Date'])
            df.reset_index(drop=True, inplace=True)
            df = df.drop_duplicates(subset=['Arrival_Date'])
            df.reset_index(drop=True, inplace=True)
            
            if 'Modal_Price' not in df.columns:
                 st.warning("Modal_Price column not found in fetched data. Calculating it as average of Min and Max price.")
                 df['Modal_Price'] = (df['Min_Price'] + df['Max_Price']) / 2 
                 
            df = df[['State', 'Commodity', 'Arrival_Date', 'Min_Price', 'Max_Price', 'Modal_Price']] 

            st.session_state.df_data = df
            st.session_state.current_state = state 
            st.session_state.current_crop = crop 
            
            fig = px.line(df, x="Arrival_Date", y="Max_Price", title=f'Max Price of {crop} in {state} Over Time')
            st.session_state.chart_data = fig

            # Forecast and Recommendation (Multilingual Output)
            forecast_prompt_en = f"""
            Given the dataframe with columns: {', '.join(df.columns)}, and last few rows:
            {df.tail().to_markdown(index=False)}

            Please provide a 30-day forecast for Min_Price, Max_Price, and Modal_Price.
            Return the forecast in a markdown table format.
            Also, provide a recommendation on when the crop should be sold, including a tentative date and brief reasoning.

            Format your response like this:
            ### Forecast Table
            <MARKDOWN_TABLE_HERE>

            ### Recommendation
            **When to Sell:** <Your recommendation text here, including tentative date>
            **Reasoning:** <Brief reasoning for the recommendation>
            """
            
            with st.spinner("Generating forecast and recommendation..."):
                forecast_llm_response_en = model_for_forecasting.invoke(forecast_prompt_en).content

            # --- Parsing and Translating LLM Output ---
            forecast_parts = re.split(r'###\s*(Forecast Table|Recommendation)', forecast_llm_response_en, flags=re.IGNORECASE)
            
            forecast_table_str_en = ""
            recommendation_text_en = ""

            if len(forecast_parts) > 2 and forecast_parts[1].lower().strip() == "forecast table":
                forecast_table_str_en = forecast_parts[2].strip()
            if len(forecast_parts) > 4 and forecast_parts[3].lower().strip() == "recommendation":
                recommendation_text_en = forecast_parts[4].strip()

            translated_recommendation = translate_text(recommendation_text_en, "en", input_lang_code)
            
            final_forecast_output = ""
            if forecast_table_str_en:
                final_forecast_output += f"### {translate_text('Forecast Table', 'en', input_lang_code)}\n"
                final_forecast_output += forecast_table_str_en + "\n\n" 
            if translated_recommendation:
                final_forecast_output += f"### {translate_text('Recommendation', 'en', input_lang_code)}\n"
                final_forecast_output += translated_recommendation.replace('*', '')

            st.session_state.forecast_text = final_forecast_output

            forecast_audio = text_to_speech_vertex(translated_recommendation.replace('*', '')[:1000], lang_code=input_lang_code)
            st.session_state.forecast_audio_bytes = forecast_audio 

            st.session_state.tail_df_display = df.tail(5)

            st.session_state.main_agent_executor = initialize_main_conversational_agent(
                st.session_state.df_data,
                main_llm,
                st.session_state.main_agent_memory 
            )

            initial_chatbot_message_en = f"I have loaded the market data for {st.session_state.current_crop} in {st.session_state.current_state}. You can now ask me questions about this data (e.g., 'What is the average max price?'), or general knowledge questions (e.g., 'What is ideal soil condition for the crop?')."
            translated_initial_message = translate_text(initial_chatbot_message_en, "en", input_lang_code)
            initial_message_audio = text_to_speech_vertex(translated_initial_message.replace('*', '')[:1000], lang_code=input_lang_code)
            
            st.session_state.messages.append(AIMessage(content=translated_initial_message.replace('*', ''), additional_kwargs={"audio": initial_message_audio}))

        else:
            st.warning("No data found for the selected State and Crop. Please try different selections.")
            st.session_state.df_data = pd.DataFrame()
            st.session_state.main_agent_executor = None
            st.session_state.chart_data = None
            st.session_state.forecast_text = ""
            st.session_state.tail_df_display = pd.DataFrame()
            st.session_state.forecast_audio_bytes = None
            st.session_state.main_agent_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
            st.session_state.messages = []


# --- UI for setting State and Crop ---
with st.container():
    st.subheader("Set Crop Market Data")
    st.session_state.state_crop_input_mode = st.radio(
        "Choose Input Mode for State/Crop:", ["Speech", "Text"], key="state_crop_input_mode_radio",
        horizontal=True
    )

    identified_state_from_input = None
    identified_crop_from_input = None
    input_language_for_dropdowns = st.session_state.user_input_lang_code 

    if st.session_state.state_crop_input_mode == "Speech":
        st.write("Click the microphone button and say the State and Crop you want to analyze (e.g., 'Show me data for Cotton in Maharashtra').")
        
        if not st.session_state.dropdowns_populated:
            speech_input_lang_choice = st.selectbox(
                "Select Language for Speech Input:",
                options=['English', 'हिन्दी', 'Bengali'],
                index=0, 
                key="speech_input_lang_select_dropdown"
            )
            stt_lang_code_for_dropdown = {
                'English': 'en-IN',
                'हिन्दी': 'hi-IN',
                'বাংলা': 'bn-IN'
            }.get(speech_input_lang_choice, 'en-IN')

            speech_input_for_dropdowns_audio = st_audiorec()

            if speech_input_for_dropdowns_audio is not None:
                dropdown_audio_hash = hashlib.sha256(speech_input_for_dropdowns_audio).hexdigest()
                if dropdown_audio_hash != st.session_state.get('last_dropdown_audio_hash'):
                    st.session_state.last_dropdown_audio_hash = dropdown_audio_hash
                    
                    with st.spinner(f"Transcribing speech for dropdowns in {speech_input_lang_choice}..."):
                        transcribed_text = speech_to_text_vertex(speech_input_for_dropdowns_audio, primary_speech_lang_code=stt_lang_code_for_dropdown)
                        st.info(f"Transcribed: {transcribed_text}")
                        
                        input_language_for_dropdowns = detect_language(transcribed_text)
                        
                        prompt_for_entities = f"""
                        The user said: "{transcribed_text}" (The original language detected was {input_language_for_dropdowns}).
                        Based on this, identify the 'State' and 'Crop' from the following allowed options in English

                        Respond ONLY with a JSON object in the format:
                        {{"state": "Identified State (English)", "crop": "Identified Crop (English)"}}
                        If a value is not found or is ambiguous, use an empty string for that field.
                        Example: {{"state": "Maharashtra", "crop": "Cotton"}}
                        Example: {{"state": "", "crop": "Rice"}}
                        """
                        
                        with st.spinner("Extracting State and Crop..."):
                            try:
                                llm_response = llm_for_entity_extraction.invoke(prompt_for_entities)
                                entity_json_str = llm_response.content.strip().replace("```json", "").replace("```", "")
                                
                                entities = {}
                                try:
                                    entities = json.loads(entity_json_str)
                                except json.JSONDecodeError:
                                    st.warning(f"Could not parse AI response as JSON for entities. Raw response: {entity_json_str}. Attempting regex fallback.")
                                    state_match = re.search(r'"state":\s*"([^"]*)"', entity_json_str, re.IGNORECASE)
                                    if state_match: entities['state'] = state_match.group(1).strip()
                                    crop_match = re.search(r'"crop":\s*"([^"]*)"', entity_json_str, re.IGNORECASE)
                                    if crop_match: entities['crop'] = crop_match.group(1).strip()


                                identified_state_from_input = entities.get("state", "").strip()
                                identified_crop_from_input = entities.get("crop", "").strip()

                                # allowed_states = ["Maharashtra", "West Bengal", "Punjab"]
                                # allowed_crops = ["Cotton", "Potato", "Rice"]

                                # state_found = False
                                # if identified_state_from_input and identified_state_from_input in allowed_states:
                                #     st.session_state.current_state = identified_state_from_input
                                #     st.success(f"State identified and set to: **{identified_state_from_input}**")
                                #     state_found = True
                                # elif identified_state_from_input:
                                #     st.warning(f"Identified state '{identified_state_from_input}' is not in the allowed list.")
                                    
                                # crop_found = False
                                # if identified_crop_from_input and identified_crop_from_input in allowed_crops:
                                #     st.session_state.current_crop = identified_crop_from_input
                                #     st.success(f"Crop identified and set to: **{identified_crop_from_input}**")
                                #     crop_found = True
                                # elif identified_crop_from_input:
                                #     st.warning(f"Identified crop '{identified_crop_from_input}' is not in the allowed list.")
                                
                                st.session_state.current_state = identified_state_from_input
                                state_found = True

                                st.session_state.current_crop = identified_crop_from_input
                                crop_found = True

                                if state_found and crop_found: 
                                    st.session_state.dropdowns_populated = True 
                                    st.session_state.dropdown_input_lang_code = input_language_for_dropdowns 
                                    st.session_state.auto_trigger_pending = True 
                                else: 
                                    st.session_state.dropdowns_populated = True
                                    st.experimental_rerun() 

                            except Exception as e:
                                st.error(f"An error occurred during entity extraction: {e}")
        else: 
            st.info("State and Crop have been set. Adjust manually below if needed.")
            
    elif st.session_state.state_crop_input_mode == "Text":
        state_text = st.text_input("Enter State Name (e.g., Maharashtra, महाराष्ट्र):", key="state_text_input")
        crop_text = st.text_input("Enter Crop Name (e.g., Cotton, कपास):", key="crop_text_input")
        
        process_text_input_button = st.button("Process Text Input", key="process_text_input_button")

        if process_text_input_button and (state_text or crop_text):
            combined_text_for_lang_detect = f"{state_text} {crop_text}"
            input_language_for_dropdowns = detect_language(combined_text_for_lang_detect)

            translated_state_for_llm = translate_text(state_text, input_language_for_dropdowns, 'en')
            translated_crop_for_llm = translate_text(crop_text, input_language_for_dropdowns, 'en')

            prompt_for_entities_text = f"""
            The user provided the following text input (which might be in a non-English language like {input_language_for_dropdowns}):
            State Input: "{state_text}"
            Crop Input: "{crop_text}"

            The translated English inputs are:
            Translated State: "{translated_state_for_llm}"
            Translated Crop: "{translated_crop_for_llm}"

            Based on this, identify the 'State' and 'Crop' from the following allowed English options

            Respond ONLY with a JSON object in the format:
            {{"state": "Identified State (English)", "crop": "Identified Crop (English)"}}
            If a value is not found or is ambiguous, use an empty string for that field.
            Example: {{"state": "Maharashtra", "crop": "Cotton"}}
            Example: {{"state": "", "crop": "Rice"}}
            """
            
            with st.spinner("Extracting State and Crop from text..."):
                try:
                    llm_response = llm_for_entity_extraction.invoke(prompt_for_entities_text)
                    entity_json_str = llm_response.content.strip().replace("```json", "").replace("```", "")
                    
                    entities = {}
                    try:
                        entities = json.loads(entity_json_str)
                    except json.JSONDecodeError:
                        st.warning(f"Could not parse AI response as JSON for entities. Raw response: {entity_json_str}. Attempting regex fallback.")
                        state_match = re.search(r'"state":\s*"([^"]*)"', entity_json_str, re.IGNORECASE)
                        if state_match: entities['state'] = state_match.group(1).strip()
                        crop_match = re.search(r'"crop":\s*"([^"]*)"', entity_json_str, re.IGNORECASE)
                        if crop_match: entities['crop'] = crop_match.group(1).strip()

                    identified_state_from_input = entities.get("state", "").strip()
                    identified_crop_from_input = entities.get("crop", "").strip()

                    # allowed_states = ["Maharashtra", "West Bengal", "Punjab"]
                    # allowed_crops = ["Cotton", "Potato", "Rice"]

                    # state_found = False
                    # if identified_state_from_input and identified_state_from_input in allowed_states:
                    #     st.session_state.current_state = identified_state_from_input
                    #     st.success(f"State identified and set to: **{identified_state_from_input}**")
                    #     state_found = True
                    # elif identified_state_from_input:
                    #     st.warning(f"Identified state '{identified_state_from_input}' is not in the allowed list.")
                        
                    # crop_found = False
                    # if identified_crop_from_input and identified_crop_from_input in allowed_crops:
                    #     st.session_state.current_crop = identified_crop_from_input
                    #     st.success(f"Crop identified and set to: **{identified_crop_from_input}**")
                    #     crop_found = True
                    # elif identified_crop_from_input:
                    #     st.warning(f"Identified crop '{identified_crop_from_input}' is not in the allowed list.")
                    
                    st.session_state.current_state = identified_state_from_input
                    state_found = True

                    st.session_state.current_crop = identified_crop_from_input
                    crop_found = True
                    
                    if state_found and crop_found: 
                        st.session_state.dropdowns_populated = True 
                        st.session_state.dropdown_input_lang_code = input_language_for_dropdowns 
                        st.session_state.auto_trigger_pending = True 
                    else: 
                        st.session_state.dropdowns_populated = True
                        st.experimental_rerun() 

                except Exception as e:
                    st.error(f"An error occurred during entity extraction: {e}")
        else:
            st.info("Enter State and Crop names, then click 'Process Text Input'.")
            
# Display manual dropdowns and manual trigger if dropdowns_populated is True
manual_submit = False
if st.session_state.dropdowns_populated:
    st.markdown("---")
    st.subheader("Or, adjust selections manually:")
    col1, col2 = st.columns(2)

    with col1:
        st.session_state.current_state = st.selectbox(
            "Please select State:",
            ("Haryana", "West Bengal", "Punjab"),
            key="state_select",
            index=("Haryana", "West Bengal", "Punjab").index(st.session_state.current_state)
        )

    with col2:
        st.session_state.current_crop = st.selectbox(
            "Please select Crop:",
            ("Cotton", "Potato", "Rice"),
            key="crop_select",
            index=("Cotton", "Potato", "Rice").index(st.session_state.current_crop)
        )
    
    manual_submit = st.button("Generate KrishiIQ🌱 Insights (Manual)", type="primary", use_container_width=True)


# --- Auto-trigger logic (runs only if flag is set) ---
if st.session_state.auto_trigger_pending:
    st.session_state.auto_trigger_pending = False 
    generate_krishiiq_insights(st.session_state.current_state, 
                               st.session_state.current_crop,
                               st.session_state.dropdown_input_lang_code)


# --- Logic for when manual submit button is clicked ---
if manual_submit: 
    generate_krishiiq_insights(st.session_state.current_state, 
                               st.session_state.current_crop,
                               st.session_state.dropdown_input_lang_code) 


# --- Display Outputs (unconditionally, if available in session state) ---
if st.session_state.chart_data:
    st.plotly_chart(st.session_state.chart_data, use_container_width=True)

if not st.session_state.tail_df_display.empty:
    st.subheader(f"Last 5 entries for {st.session_state.current_crop} in {st.session_state.current_state}:")
    st.dataframe(data=st.session_state.tail_df_display, use_container_width=True)

if st.session_state.forecast_text:
    st.subheader("KrishiIQ🌱 Forecast & Recommendation:")
    st.markdown(st.session_state.forecast_text)
    if st.session_state.get('forecast_audio_bytes'):
        st.audio(st.session_state.forecast_audio_bytes, format='audio/mp3') 

st.markdown("---")
st.subheader("Chat with KrishiIQ🌱")

# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message.type):
        if message.type == "human" and message.additional_kwargs.get("audio_bytes"):
            # Display audio as a playable widget
            st.audio(message.additional_kwargs["audio_bytes"], format='audio/wav')
            # Display transcribed text below the audio
            st.markdown(f"_(Audio Input)_: {message.content}")
        else:
            st.markdown(message.content)
            if message.type == "ai" and message.additional_kwargs.get("audio"):
                st.audio(message.additional_kwargs["audio"], format='audio/mp3')

# --- Chat Input ---
chat_input_container = st.container()
with chat_input_container:
    st.session_state.user_input_mode = st.radio(
        "Choose Chat Input Mode:", ["Text", "Speech"], key="chat_input_mode_radio",
        horizontal=True
    )

    # Use session state to manage user input
    user_chat_input_value = "" # Temporary variable to hold new input from widget

    if st.session_state.user_input_mode == "Text":
        user_chat_input_value = st.chat_input("Ask me about the crop data or general knowledge...", key="chat_text_input")
        if user_chat_input_value:
            st.session_state.user_chat_input_text = user_chat_input_value
            st.session_state.user_chat_input_audio_bytes = None # Clear audio if switching to text
            st.session_state.user_input_lang_code = detect_language(user_chat_input_value)
            st.session_state.process_chat_input = True # Flag to process this input

    elif st.session_state.user_input_mode == "Speech":
        st.write("Click the microphone below to speak your question.")
        chat_speech_input_lang_choice = st.selectbox(
            "Select Language for Chat Speech Input:",
            options=['English', 'हिन्दी', 'বাংলা'],
            index=0, 
            key="chat_speech_input_lang_select_chat"
        )
        stt_lang_code_for_chat = {
            'English': 'en-IN',
            'हिन्दी': 'hi-IN',
            'বাংলা': 'bn-IN'
        }.get(chat_speech_input_lang_choice, 'en-IN')

        chat_audio_output = st_audiorec()
        if chat_audio_output is not None:
            chat_audio_hash = hashlib.sha256(chat_audio_output).hexdigest()
            if chat_audio_hash != st.session_state.get('last_chat_audio_hash_chat'): 
                st.session_state.last_chat_audio_hash_chat = chat_audio_hash
                
                # Store audio bytes and transcribed text in session state
                st.session_state.user_chat_input_audio_bytes = chat_audio_output 
                st.session_state.user_chat_input_text = speech_to_text_vertex(chat_audio_output, primary_speech_lang_code=stt_lang_code_for_chat)
                st.session_state.user_input_lang_code = detect_language(st.session_state.user_chat_input_text)
                
                st.session_state.process_chat_input = True # Flag to process this input
                st.experimental_rerun() # Trigger rerun to process the input


# --- Process Chat Input and Generate Response (using session state variables) ---
# This block will execute when st.session_state.process_chat_input is True
if st.session_state.process_chat_input:
    # Reset the flag immediately to prevent reprocessing on subsequent reruns
    st.session_state.process_chat_input = False 

    # Only proceed if there's actual text input to process
    if st.session_state.user_chat_input_text:
        # Add human message to history based on input mode
        if st.session_state.user_input_mode == "Text":
            st.session_state.messages.append(HumanMessage(content=st.session_state.user_chat_input_text))
        elif st.session_state.user_input_mode == "Speech" and st.session_state.user_chat_input_audio_bytes:
            st.session_state.messages.append(HumanMessage(content=st.session_state.user_chat_input_text, 
                                                           additional_kwargs={"audio_bytes": st.session_state.user_chat_input_audio_bytes}))
        
        # Display the human message instantly
        with st.chat_message("human"):
            if st.session_state.user_input_mode == "Speech":
                 if st.session_state.user_chat_input_audio_bytes:
                     st.audio(st.session_state.user_chat_input_audio_bytes, format='audio/wav')
                 st.markdown(f"_(Audio Input)_: {st.session_state.user_chat_input_text}")
            else:
                st.markdown(st.session_state.user_chat_input_text)


        if st.session_state.main_agent_executor:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        translated_user_input_for_llm = translate_text(st.session_state.user_chat_input_text, st.session_state.user_input_lang_code, "en")
                        
                        response = st.session_state.main_agent_executor.invoke({"input": translated_user_input_for_llm})
                        
                        translated_ai_response = translate_text(response["output"], "en", st.session_state.user_input_lang_code)
                        
                        ai_response_audio = text_to_speech_vertex(translated_ai_response.replace('*', '')[:1000], lang_code=st.session_state.user_input_lang_code)

                        st.markdown(translated_ai_response.replace('*', ''))
                        st.audio(ai_response_audio, format='audio/mp3') 
                        
                        st.session_state.messages.append(AIMessage(content=translated_ai_response, additional_kwargs={"audio": ai_response_audio}))
                    
                    except Exception as e:
                        error_message_en = f"I apologize, but I encountered an error while processing your request: {e}. Could you please rephrase your question?"
                        translated_error_message = translate_text(error_message_en, "en", st.session_state.user_input_lang_code)
                        error_audio = text_to_speech_vertex(translated_error_message.replace('*', '')[:1000], lang_code=st.session_state.user_input_lang_code)
                        
                        st.error(translated_error_message.replace('*', ''))
                        st.audio(error_audio, format='audio/mp3')
                        st.session_state.messages.append(AIMessage(content=translated_error_message.replace('*', ''), additional_kwargs={"audio": error_audio}))
        else:
            with st.chat_message("assistant"):
                no_data_message_en = "Please use the input options above to set a State and Crop, then click 'Generate KrishiIQ🌱 Insights' to load the data and enable the chatbot."
                translated_no_data_message = translate_text(no_data_message_en, "en", st.session_state.user_input_lang_code)
                no_data_audio = text_to_speech_vertex(translated_no_data_message.replace('*', '')[:1000], lang_code=st.session_state.user_input_lang_code)

                st.markdown(translated_no_data_message.replace('*', ''))
                st.audio(no_data_audio, format='audio/mp3')
                st.session_state.messages.append(AIMessage(content=translated_no_data_message.replace('*', ''), additional_kwargs={"audio": no_data_audio}))

    # Clear the temporary stored input after processing
    st.session_state.user_chat_input_text = ""
    st.session_state.user_chat_input_audio_bytes = None


# Optional: Display the raw DataFrame for debugging/overview
if st.checkbox("Show Raw Data", key="show_raw_data_checkbox"):
    if not st.session_state.df_data.empty:
        st.dataframe(st.session_state.df_data)
    else:
        st.info("No data to display.")