# Full Updated Code for KrishiIQ: Multilingual, Agentic, RAG-Powered Crop Disease Bot with Vertex AI Speech

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import re
import requests
from langdetect import detect
from googletrans import Translator

from google.cloud import texttospeech as tts
from google.cloud import speech_v1p1beta1 as speech
import tempfile
import io # Import io for BytesIO

# Import the audio recorder component
from st_audiorec import st_audiorec
# Import pydub
from pydub import AudioSegment

# Streamlit config
st.set_page_config(page_title="KrishiIQ | Crop Disease", page_icon="🌱")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "idyllic-kit-466219-h5-51a1b0253c05.json"

if st.button("Back to Home"):
    st.switch_page("main_app.py")

# Hide sidebar
st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
translator = Translator()

# PDF loading and processing
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

@tool
def search_external_videos_or_links(query: str) -> str:
    """
    Searches for external videos or articles related to crop diseases, treatments, or agricultural practices
    using a web search API. Returns a markdown list of relevant links.
    """
    GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_ID:
        return "Google Custom Search credentials missing."

    params = {"key": GOOGLE_CSE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": 5}
    try:
        resp = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        resp.raise_for_status()
        results = resp.json().get("items", [])
        if not results:
            return "No relevant external links found."

        formatted_links = "\n".join([f"- [{item['title']}]({item['link']})\n  {item.get('snippet', '')}" for item in results])
        return f"Here are some external resources:\n\n{formatted_links}"
    except Exception as e:
        return f"TOOL_ERROR: {e}"

def get_conversational_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    tools = [search_external_videos_or_links]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are KrishiIQ, an agricultural expert chatbot.
        1. Try to answer the question based ONLY on the PDF context and call search_external_videos_or_links.
        2. Present results clearly and recommend safe use.
        3. Never invent treatments.
        """),
        ("user", "Context from PDF: {context}"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def user_input_pipeline(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(query, k=7)
    agent = get_conversational_agent()
    context = "\n\n".join([d.page_content for d in docs])
    return agent.invoke({"input": query, "context": context})

model = genai.GenerativeModel("gemini-2.5-pro")

def get_gemini_response(input_text='', image=None, target_lang_code='en'):
    # Translate user input to English for internal processing by Gemini
    translated_input_for_gemini = translator.translate(input_text, dest="en").text if input_text else "What is wrong with this plant?"

    # The system instruction for Gemini should ask it to respond in the target_lang_code
    system_instruction = f"""
You are an expert agronomist AI assistant. Your task is to analyze plant disease symptoms from an image and describe the issue.
Respond only in this language: {target_lang_code}.
Be clear, avoid English words unless no proper translation exists.
"""

    if image:
        return model.generate_content([
            system_instruction,
            f"User's query in English (for processing): {translated_input_for_gemini}",
            image
        ]).text
    else:
        return model.generate_content([
            system_instruction,
            f"User's query in English (for processing): {translated_input_for_gemini}"
        ]).text


def extract_disease_name(text):
    try:
        summarizer_prompt = (
            "Extract the specific crop disease, pest, or deficiency mentioned in the following diagnosis text."
            " Return only the name of the disease or issue, no explanations."
            f"\n\nDiagnosis:\n{text.strip()}"
        )
        response = model.generate_content(summarizer_prompt)
        return response.text.strip()
    except Exception:
        return ""

def translate_if_needed(text, target_lang):
    # if target_lang == "Original" or target_lang == detect(text): # Don't translate if original is requested or already in target lang
    #     return text
    return translator.translate(text, dest=target_lang).text

def text_to_speech_vertex(text, lang_code="en-US"):
    client = tts.TextToSpeechClient()
    synthesis_input = tts.SynthesisInput(text=text)
    # Map common short codes to BCP-47 for TTS
    if lang_code == 'hi':
        tts_lang_code = 'hi-IN'
    elif lang_code == 'bn':
        tts_lang_code = 'bn-IN'
    elif lang_code == 'en':
        tts_lang_code = 'en-IN' # Indian English for consistency
    else:
        tts_lang_code = 'en-US' # Fallback

    voice = tts.VoiceSelectionParams(language_code=tts_lang_code, ssml_gender=tts.SsmlVoiceGender.NEUTRAL)
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return response.audio_content

def speech_to_text_vertex(audio_bytes, speech_lang_code="en-IN"): # Added speech_lang_code parameter
    client = speech.SpeechClient()

    audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))

    # Convert to mono if needed
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)

    # Resample to 16000 Hz
    if audio_segment.frame_rate != 16000:
        audio_segment = audio_segment.set_frame_rate(16000)

    # Export the processed audio back to bytes
    processed_audio_bytes_io = io.BytesIO()
    audio_segment.export(processed_audio_bytes_io, format="wav")
    processed_audio_bytes = processed_audio_bytes_io.getvalue()

    audio = speech.RecognitionAudio(content=processed_audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code=speech_lang_code # Use the dynamic language code here
        # You could also try alternative_language_codes=["en-IN", "hi-IN", "bn-IN"] here
        # and then check response.results[0].language_code to see what was detected
    )
    response = client.recognize(config=config, audio=audio)
    return " ".join([result.alternatives[0].transcript for result in response.results])

# --- UI ---
st.header("KrishiIQ🌱 Crop Disease Identification & Treatment")

# Initialize session state variables if they don't exist
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'wav_audio_data' not in st.session_state:
    st.session_state.wav_audio_data = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'image_analysis_responses' not in st.session_state:
    st.session_state.image_analysis_responses = []
if 'detected_diseases' not in st.session_state:
    st.session_state.detected_diseases = []
if 'rag_output' not in st.session_state:
    st.session_state.rag_output = None
if 'translated_output' not in st.session_state:
    st.session_state.translated_output = None
if 'tts_audio_bytes' not in st.session_state:
    st.session_state.tts_audio_bytes = None
if 'user_input_language' not in st.session_state: # New state variable for detected input language
    st.session_state.user_input_language = 'en' # Default to English

# Input widgets, update session state on change
st.session_state.input_text = st.text_input("Enter your query (any language):", value=st.session_state.input_text)

st.write("Or record a voice query:")
new_wav_audio_data = st_audiorec()
if new_wav_audio_data is not None:
    st.session_state.wav_audio_data = new_wav_audio_data

new_uploaded_files = st.file_uploader("Upload one or more images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
if new_uploaded_files:
    st.session_state.uploaded_files = new_uploaded_files

# target_lang_selection = st.selectbox("Translate response to:", options=["Original", "en", "hi", "bn"])

target_lang = st.selectbox("Translate response to:", options=["English", "हिन्दी", "বাংলা"])
if target_lang == 'हिन्दी':
    convert_lang = 'hi'
elif target_lang == 'বাংলা':
    convert_lang = 'bn'
elif target_lang == 'English':
    convert_lang = 'en'

submit = st.button("Ask KrishiIQ🌱")


# Logic for processing input and generating response
if submit:
    with st.spinner("Querying KrishiIQ🌱 agent..."):
        st.session_state.detected_diseases = []
        st.session_state.image_analysis_responses = []
        # Reset user_input_language for new submission
        st.session_state.user_input_language = 'en' # Default for now, will be updated if text or audio is processed

        if st.session_state.wav_audio_data is not None:
            with st.spinner("Processing audio..."):
                # Try to infer spoken language from the initial text input if any, or default to broad codes
                # For robust voice language detection, you might need to try common Indian languages
                # or rely on an `alternative_language_codes` approach as mentioned above.
                # For this example, let's assume if there's no text input, we default to hi-IN for audio if target_lang is hi.
                # A more advanced approach would be to prompt the user to select the audio language.

                # Simple heuristic: If target_lang_selection is Hindi, try hi-IN for STT, otherwise default to en-IN
                # stt_lang_code = "hi-IN" if target_lang_selection == "hi" else "en-IN"
                st.session_state.input_text = speech_to_text_vertex(st.session_state.wav_audio_data, speech_lang_code=convert_lang)
                st.write(f"Recognized Speech: {st.session_state.input_text}")
            st.session_state.wav_audio_data = None # Clear audio data

        # After getting text input (either from text box or speech-to-text), detect its language
        if st.session_state.input_text:
            try:
                st.session_state.user_input_language = detect(st.session_state.input_text)
            except:
                st.session_state.user_input_language = 'en' # Fallback if detection fails

        # Now, pass the *detected input language* to get_gemini_response for its internal instruction
        # and use the *selected target language* for the final translation.

        if st.session_state.uploaded_files:
            for uploaded_file in st.session_state.uploaded_files:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, width=200)
                # Gemini should respond in the user's *selected* target language
                res = get_gemini_response(st.session_state.input_text, image, target_lang_code=target_lang)
                st.session_state.image_analysis_responses.append(res)
                disease = extract_disease_name(res)
                if disease:
                    st.session_state.detected_diseases.append(disease)

        if st.session_state.image_analysis_responses:
            st.subheader("Image Analysis:")
            for i, res in enumerate(st.session_state.image_analysis_responses):
                st.markdown(f"**{st.session_state.uploaded_files[i].name}:** {res}")

        rag_query = ""
        if st.session_state.detected_diseases:
            disease_list = ", ".join(st.session_state.detected_diseases)
            rag_query = f"What is the treatment for {disease_list}? Provide external videos or links."
        elif st.session_state.input_text:
            # If the input text is in Hindi, the query should also be intelligent about that.
            # However, for the RAG part, it's generally best to query in English if your PDF is in English.
            # The translation of the *final answer* handles the output language.
            rag_query = st.session_state.input_text + " How can this be treated? Provide external videos or links."
        else:
            rag_query = "How to treat common crop diseases? Provide examples."

        st.write("---")
        pdf_text = get_pdf_text('PDF/crop_disease/crop_diseases_treatment.pdf')
        chunks = get_text_chunks(pdf_text)
        get_vector_store(chunks)
        with st.spinner("Querying KrishiIQ🌱 agent..."):
            result = user_input_pipeline(rag_query)
            # The `translate_if_needed` function uses `target_lang_selection`
            st.session_state.translated_output = translate_if_needed(result["output"], convert_lang)
            st.session_state.uploaded_files = [] # Clear uploaded files

        # Display results if available in session state
        if st.session_state.translated_output:
            st.write("### Final Answer:")
            st.write(st.session_state.translated_output.replace('*', ''))
            st.session_state.tts_audio_bytes = text_to_speech_vertex(st.session_state.translated_output.replace('*', '')[:1000], lang_code=convert_lang)

            # if st.button("🔊 Hear the Response", key="tts_button"):
            #     with st.spinner("Generating audio..."):
            #         # Use the selected target language for TTS
            #         st.session_state.tts_audio_bytes = text_to_speech_vertex(st.session_state.translated_output[:4999], lang_code=convert_lang)

        # Render the audio player if tts_audio_bytes exists in session state
        if st.session_state.tts_audio_bytes:
            st.audio(st.session_state.tts_audio_bytes, format='audio/mp3')

        # elif not st.session_state.input_text or st.session_state.wav_audio_data is None or not st.session_state.uploaded_files:
        #     st.warning("Please provide either text, voice, or image input to ask KrishiIQ🌱")