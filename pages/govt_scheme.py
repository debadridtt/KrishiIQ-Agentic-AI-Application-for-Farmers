from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langdetect import detect
from deep_translator import GoogleTranslator
import requests
import unicodedata
from google.cloud import texttospeech as tts
from google.cloud import speech_v1p1beta1 as speech
import io
from pydub import AudioSegment
from st_audiorec import st_audiorec
import hashlib
from langchain_core.messages import HumanMessage, AIMessage # Essential for handling Langchain chat history messages


# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit page configuration
st.set_page_config(
    page_title="KrishiIQ | Government Scheme Chatbot",
    page_icon="🌱",
    layout="centered"
)

# Function to hide Streamlit sidebar
def hide_pages():
    st.markdown("""
        <style>
            section[data-testid="stSidebar"][aria-expanded="true"]{
                display: none;
            }
        </style>
        """, unsafe_allow_html=True)

hide_pages()

st.header("KrishiIQ🌱 Government Scheme Chatbot")
st.markdown("This chatbot can answer questions about different government schemes and how to apply for them!")

# Initialize session state variables if they don't exist
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = False
if 'user_input_language' not in st.session_state:
    st.session_state.user_input_language = 'en' # This might not be strictly needed, but kept for consistency
if 'tts_audio_bytes' not in st.session_state:
    st.session_state.tts_audio_bytes = None
if 'current_text_input' not in st.session_state: # Tracks the last text input submitted
    st.session_state.current_text_input = ""
if 'recorded_audio_bytes' not in st.session_state: # Stores raw audio bytes for STT processing
    st.session_state.recorded_audio_bytes = None
if 'st_audiorec_output' not in st.session_state: # Stores the direct output from st_audiorec component
    st.session_state.st_audiorec_output = None
if 'last_processed_audio_hash' not in st.session_state: # Helps prevent re-processing same audio
    st.session_state.last_processed_audio_hash = None
if 'actual_user_input_lang' not in st.session_state: # Stores the detected language of user's current input
    st.session_state.actual_user_input_lang = 'en' # Default language


DEFAULT_PDF_PATH = "PDF/govt_scheme/scheme_details.pdf"

def get_pdf_text(pdf_path):
    """Reads text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
    except FileNotFoundError:
        st.error(f"Error: Default PDF file not found at '{pdf_path}'")
        return ""
    except Exception as e:
        st.error(f"Error reading PDF file '{pdf_path}': {e}")
        return ""

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_govt_scheme_index")
        st.session_state.vector_store_loaded = True
        st.success("Chatbot Ready to be used!")
    except Exception as e:
        st.error(f"Vector store error: {e}")
        st.session_state.vector_store_loaded = False

def get_conversational_chain():
    """Initializes and returns a Langchain conversational QA chain."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt = PromptTemplate(
        template="""
        You are a helpful chatbot answering questions about a government scheme.
        Use the context below. If not found, say so politely.

        Chat History:
        {chat_history}
        Context:
        {context}
        Question:
        {question}
        Answer:
        """,
        input_variables=["chat_history", "context", "question"]
    )
    return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt, memory=memory, verbose=True)

@tool
def search_govt_links(query: str) -> str:
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

def detect_language(text):
    """Detects the language of a given text."""
    if not text:
        return 'en'
    try:
        return detect(text)
    except:
        return 'en'

def translate_text(text, source_lang, target_lang):
    """Translates text from source to target language using GoogleTranslator."""
    if not text:
        return ""
    if source_lang == target_lang:
        return text
    
    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return unicodedata.normalize("NFKC", translated)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def text_to_speech_vertex(text, lang_code="en-US"):
    """Converts text to speech using Google Cloud Text-to-Speech API."""
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
        tts_lang_code = 'en-US' # Fallback for unknown/original languages

    voice = tts.VoiceSelectionParams(language_code=tts_lang_code, ssml_gender=tts.SsmlVoiceGender.NEUTRAL)
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return response.audio_content

def speech_to_text_vertex(audio_bytes, speech_lang_code="en-IN"):
    """Converts speech audio bytes to text using Google Cloud Speech-to-Text API."""
    client = speech.SpeechClient()

    # Audio Processing with pydub: ensures correct format for Vertex AI
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
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # LINEAR16 is common for WAV
        sample_rate_hertz=16000,
        language_code=speech_lang_code,
        enable_automatic_punctuation=True
    )
    
    with st.spinner(f"Transcribing your speech... (Expecting {speech_lang_code})"):
        response = client.recognize(config=config, audio=audio)
    
    if response.results:
        return response.results[0].alternatives[0].transcript
    return "Sorry, could not understand speech."

def summarize_text_for_tts(long_text, target_lang):
    """Summarizes text for TTS to fit within byte limits."""
    if len(long_text.encode('utf-8')) <= 4500: # Leave some buffer
        return long_text

    summarizer_prompt = f"""
    You are a helpful assistant. Summarize the following text concisely, focusing on the key information about the government scheme.
    Keep the summary as short as possible, ideally under 4000 characters, while retaining essential details.
    Respond ONLY in {target_lang}.

    Text to summarize:
    {long_text}
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(summarizer_prompt)
        summarized_text = response.text.strip()
        
        # Fallback truncation if summarization still results in too long text
        if len(summarized_text.encode('utf-8')) > 4500:
            return summarized_text.encode('utf-8')[:4500].decode('utf-8', errors='ignore')
        return summarized_text
    except Exception as e:
        st.warning(f"Could not summarize for TTS: {e}. Attempting to truncate.")
        return long_text.encode('utf-8')[:4500].decode('utf-8', errors='ignore')


def get_assistant_response_content(user_question, user_lang_code):
    """Generates assistant's response using Langchain agent and RAG."""
    # Always translate user input to English for RAG and agent processing,
    # as the PDF content and tool search are likely in English.
    translated_question_for_rag = translate_text(user_question, user_lang_code, "en")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")
        vector_store = FAISS.load_local("faiss_govt_scheme_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Vector store load error: {e}")
        return "Document knowledge base error."

    docs = vector_store.similarity_search(translated_question_for_rag, k=5)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    tools = [search_govt_links]
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful bot answering questions about a government scheme.
        Use the context provided from the documents for information.
        If the context does not contain sufficient information, use the `search_govt_links` tool to find official government links.

        **Instructions for using search_govt_links tool:**
        -   If you use the tool, always present the found links *clearly formatted as clickable Markdown links*.
        -   Explain what the links are for (e.g., "Here are some official links to apply for...", or "You can find more details here:").
        -   **NEVER repeat links or link descriptions if they have already been provided in the current turn.**
        -   If the tool returns "No relevant external links found.", state that no direct links were found.
        -   Integrate the links smoothly into your answer.

        Ensure your final answer is clear, comprehensive, and directly addresses the user's question.
        """),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # Convert Langchain chat history messages to a simple string for agent_input's chat history
    # Only send text content to the LLM
    history_for_llm = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            history_for_llm.append(HumanMessage(content=msg["content"])) # Always use the text content for LLM
        else: # assistant message
            history_for_llm.append(AIMessage(content=msg["content"]))

    # Pass the history to the agent_executor's memory directly if possible, or construct for input
    # If using ConversationBufferMemory, it already manages this based on `return_messages=True`
    # and the prompt template's `chat_history` variable.
    # The `history_str` below is for the agent's prompt input, not its internal memory.
    history_str = "\n".join([f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}" for m in st.session_state.conversation_chain.memory.buffer])
    context_str = "\n\n".join([doc.page_content for doc in docs])

    agent_input = f"""
    Chat History:
    {history_str}

    Context from document:
    {context_str}

    User Question: {translated_question_for_rag}
    """

    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=st.session_state.conversation_chain.memory)
    
    response = agent_executor.invoke({"input": agent_input})
    
    # Translate the final output of the agent back to the user's input language
    final_response = translate_text(response["output"], "en", user_lang_code)
    return final_response


# --- Streamlit UI ---

# Load vector store if not already loaded (happens only once)
if not st.session_state.vector_store_loaded:
    with st.spinner("Setting up chatbot..."):
        raw_text = get_pdf_text(DEFAULT_PDF_PATH)
        if raw_text:
            chunks = get_text_chunks(raw_text)
            get_vector_store(chunks)
            st.session_state.conversation_chain = get_conversational_chain()

# Placeholder for chat messages (the main chat history display area)
# All messages will be drawn inside this container
chat_messages_container = st.container()

# Display chat history within the chat_messages_container
# This loop is the ONLY place where messages are displayed to avoid duplication.
with chat_messages_container:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                if msg.get("type") == "audio":
                    st.write(f"_(Audio Input)_: {msg['content']}") # Show transcription
                    if msg.get("audio_bytes"):
                        st.audio(msg["audio_bytes"], format='audio/wav') # Play original audio
                else: # Default to text
                    st.write(msg["content"])
            elif msg["role"] == "assistant":
                st.write(msg["content"])
                if "audio" in msg and msg["audio"] is not None:
                    st.audio(msg["audio"], format='audio/mp3')

# Place input controls at the very bottom of the page
# This ensures they are always visible and below the chat history
input_controls_container = st.container()

with input_controls_container:
    input_mode = st.radio("Choose input mode:", ["Text", "Speech"], key="input_mode_radio")

    target_tts_lang_code = st.selectbox(
        "Select Assistant Voice Language:",
        options=["en", "hi", "bn"],
        format_func=lambda x: {"en": "English", "hi": "हिन्दी", "bn": "বাংলা"}.get(x, x),
        key="tts_lang_selector"
    )

    user_question_text = "" # Initialize for the current turn, will be populated if new input
    user_audio_data = None # To store audio bytes for history

    if input_mode == "Text":
        user_input_from_text_box = st.chat_input("Ask about a government scheme...", key="text_input_box")

        if user_input_from_text_box and user_input_from_text_box != st.session_state.current_text_input:
            user_question_text = user_input_from_text_box
            st.session_state.current_text_input = user_input_from_text_box # Update last processed text input
            st.session_state.actual_user_input_lang = detect_language(user_question_text) # Detect language for text input
            
            # Clear any leftover audio states if switching to text input
            st.session_state.recorded_audio_bytes = None
            st.session_state.st_audiorec_output = None
            st.session_state.last_processed_audio_hash = None


    elif input_mode == "Speech":
        st.write("Click the microphone button below and speak clearly. Your transcription will appear here.")
        
        st.session_state.st_audiorec_output = st_audiorec() # No 'key' argument needed/supported here

        if st.session_state.st_audiorec_output is not None:
            current_audio_hash = hashlib.sha256(st.session_state.st_audiorec_output).hexdigest()
            if current_audio_hash != st.session_state.last_processed_audio_hash:
                st.session_state.recorded_audio_bytes = st.session_state.st_audiorec_output
                user_audio_data = st.session_state.recorded_audio_bytes # Store for history

                stt_lang_code_map = {"en": "en-IN", "hi": "hi-IN", "bn": "bn-IN"}
                inferred_stt_lang = stt_lang_code_map.get(target_tts_lang_code, "en-IN")

                user_question_text = speech_to_text_vertex(st.session_state.recorded_audio_bytes, speech_lang_code=inferred_stt_lang)
                st.success(f"You said: {user_question_text}")
                
                st.session_state.actual_user_input_lang = detect_language(user_question_text) # Detect language for speech input
                
                st.session_state.last_processed_audio_hash = current_audio_hash # Store the hash of the processed audio
                # Reset st_audiorec_output to allow a new recording to be triggered on next rerun
                st.session_state.st_audiorec_output = None
                st.session_state.recorded_audio_bytes = None # Clear this as well

# --- Main Logic for Processing Questions and Generating Responses ---
# This block executes if there's any user input (text or transcribed speech)
# and the vector store is ready.
if user_question_text and st.session_state.vector_store_loaded:
    # Append user's original question to chat history based on input type
    if input_mode == "Text":
        st.session_state.chat_history.append({"role": "user", "type": "text", "content": user_question_text})
    elif input_mode == "Speech" and user_audio_data is not None:
        st.session_state.chat_history.append({
            "role": "user",
            "type": "audio",
            "content": user_question_text, # The transcribed text
            "audio_bytes": user_audio_data # The original audio bytes
        })

    # Generate assistant's response
    with st.spinner("Thinking..."):
        assistant_response_text = get_assistant_response_content(user_question_text, st.session_state.actual_user_input_lang)

        summarized_response_for_tts = summarize_text_for_tts(assistant_response_text, target_tts_lang_code)
        st.session_state.tts_audio_bytes = text_to_speech_vertex(summarized_response_for_tts.replace('*', '')[:1000], lang_code=target_tts_lang_code)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_response_text.replace('*', ''),
        "audio": st.session_state.tts_audio_bytes
    })
    
    # Crucial: Rerun the app to immediately display the newly added messages
    st.experimental_rerun()

# Information message if chatbot is not yet ready (e.g., PDF loading)
if not st.session_state.vector_store_loaded:
    st.info("The chatbot is still setting up. Please wait a moment.")