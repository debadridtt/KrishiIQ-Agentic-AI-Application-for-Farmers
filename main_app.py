import streamlit as st
import re
from googletrans import Translator

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="KrishiIQ",
    page_icon="🌱",
)

# --- Hide Streamlit multipage sidebar navigation but keep sidebar ---
def hide_sidebar_nav():
    st.markdown(
        """
        <style>
        /* Hide the Streamlit sidebar page navigation (multipage menu) */
        section[data-testid="stSidebarNav"] {display: none !important;}
        /* Additional selector for Streamlit’s nav list, for robustness */
        [data-testid="stSidebar"] ul {display: none !important;}
        /* Optionally hide Streamlit's main header */
        .block-container > header {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

hide_sidebar_nav()

# --- Language options ---
LANGUAGE_OPTIONS = {
    "English": "en",
    "हिन्दी": "hi",
    "বাংলা": "bn",
}

# --- Sidebar language dropdown ---
with st.sidebar:
    st.markdown("### 🌐 Select Language / भाषा चुनें")
    selected_language = st.selectbox("Choose your language:", list(LANGUAGE_OPTIONS.keys()), index=0)

target_lang = LANGUAGE_OPTIONS[selected_language]

# --- Google Translator instance ---
translator = Translator()

# --- Improved translate_text function ---
def translate_text(text, dest_lang='en'):
    """
    Translate all text including markdown plain text and link labels,
    but keep URLs in markdown links unchanged.
    """
    if not text or dest_lang == "en":
        return text

    try:
        # Regex pattern to find markdown links: [label](url)
        pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

        parts = []
        last_end = 0

        # Find all markdown links in text
        for match in pattern.finditer(text):
            start, end = match.span()
            label = match.group(1)
            url = match.group(2)

            # Translate preceding plain text
            plain_text = text[last_end:start]
            if plain_text.strip():
                translated_plain = translator.translate(plain_text, dest=dest_lang).text
            else:
                translated_plain = plain_text

            # Translate link label
            translated_label = translator.translate(label, dest=dest_lang).text

            # Append translated plain text + translated link
            parts.append(translated_plain)
            parts.append(f'[{translated_label}]({url})')

            last_end = end

        # Translate any remaining plain text after last link
        tail_text = text[last_end:]
        if tail_text.strip():
            translated_tail = translator.translate(tail_text, dest=dest_lang).text
        else:
            translated_tail = tail_text
        parts.append(translated_tail)

        # Reconstruct full translated text
        return ''.join(parts)

    except Exception:
        # Fallback: return original text if translation fails
        return text

# --- UI text in English ---
welcome_header = "# Welcome to KrishiIQ🌱"

app_intro = """
**KrishiIQ** is an Agentic AI application specifically designed to help farmers take the right decisions for their farming.

👇 **Select the following options below** to explore the app!
"""

crop_disease_title = "What is the functionality of Crop Disease and Identification?"

crop_disease_bullets = """
- Check out [Crop Disease Identification & Treatment](/crop_disease)
- Ask our GenAI agent to understand crop health, any issues with it and its treatment if any, etc.
- Finally use the output to take correct decisions related to your crop
"""

market_analysis_title = "What is the functionality of Real-time Market Analysis of Crops?"

market_analysis_bullets = """
- Check out [Market Analytics of Crop](/crop_insights)
- Ask our GenAI agent to understand current market rate of crops, any supply-demand issues, etc.
- Finally use the output to take correct decisions regarding selling, etc.
"""

scheme_title = "Want to know about different Govt. schemes and apply for them?"

scheme_bullets = """
- Check out [Govt. Schemes Q&A Chatbot](/govt_scheme)
- Ask our GenAI agent to know about latest govt. schemes, loan offers for farmers, etc.
- Finally get links to apply as well
"""

# --- Display UI with translation ---
st.write(translate_text(welcome_header, target_lang))

col1, col2, col3 = st.columns(3)

with col1:
    st.image("images/image_1.jpg", width=600)

with col2:
    st.write(" ")

with col3:
    st.write(" ")

st.markdown(f"#### {translate_text(app_intro, target_lang)}")

st.markdown(f"#### {translate_text(crop_disease_title, target_lang)}")
st.markdown(translate_text(crop_disease_bullets, target_lang))

st.markdown(f"#### {translate_text(market_analysis_title, target_lang)}")
st.markdown(translate_text(market_analysis_bullets, target_lang))

st.markdown(f"#### {translate_text(scheme_title, target_lang)}")
st.markdown(translate_text(scheme_bullets, target_lang))
