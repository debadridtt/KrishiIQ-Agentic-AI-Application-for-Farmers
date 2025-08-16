# KrishiIQ-Agentic-AI-Application-for-Farmers
Agentic AI Application for Farmers

- Open a Google Cloud account and enable all the APIs, like VertexAI, Texttospeech, Speechtotext, etc. and then download the JSON for the service account
- Please ignore above step if you already have the JSON of the service account
- Once you have the JSON, place it inside the same directory as "main_app.py" and then follow the below instructions

Make sure you have a laptop and a stable internet connection, then start with the following steps:
- Install anaconda with Python from https://www.anaconda.com/download/success
- Now copy the Krishi IQ ZIP file to a proper location and extract it
- Once folder is extracted, go into the "KrishiIQ folder" and you will see another "KrishiIQ folder"
- Now open command prompt from this location and type "cd KrishiIQ"

Then type the following commands one by one:
- conda create --name capitalone python=3.12.4
- conda activate capitalone
- pip install -r requirements.txt
- streamlit run main_app.py
