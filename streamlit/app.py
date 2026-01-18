import streamlit as st
import tempfile
import json
import pandas as pd
import numpy as np

import xgboost as xgb
from faster_whisper import WhisperModel


from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser

import os
import sys

# <<------------------------------->>

# Configuration

HOME_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HOME_DIR, "..", "models")
DATA_DIR = os.path.join(HOME_DIR, "..", "data")

# <<------------------------------->>

# Loads Model and Metadata

@st.cache_resource
def load_assets():
    """Loads XGBoost model and all encoder mappings in Json """
    try:
        model = xgb.XGBClassifier()
        model_path = os.path.join(MODELS_DIR, "xgb_severity_model.json")
        model.load_model(model_path)

        features_path = os.path.join(MODELS_DIR, "model_features.json")
        with open(features_path) as f:
            feature_order = json.load(f)
        
        encoders_path = os.path.join(MODELS_DIR, "label_encoders_mapping.json")
        with open(encoders_path) as f:
            encoders = json.load(f)

        

        # Create reverse mappings for all encoders
        prod_ai_reverse = {}

        for k, v in encoders["prod_ai"].items():
            prod_ai_reverse[v] = int(k)

        indi_pt_reverse = {}

        for k, v in encoders["indi_pt"].items():
            indi_pt_reverse[v] = int(k)

        pt_reverse = {}

        for k, v in encoders["pt"].items():
            pt_reverse[v] = int(k)

        role_cod_reverse = {}

        for k, v in encoders["role_cod"].items():
            role_cod_reverse[v] = int(k)


        return model, encoders, feature_order, prod_ai_reverse, indi_pt_reverse, pt_reverse, role_cod_reverse
    
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.error(f"Models directory: {MODELS_DIR}")
        st.error(f"Files in directory: {os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'Directory not found'}")
        sys.exit(1)


model, encoders, FEATURE_ORDER, PROD_AI_MAP, INDI_PT_MAP, PT_MAP, ROLE_COD_MAP = load_assets()

# <<------------------------------->>

# Loads Whisper

@st.cache_resource
def load_whisper():
    """Load Whisper model for audio transcription"""
    try:
        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as e:
        st.error(f"❌ Error loading Whisper model: {str(e)}")
        sys.exit(1)

whisper_model = load_whisper()

# <<------------------------------->>

# Gemini LLM

from dotenv import load_dotenv

load_dotenv()

# Verify API key is loaded
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in environment variables!")
    st.info("Please set GEMINI_API_KEY in your .env file")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

# Fixed: Escaped curly braces in JSON example with double braces
prompt = PromptTemplate(
    template="""
You are a medical information extractor.
You'll be getting a transcribed text from a audio file and your task is to get the drugs which are present in the text 
how many drugs are there and from the conversation you have to find the indication point and the reactions,
These drugs indication point and reaction are the columns of my data on which my model is trained so I have to make inference 
Based on these points which you give to me.

I am giving you the list of drugs which I am using and the reaction and the 
indication point on which my model is trained you have to infer from the text file and you have to find out if they exist in the 
given list or not I am giving you the list

Valid Medication List:
[
"ACETAMINOPHEN","ADALIMUMAB","CARBIDOPA\\LEVODOPA","DEXAMETHASONE","EFAVIRENZ\\EMTRICITABINE\\TENOFOVIR DISOPROXIL FUMARATE",
"EMTRICITABINE\\TENOFOVIR DISOPROXIL FUMARATE","ESOMEPRAZOLE MAGNESIUM","FUROSEMIDE","IBRUTINIB","INFLIXIMAB-DYYB","LANSOPRAZOLE",
"LENALIDOMIDE","MACITENTAN","NIVOLUMAB","OCTREOTIDE ACETATE","OMALIZUMAB","OMEPRAZOLE MAGNESIUM","PREDNISOLONE","PREDNISONE","RANITIDINE",
"RANITIDINE HYDROCHLORIDE","RIBOCICLIB","RITUXIMAB","RIVAROXABAN","RUXOLITINIB","SECUKINUMAB","TENOFOVIR DISOPROXIL FUMARATE","TOCILIZUMAB",
"TOFACITINIB CITRATE","VEDOLIZUMAB"
]

Valid Indication Terms:
[
"Abdominal discomfort","Acromegaly","Ankylosing spondylitis","Asthma","Atrial fibrillation","Breast cancer","Breast cancer metastatic",
"Carcinoid tumour","Chronic lymphocytic leukaemia","Chronic spontaneous urticaria","Colitis ulcerative","Crohn's disease",
"Diffuse large B-cell lymphoma","Dyspepsia","Gastric ulcer","Gastrooesophageal reflux disease","HIV infection","Malignant melanoma",
"Myelofibrosis","Neuroendocrine tumour","Pain","Parkinson's disease","Plasma cell myeloma","Polycythaemia vera","Premedication",
"Prophylaxis","Psoriasis","Psoriatic arthropathy","Pulmonary arterial hypertension","Rheumatoid arthritis"
]

Valid Reaction Terms:
[
"Acute kidney injury","Anxiety","Arthralgia","Bladder cancer","Bone density decreased","Bone loss","Breast cancer","Chronic kidney disease",
"Colorectal cancer","Diarrhoea","Dyspnoea","End stage renal disease","Fatigue","Gastric cancer","Hepatic cancer","Lung neoplasm malignant",
"Multiple fractures","Nausea","Neoplasm malignant","Oesophageal carcinoma","Osteonecrosis","Osteoporosis","Pain","Pancreatic carcinoma",
"Pneumonia","Prostate cancer","Renal cancer","Renal failure","Renal injury","Skeletal injury"
]

Return the result strictly in the following JSON format:

{{
  "drugs": [],
  "indications": [],
  "reactions": []
}}

Transcript:
{conversation}

Valid Drugs:
{drug_list}

Valid Indications:
{indi_list}

Valid Reactions:
{pt_list}
""",
    input_variables=["conversation", "drug_list", "indi_list", "pt_list"]
)

parser = JsonOutputParser()

# <<------------------------------->>
