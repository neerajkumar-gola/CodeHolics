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

# Entity Extraction from the conversation

def extract_entities(text):
    """Extracting medical entities from text using Gemini LLM"""
    try:
        chain = prompt | llm | parser
        entities = chain.invoke({
            "conversation": text,
            "drug_list": list(PROD_AI_MAP.keys()),
            "indi_list": list(INDI_PT_MAP.keys()),
            "pt_list": list(PT_MAP.keys())
        })
        return entities
    except Exception as e:
        st.error(f"❌ Error extracting entities: {str(e)}")
        return {"drugs": [], "indications": [], "reactions": []}

# <<------------------------------->>

# Building Features using feature engineering

def build_features(entities):
    """Building feature matrix from extracted entities"""
    rows = []
    drug_stats = encoders["drug_stats"]

    for drug in entities["drugs"]:
        if drug not in PROD_AI_MAP:
            continue

        drug_id = str(PROD_AI_MAP[drug])
        stats = drug_stats.get(drug_id, {})
        
        # Get first indication if available, else default to 0
        indi_encoded = 0
        if entities.get("indications") and len(entities["indications"]) > 0:
            first_indication = entities["indications"][0]
            if first_indication in INDI_PT_MAP:
                indi_encoded = INDI_PT_MAP[first_indication]
        
        # role_cod: PS=2 (Primary Suspect) is the default
        role_cod_encoded = 2  # 'PS' = Primary Suspect

        rows.append({
            "prod_ai_encoded": int(drug_id),
            "indi_pt_encoded": indi_encoded,
            "role_cod_encoded": role_cod_encoded,
            "drug_frequency": stats.get("drug_frequency", 0),
            "drug_avg_severity": stats.get("drug_avg_severity", 0.0),
            "num_drugs": len(entities["drugs"]),
            "num_reactions": len(entities.get("reactions", [])),
            "treatment_duration": 0,  # Default - could be extracted from text
            "dechal_binary": 0,
            "is_primary_suspect": 1,
            "is_secondary_suspect": 0,
            "is_concomitant": 0
        })

    return rows

# <<------------------------------->>

# Streamlit UI

st.set_page_config(page_title="Medical Audio Risk Detecter", layout="wide")
# Header
st.title("🩺 Medical Audio Risk Detecter")
st.markdown("""
### 🧠 How It Works  

Upload a **medical conversation audio file** and let our AI-powered analyzer do the rest!  

1️⃣ **🎧 Transcribe the audio** — Converts speech to accurate text using **OpenAI Whisper**.  
2️⃣ **🧬 Extract medical entities** — Identifies **Drugs**, **Indications**, and **Reactions** with **Gemini AI**.  
3️⃣ **📈 Predict adverse event severity** — Evaluates potential drug event seriousness via **XGBoost**.  
4️⃣ **🚨 Classify overall risk tier** — Determines if intervention is needed: **Safe**, **Monitor**, or **Critical**.  

---
💡 *Built for pharmacovigilance research — combining NLP, machine learning, and clinical insights.*
""")


# <<------------------------------->>

# Main Content

audio_file = st.file_uploader("Select an audio file", type=["mp3", "wav", "m4a"])

if audio_file:
    st.audio(audio_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    try:
        # STEP 1: Audio Transcription
        with st.spinner("🎧 Transcribing audio..."):
            segments, info = whisper_model.transcribe(audio_path)
            transcript = " ".join([seg.text for seg in segments])

        st.subheader("📝 Transcription")
        with st.expander("View full transcript", expanded=True):
            st.write(transcript)

        # STEP 2: Entity Extraction
        with st.spinner("🔍 Extracting medical entities..."):
            entities = extract_entities(transcript)

        st.subheader("💡 Extracted Entities")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Drugs Found", len(entities.get("drugs", [])))
            for drug in entities.get("drugs", []):
                st.success(f"💊 {drug}")
        with col2:
            st.metric("Indications Found", len(entities.get("indications", [])))
            for indi in entities.get("indications", []):
                st.info(f"🏥 {indi}")
        with col3:
            st.metric("Reactions Found", len(entities.get("reactions", [])))
            for react in entities.get("reactions", []):
                st.warning(f"⚠️ {react}")

                # STEP 3: Feature Engineering
        with st.spinner("⚙️ Building model features..."):
            feature_rows = build_features(entities)

        if not feature_rows:
            st.error("❌ No valid drugs detected in recognized list.")
            st.stop()

        X = pd.DataFrame(feature_rows)[FEATURE_ORDER]
        with st.expander("📋 View Feature Matrix"):
            st.dataframe(X, use_container_width=True)
            
            
        # STEP 4: Model Prediction
        with st.spinner("🤖 Predicting severity..."):
            preds = model.predict(X)
            proba = model.predict_proba(X)

        st.success("✅ Prediction Complete!")
        st.subheader("📊 Severity Predictions & Risk Assessment")

        severity_map = {0: "Low", 1: "Medium", 2: "High"}
        severity_colors = {0: "🟢", 1: "🟡", 2: "🔴"}
        severity_weights = {0: 0.2, 1: 0.5, 2: 0.9} 

        # STEP 5: Risk Tier Calculation
        for idx, (drug, pred, prob) in enumerate(zip(entities["drugs"], preds, proba)):
            severity = severity_map.get(pred, "Unknown")
            severity_icon = severity_colors.get(pred, "⚪")
            severity_score = severity_weights.get(pred, 0.5)
            confidence = prob[pred] * 100

            avg_severity = severity_score
            num_reactions = len(entities.get("reactions", []))
            drug_frequency = np.random.randint(1, 3)
            is_primary_suspect = 1 if idx == 0 else 0

            reaction_factor = min(num_reactions / 3, 1.0)
            drug_freq_factor = min(drug_frequency / 3, 1.0)

            risk_score = (
                avg_severity * 0.6 +
                reaction_factor * 0.25 +
                is_primary_suspect * 0.1 +
                drug_freq_factor * 0.05
            )
            risk_score = min(max(risk_score, 0), 1)
            risk_percentage = int(risk_score * 100)

            if risk_score < 0.4:
                tier = "🟢 Safe"
                message = "Low likelihood of adverse event – continue regular monitoring."
            elif risk_score < 0.7:
                tier = "🟡 Monitor Closely"
                message = "Moderate risk detected – monitor patient closely for new symptoms."
            else:
                tier = "🔴 Immediate Intervention Required"
                message = "High likelihood of adverse event – intervene immediately."

            st.divider()
            st.markdown(f"### {severity_icon} {drug}")
            st.metric("Predicted Severity", severity)
            st.metric("Confidence", f"{confidence:.1f}%")
            st.markdown(f"**Risk Tier:** {tier}")
            st.progress(risk_score)
            st.caption(f"Computed Risk Score: **{risk_score:.2f}** — {message}")

            prob_df = pd.DataFrame({
                'Severity': ['Low', 'Medium', 'High'],
                'Probability': [prob[0], prob[1], prob[2]]
            })
            st.bar_chart(prob_df.set_index('Severity'), height=200)

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        import traceback
        with st.expander("View error details"):
            st.code(traceback.format_exc())

    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

# <<------------------------------->>   

# Footer

st.divider()
st.caption("⚠️ Disclaimer: This tool is intended solely for research and educational purposes.It should not be used as a substitute for professional medical advice, diagnosis, or treatment.")