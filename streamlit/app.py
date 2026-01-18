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

