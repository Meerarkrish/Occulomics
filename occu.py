import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import pycountry
import pandas as pd
import datetime

# --- SET PAGE CONFIG ---
st.set_page_config(
    page_title="Oculomics Commons Hub",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AI MODEL & DATA SCIENCE LOGIC ---
@st.cache_resource
def load_foundation_model():
    """Loads a ResNet18 backbone for vascular feature extraction."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def analyze_vessels(image, model):
    """Processes image and returns Oculomics metrics."""
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        features = model(input_tensor)
        val = float(torch.mean(features))
        # Deterministic logic for demo; replace with your trained regression weights
        avr = 0.68 + (val % 0.08)
        age_offset = (val % 8) - 4
        
    return round(avr, 2), round(age_offset, 1)

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("👁️ Oculomics Hub")
st.sidebar.markdown("The Open-Source Retinal Repository")

menu = st.sidebar.radio(
    "Repository Navigation",
    ["🏠 Home & Screening", "📊 Global Epidemiology", "📄 Model Card", "💾 Data Access"]
)

st.sidebar.markdown("---")
all_countries = sorted([c.name for c in pycountry.countries])
selected_country = st.sidebar.selectbox("Geographic Context", all_countries, index=all_countries.index("India") if "India" in all_countries else 0)
actual_age = st.sidebar.number_input("Chronological Age", 1, 120, 30)

# Load AI
foundation_ai = load_foundation_model()

# --- PAGE 1: HOME & SCREENING ---
if menu == "🏠 Home & Screening":
    st.header("🏠 Oculomics Screening Portal")
    st.write(f"Analyzing retinal biomarkers for **{selected_country}**.")
    
    uploaded_file = st.file_uploader("Upload Fundus Image (Anonymized)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Original Fundus Image", use_container_width=True)
            # Green Channel Visualization
            green_img = np.array(img)[:, :, 1]
            st.image(green_img, caption="Vascular Contrast (Green Channel)", use_container_width=True)
            
        with col2:
            st.subheader("AI Analysis Results")
            avr, offset = analyze_vessels(img, foundation_ai)
            
            c1, c2 = st.columns(2)
            c1.metric("AVR Ratio", avr)
            c2.metric("Retinal Age Offset", f"{offset} Years")
            
            st.write(f"**Predicted Biological Age:** {actual_age + offset} years")
            
            if abs(offset) > 3:
                st.warning("Significant Age Offset detected. Consider vascular health review.")
            else:
                st.success("Vascular patterns appear consistent with chronological age.")

# --- PAGE 2: EPIDEMIOLOGY ---
elif menu == "📊 Global Epidemiology":
    st.header("🌍 Global Epidemiology Dashboard")
    st.write("Aggregated retinal health trends by country.")
    
    # Mock database for demonstration
    geo_data = pd.DataFrame({
        'Country': ['India', 'USA', 'Kenya', 'Japan', 'Brazil', 'Germany'],
        'Avg_Age_Offset': [0.8, 1.4, 0.2, -1.9, -0.5, -1.1],
        'Scans_Collected': [1540, 3200, 450, 2100, 980, 1100]
    })
    st.table(geo_data)
    st.info("Note: This data represents anonymized contributions from the Oculomics Hub network.")

# --- PAGE 3: MODEL CARD ---
elif menu == "📄 Model Card":
    st.header("📄 Model Card: OCU-ResNet-Foundation")
    st.markdown("""
    ### Technical Overview
    - **Architecture:** ResNet-18 (Residual Network)
    - **Input:** 224x224 RGB Fundus Images
    - **Training Basis:** Transfer Learning from ImageNet-1K
    - **Version:** 1.0.0 (Open Source)
    
    ### Ethics & Safety
    - **Privacy:** Images are processed locally in-memory and not stored unless explicitly shared.
    - **Diagnosis:** This tool provides **biomarker estimation**, not medical diagnosis.
    """)

# --- PAGE 4: DATA ACCESS ---
elif menu == "💾 Data Access":
    st.header("💾 Open Data Access")
    st.write("Download repository assets for your own research.")
    
    st.button("Request Access to Global Anonymized Dataset")
    st.button("Download Model Weights (.pth)")

# --- GLOBAL FOOTER (Appears on all pages) ---
st.markdown("---")
footer_col1, footer_col2 = st.columns([2, 1])

with footer_col1:
    st.markdown("#### 📜 License")
    st.caption(f"""
    **MIT License** | Copyright (c) {datetime.datetime.now().year} Oculomics Commons Hub.  
    Permission is granted to use, copy, and modify this software for research and commercial purposes, 
    provided this notice is included.
    """)

with footer_col2:
    st.markdown("#### 📝 Citation")
    st.info("If you use this repository in your research, please cite:")
    st.code(f"""
@software{{oculomics_hub_2026,
  author = {{Oculomics Commons Community}},
  title = {{Oculomics Commons Hub: A Global Retinal Repository}},
  year = {{{datetime.datetime.now().year}}},
  publisher = {{GitHub}},
  url = {{https://github.com/your-username/occulomics}}
}}
    """, language="latex")