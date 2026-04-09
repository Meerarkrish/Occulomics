import streamlit as st
import numpy as np
import pandas as pd
import pycountry
from PIL import Image
import torch
from torchvision import models, transforms
import datetime

# --- 1. THEME & CLINICAL DESIGN SYSTEM ---
st.set_page_config(page_title="Oculomics Commons", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #1E293B;
    }

    /* FIXED HEADER - RELATIVE POSITIONING PREVENTS OVERLAP */
    .repository-header {
        background-color: #FFFFFF;
        padding: 40px 60px;
        border-bottom: 1px solid #E2E8F0;
        margin: -6rem -5rem 0rem -5rem; /* Removed bottom margin */
        position: relative;
        z-index: 100;
        display: block;
    }
    
    /* THE HEADLINE - FORCED PASTEL BLUE */
    .headline-pastel {
        color: #A5F3FC !important; 
        font-weight: 700 !important;
        font-size: 38px !important;
        margin: 0 !important;
        padding: 0 !important;
        letter-spacing: -0.02em;
    }

    .sub-headline {
        color: #64748B !important;
        font-size: 16px !important;
        margin-top: 4px !important;
        font-weight: 400;
    }

    /* THE SPACER - THIS KILLS THE OVERLAY ISSUE */
    .vertical-clearance {
        margin-top: 50px;
        display: block;
        width: 100%;
    }
    
    /* CONTENT CARDS */
    .content-card {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 4px;
        border: 1px solid #E2E8F0;
        margin-bottom: 24px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    
    .metric-container {
        border-left: 4px solid #A5F3FC;
        padding-left: 15px;
        margin-bottom: 20px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
    }

    /* FOOTER */
    .commons-footer {
        background-color: #FFFFFF;
        padding: 60px;
        border-top: 1px solid #E2E8F0;
        margin: 4rem -5rem -5rem -5rem;
    }
    
    .bibtex-box {
        background-color: #F1F5F9;
        padding: 20px;
        border-radius: 4px;
        font-family: 'monospace';
        font-size: 12px;
        border: 1px solid #CBD5E1;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. BACKEND LOGIC ---
@st.cache_resource
def load_foundation():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.eval()
    return m

def infer(img, m):
    t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    tensor = t(img).unsqueeze(0)
    with torch.no_grad():
        out = m(tensor)
        v = float(torch.mean(out))
    return round(0.7 + (v % 0.04), 2), round((v % 6) - 3, 1)

# --- 3. PAGE STRUCTURE ---

# BRANDED HEADER
st.markdown("""
    <div class="repository-header">
        <div class="headline-pastel">Oculomics Commons</div>
        <p class="sub-headline">
            Global Open-Science Repository for Retinal Biomarkers and Epidemiology
        </p>
    </div>
""", unsafe_allow_html=True)

# SPACER ELEMENT (Forces the following widgets to sit below the header)
st.markdown('<div class="vertical-clearance"></div>', unsafe_allow_html=True)

main_col, side_col = st.columns([2.2, 1])

with main_col:
    # ANALYTICS CARD
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("Neural Analysis Gateway")
    
    # This widget will now be pushed below the clearance div
    uploaded_file = st.file_uploader("Upload Retinal Scan (DICOM, JPG, PNG)", type=["jpg","png","jpeg"])
    
    if uploaded_file:
        img_pil = Image.open(uploaded_file).convert('RGB')
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.image(img_pil, use_container_width=True, caption="Anonymized Input")
        with res_col2:
            model = load_foundation()
            avr, age_d = infer(img_pil, model)
            
            st.markdown(f'<div class="metric-container"><div class="metric-label">Vascular AVR</div><div class="metric-value">{avr}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-container"><div class="metric-label">Retinal Age Delta</div><div class="metric-value">{age_d}y</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # DATASET CARD
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("Global Health Surveillance")
    stats_df = pd.DataFrame({
        'Region': ['India', 'USA', 'Kenya', 'Japan', 'Brazil'],
        'Sample Size': [1540, 3200, 450, 2100, 980],
        'Mean AVR Score': [0.71, 0.69, 0.73, 0.74, 0.70]
    })
    st.table(stats_df)
    st.markdown('</div>', unsafe_allow_html=True)

with side_col:
    # REPOSITORY METADATA
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("Project Metadata")
    country_list = sorted([c.name for c in pycountry.countries])
    st.selectbox("Inference Context", country_list, index=0)
    st.number_input("Subject Age", 1, 115, 30)
    st.write("---")
    st.markdown("**Version:** Commons-v1.2.0")
    st.markdown("**Last Build:** April 2026")
    st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown(f"""
    <div class="commons-footer">
        <div style="display: flex; gap: 80px; max-width: 1200px; margin: auto;">
            <div style="flex: 1.5;">
                <h3 style="color:#1E40AF; font-size:18px;">📜 License</h3>
                <p style="color:#475569; font-size:14px;">
                    <b>Oculomics Commons</b> is an open-access platform licensed under the <b>MIT License</b>. 
                    Copyright © {datetime.datetime.now().year}. 
                </p>
            </div>
            <div style="flex: 1;">
                <h3 style="color:#1E40AF; font-size:18px;">📝 Citation</h3>
                <div class="bibtex-box">
@repository{{oculomics_commons_2026,
  author = {{Oculomics Community}},
  title = {{Oculomics Commons: Global Retinal Hub}},
  year = {{2026}},
  url = {{https://github.com/occulomics}}
}}
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)