import streamlit as st
import pandas as pd
import pickle
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CareerCompass AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #f7f9fc;
    }

    /* Hide Streamlit default menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hero */
    .hero {
        padding: 3rem 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #111827 0%, #1e3a5f 100%);
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }

    .hero h1 {
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.15rem;
        color: #dbeafe;
        max-width: 700px;
        margin: auto;
    }

    /* Section titles */
    .section-title {
        font-size: 1.6rem;
        font-weight: 750;
        color: #111827;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Profile card */
    .profile-card {
        background: white;
        padding: 1.8rem;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.5rem;
    }

    /* Recommendation card */
    .job-card {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.2rem;
        min-height: 320px;
    }

    .job-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .company {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    .match-score {
        font-size: 1.
