import streamlit as st
import pandas as pd
import pickle
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Career Navigator",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .stApp {
        background: #f7f9fc;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* HERO */
    .hero {
        padding: 3rem 2rem;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e3a5f 100%
        );
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

    /* SECTION TITLES */
    .section-title {
        font-size: 1.6rem;
        font-weight: 750;
        color: #111827;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* PROFILE CARD */
    .profile-card {
        background: white;
        padding: 1.8rem;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.5rem;
    }

    /* JOB CARD */
    .job-card {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.2rem;
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
        font-size: 1.8rem;
        font-weight: 800;
        color: #2563eb;
        margin-bottom: 0.8rem;
    }

    .job-info {
        color: #475569;
        font-size: 0.9rem;
        margin: 0.35rem 0;
    }

    /* SKILL BOXES */
    .skill-box {
        background: #f0fdf4;
        border-radius: 12px;
        padding: 0.8rem;
        margin-top: 1rem;
        min-height: 80px;
    }

    .gap-box {
        background: #fff7ed;
        border-radius: 12px;
        padding: 0.8rem;
        margin-top: 1rem;
        min-height: 80px;
    }

    .label {
        font-weight: 700;
        font-size: 0.85rem;
        color: #334155;
        margin-bottom: 0.4rem;
    }

    /* WHY THIS JOB */
    .reason-box {
        background: #eff6ff;
        border-radius: 12px;
        padding: 0.9rem;
        margin-top: 1rem;
    }

    .reason-title {
        font-weight: 700;
        font-size: 0.85rem;
        color: #1e3a8a;
        margin-bottom: 0.4rem;
    }

    .reason-text {
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* BUTTON */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3.2rem;
        font-size: 1.05rem;
        font-weight: 700;
        border: none;
        background: #2563eb;
        color: white;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        color: white;
    }

    /* METRICS */
    [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_model():

    with open("job_data.pkl", "rb") as file:
        data = pickle.load(file)

    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    skill_matrix = load_npz("job_skill_matrix.npz")

    return data, vectorizer, skill_matrix


# Load saved model components
df_model, vectorizer, job_skill_matrix = load_model()


# ============================================================
# SKILL MATCHING
# ============================================================

def get_skill_match(user_skills, job_skills):

    user_skills = set(
        skill.lower().strip()
        for skill in user_skills
    )

    job_skills = set(
        skill.lower().strip()
        for skill in job_skills
    )

    matched = user_skills.intersection(job_skills)

    missing = job_skills - user_skills

    return list(matched), list(missing)


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommend_jobs(
    skills,
    experience,
    industry,
    location,
    min_salary,
    top_n=10
):

    # Clean user skills
    user_skills = [
        skill.strip().lower()
        for skill in skills
        if skill.strip()
    ]

    # Convert user skills into text
    user_skill_text = " ".join(user_skills)

    # Transform user skills using trained TF-IDF vectorizer
    user_vector = vectorizer.transform(
        [user_skill_text]
    )

    # Calculate cosine similarity
    skill_scores = cosine_similarity(
        user_vector,
        job_skill_matrix
    ).flatten()

    # Copy dataset
    results = df_model.copy()

    # Skill similarity
    results["Skill Similarity"] = skill_scores

    # Industry match
    results["Industry Match"] = (
        results["Industry"]
        .astype(str)
        .str.lower()
        == industry.lower()
    ).astype(float)

    # Experience match
    results["Experience Match"] = (
        results["Experience Level"]
        .astype(str)
        .str.lower()
        == experience.lower()
    ).astype(float)

    # Location match
    results["Location Match"] = (
        results["Location"]
        .astype(str)
        .str.lower()
        == location.lower()
    ).astype(float)

    # Salary score
    results["Salary Score"] = (
        results["Salary"] / min_salary
    ).clip(upper=1)

    # ========================================================
    # FINAL RECOMMENDATION SCORE
    # ========================================================

    results["Final Score"] = (
        0.50 * results["Skill Similarity"]
        + 0.20 * results["Industry Match"]
        + 0.15 * results["Experience Match"]
        + 0.10 * results["Location Match"]
        + 0.05 * results["Salary Score"]
    )

    # Sort by highest recommendation score
    results = results.sort_values(
        by="Final Score",
        ascending=False
    ).head(top_n)

    return results


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

    <h1>💼 AI Career Navigator</h1>

    <p>
        Discover career opportunities that match your
        skills, experience, interests, location,
        and salary preferences.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# USER PROFILE SECTION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Build Your Career Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="profile-card">',
    unsafe_allow_html=True
)


# Skills input
skills_input = st.text_input(
    "Your skills",
    placeholder="Example: Python, SQL, Machine Learning"
)


# Three preference columns
col1, col2, col3 = st.columns(3)


with col1:

    experience = st.selectbox(
        "Experience level",
        sorted(
            df_model["Experience Level"]
            .dropna()
            .unique()
        )
    )


with col2:

    industry = st.selectbox(
        "Preferred industry",
        sorted(
            df_model["Industry"]
            .dropna()
            .unique()
        )
    )


with col3:

    location = st.selectbox(
        "Preferred location",
        sorted(
            df_model["Location"]
            .dropna()
            .unique()
        )
    )


# Salary preference
min_salary = st.slider(
    "Minimum salary preference",
    min_value=int(df_model["Salary"].min()),
    max_value=int(df_model["Salary"].max()),
    value=80000,
    step=5000
)


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# RECOMMENDATION BUTTON
# ============================================================

if st.button("✨ Find My Best Matches"):

    if not skills_input.strip():

        st.warning(
            "Please enter at least one skill to get recommendations."
        )

    else:

        # Convert comma-separated skills into a list
        skills = [
            skill.strip()
            for skill in skills_input.split(",")
            if skill.strip()
        ]

        # Generate recommendations
        recommendations = recommend_jobs(
            skills=skills,
            experience=experience,
            industry=industry,
            location=location,
            min_salary=min_salary,
            top_n=10
        )

        # Store results in session state
        st.session_state["recommendations"] = recommendations

        st.session_state["user_skills"] = skills

        st.session_state["user_experience"] = experience

        st.session_state["user_industry"] = industry

        st.session_state["user_location"] = location

        st.session_state["user_min_salary"] = min_salary


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

if "recommendations" in st.session_state:

    recommendations = st.session_state["recommendations"]

    user_skills = st.session_state["user_skills"]

    user_experience = st.session_state["user_experience"]

    user_industry = st.session_state["user_industry"]

    user_location = st.session_state["user_location"]

    user_min_salary = st.session_state["user_min_salary"]


    # --------------------------------------------------------
    # SECTION TITLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">✨ Your Top Career Matches</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Jobs analyzed",
            f"{len(df_model):,}"
        )


    with c2:

        st.metric(
            "Recommendations",
            len(recommendations)
        )


    with c3:

        best_score = (
            recommendations.iloc[0]["Final Score"]
            * 100
        )

        st.metric(
            "Best match",
            f"{best_score:.0f}%"
        )


    st.write("")


    # --------------------------------------------------------
    # DISPLAY EACH JOB
    # --------------------------------------------------------

    for i, (_, job) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        # Find matched and missing skills
        matched, missing = get_skill_match(
            user_skills,
            job["Skill_List"]
        )


        # Calculate percentage
        score = job["Final Score"] * 100


        # ----------------------------------------------------
        # JOB CARD
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="job-card">

                <div class="job-title">
                    #{i} {job["Job Title"]}
                </div>

                <div class="company">
                    {job["Company"]}
                </div>

                <div class="match-score">
                    {score:.0f}% Match
                </div>

                <div class="job-info">
                    📍 <b>Location:</b>
                    {job["Location"]}
                </div>

                <div class="job-info">
                    💼 <b>Experience:</b>
                    {job["Experience Level"]}
                </div>

                <div class="job-info">
                    🏢 <b>Industry:</b>
                    {job["Industry"]}
                </div>

                <div class="job-info">
                    💰 <b>Salary:</b>
                    {job["Salary"]:,.0f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # MATCHING SKILLS + SKILLS TO DEVELOP
        # ----------------------------------------------------

        col_a, col_b = st.columns(2)


        with col_a:

            matched_text = (
                ", ".join(matched)
                if matched
                else "No direct skill matches"
            )

            st.markdown(
                f"""
                <div class="skill-box">

                    <div class="label">
                        ✅ Matching skills
                    </div>

                    {matched_text}

                </div>
                """,
                unsafe_allow_html=True
            )


        with col_b:

            missing_text = (
                ", ".join(missing)
                if missing
                else "No additional skills identified"
            )

            st.markdown(
                f"""
                <div class="gap-box">

                    <div class="label">
                        📚 Skills to develop
                    </div>

                    {missing_text}

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # WHY THIS JOB?
        # ----------------------------------------------------

        reasons = []


        if (
            str(job["Industry"]).lower()
            == user_industry.lower()
        ):

            reasons.append(
                "Industry matches your preference"
            )


        if (
            str(job["Experience Level"]).lower()
            == user_experience.lower()
        ):

            reasons.append(
                "Experience level matches your profile"
            )


        if (
            str(job["Location"]).lower()
            == user_location.lower()
        ):

            reasons.append(
                "Location matches your preference"
            )


        if job["Salary"] >= user_min_salary:

            reasons.append(
                "Meets your minimum salary preference"
            )


        if matched:

            reasons.append(
                f"{len(matched)} of your skills match the job requirements"
            )


        if reasons:

            reason_text = "<br>• ".join(reasons)

        else:

            reason_text = (
                "Recommended primarily because of "
                "skill similarity."
            )


        st.markdown(
            f"""
            <div class="reason-box">

                <div class="reason-title">
                    💡 Why this job?
                </div>

                <div class="reason-text">
                    • {reason_text}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        padding-top:3rem;
        font-size:0.85rem;
    ">

        AI Career Navigator
        • Personalized Career Recommendation System

    </div>
    """,
    unsafe_allow_html=True
)
