import streamlit as st
import pandas as pd
import pickle
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from textwrap import dedent


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

st.markdown(
    dedent("""
    <style>

    /* ================================
       GLOBAL
       ================================ */

    .stApp {
        background: #f6f8fc;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ================================
       HERO
       ================================ */

    .hero {
        padding: 3.5rem 2rem;
        border-radius: 28px;
        background:
            radial-gradient(
                circle at top right,
                rgba(59, 130, 246, 0.35),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #0f172a 0%,
                #172554 55%,
                #1e3a8a 100%
            );
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);
    }

    .hero-badge {
        display: inline-block;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #dbeafe;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .hero h1 {
        font-size: 3.4rem;
        font-weight: 850;
        letter-spacing: -1.5px;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #dbeafe;
        max-width: 720px;
        margin: 1rem auto 0 auto;
        line-height: 1.7;
    }


    /* ================================
       SECTION TITLE
       ================================ */

    .section-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .section-description {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: -0.5rem;
        margin-bottom: 1.2rem;
    }


    /* ================================
       PROFILE CARD
       ================================ */

    .profile-card {
        background: white;
        padding: 1.8rem;
        border-radius: 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 35px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.5rem;
    }


    /* ================================
       BUTTON
       ================================ */

    .stButton > button {
        width: 100%;
        height: 3.4rem;
        border-radius: 14px;
        border: none;
        background: linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );
        color: white;
        font-size: 1.05rem;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(37, 99, 235, 0.35);
        color: white;
    }


    /* ================================
       METRIC CARDS
       ================================ */

    [data-testid="stMetric"] {
        background: white;
        padding: 1.1rem;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
    }


    /* ================================
       JOB CARD
       ================================ */

    .job-card {
        background: white;
        border-radius: 24px;
        border: 1px solid #e2e8f0;
        padding: 1.7rem;
        margin-top: 1.5rem;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.07);
    }

    .job-number {
        display: inline-block;
        background: #eff6ff;
        color: #2563eb;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }

    .job-title {
        font-size: 1.45rem;
        font-weight: 850;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }

    .company-name {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1rem;
    }


    /* ================================
       MATCH SCORE
       ================================ */

    .match-container {
        background: #f8fafc;
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
        border: 1px solid #e2e8f0;
    }

    .match-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .match-score {
        font-size: 2rem;
        font-weight: 900;
        color: #2563eb;
        margin-top: 0.1rem;
    }


    /* ================================
       JOB INFORMATION
       ================================ */

    .info-pill {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .info-label {
        color: #94a3b8;
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.4px;
    }

    .info-value {
        color: #334155;
        font-size: 0.9rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }


    /* ================================
       SKILL BOXES
       ================================ */

    .skill-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 16px;
        padding: 1rem;
        min-height: 105px;
        margin-top: 1rem;
    }

    .gap-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 16px;
        padding: 1rem;
        min-height: 105px;
        margin-top: 1rem;
    }

    .box-title {
        font-weight: 800;
        font-size: 0.85rem;
        margin-bottom: 0.55rem;
    }

    .skill-content {
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.6;
    }


    /* ================================
       WHY THIS JOB
       ================================ */

    .reason-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-top: 1rem;
    }

    .reason-title {
        color: #1e3a8a;
        font-weight: 800;
        font-size: 0.9rem;
        margin-bottom: 0.55rem;
    }

    .reason-content {
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.7;
    }


    /* ================================
       FOOTER
       ================================ */

    .custom-footer {
        text-align: center;
        color: #94a3b8;
        padding-top: 3rem;
        padding-bottom: 1rem;
        font-size: 0.85rem;
    }

    .footer-brand {
        color: #64748b;
        font-weight: 800;
    }

    </style>
    """),
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("job_data.pkl", "rb") as file:
        data = pickle.load(file)

    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    skill_matrix = load_npz("job_skill_matrix.npz")

    return data, vectorizer, skill_matrix


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

    user_skills = [
        skill.strip().lower()
        for skill in skills
        if skill.strip()
    ]

    user_skill_text = " ".join(user_skills)

    user_vector = vectorizer.transform(
        [user_skill_text]
    )

    skill_scores = cosine_similarity(
        user_vector,
        job_skill_matrix
    ).flatten()

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

    # Final recommendation score
    results["Final Score"] = (
        0.50 * results["Skill Similarity"]
        + 0.20 * results["Industry Match"]
        + 0.15 * results["Experience Match"]
        + 0.10 * results["Location Match"]
        + 0.05 * results["Salary Score"]
    )

    results = (
        results
        .sort_values(
            by="Final Score",
            ascending=False
        )
        .head(top_n)
    )

    return results


# ============================================================
# HERO
# ============================================================

st.markdown(
    dedent("""
    <div class="hero">

        <div class="hero-badge">
            ✦ AI-POWERED CAREER RECOMMENDATIONS
        </div>

        <h1>AI Career Navigator</h1>

        <div class="hero-subtitle">
            Discover opportunities that align with your skills,
            experience, interests, location, and salary preferences.
        </div>

    </div>
    """),
    unsafe_allow_html=True
)


# ============================================================
# PROFILE SECTION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Build Your Career Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Tell us about yourself and the recommendation engine will find '
    'the opportunities that best match your profile.'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="profile-card">',
    unsafe_allow_html=True
)


# Skills
skills_input = st.text_input(
    "Your skills",
    placeholder="Example: Python, SQL, Machine Learning"
)


# Preferences
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


# Salary
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
# FIND MATCHES BUTTON
# ============================================================

if st.button("✨ Find My Best Matches"):

    if not skills_input.strip():

        st.warning(
            "Please enter at least one skill to get recommendations."
        )

    else:

        skills = [
            skill.strip()
            for skill in skills_input.split(",")
            if skill.strip()
        ]

        recommendations = recommend_jobs(
            skills=skills,
            experience=experience,
            industry=industry,
            location=location,
            min_salary=min_salary,
            top_n=10
        )

        st.session_state["recommendations"] = recommendations
        st.session_state["user_skills"] = skills
        st.session_state["user_experience"] = experience
        st.session_state["user_industry"] = industry
        st.session_state["user_location"] = location
        st.session_state["user_min_salary"] = min_salary


# ============================================================
# RESULTS
# ============================================================

if "recommendations" in st.session_state:

    recommendations = st.session_state["recommendations"]

    user_skills = st.session_state["user_skills"]

    user_experience = st.session_state["user_experience"]

    user_industry = st.session_state["user_industry"]

    user_location = st.session_state["user_location"]

    user_min_salary = st.session_state["user_min_salary"]


    # ========================================================
    # RESULTS HEADER
    # ========================================================

    st.markdown(
        '<div class="section-title">✨ Your Top Career Matches</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Here are the opportunities ranked according to your personalized profile.'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    best_score = (
        recommendations.iloc[0]["Final Score"] * 100
    )

    average_score = (
        recommendations["Final Score"].mean() * 100
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Jobs analyzed",
            f"{len(df_model):,}"
        )


    with c2:

        st.metric(
            "Top matches",
            len(recommendations)
        )


    with c3:

        st.metric(
            "Best match",
            f"{best_score:.0f}%"
        )


    with c4:

        st.metric(
            "Average match",
            f"{average_score:.0f}%"
        )


    st.write("")


    # ========================================================
    # JOB RESULTS
    # ========================================================

    for i, (_, job) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        # ----------------------------------------------------
        # Skills
        # ----------------------------------------------------

        matched, missing = get_skill_match(
            user_skills,
            job["Skill_List"]
        )


        score = job["Final Score"] * 100


        # ----------------------------------------------------
        # Reasons
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


        if not reasons:

            reasons.append(
                "Strong similarity with your overall skill profile"
            )


        # ----------------------------------------------------
        # JOB CARD
        # ----------------------------------------------------

        st.markdown(
            dedent(
                f"""
                <div class="job-card">

                    <div class="job-number">
                        #{i} RECOMMENDED OPPORTUNITY
                    </div>

                    <div class="job-title">
                        {job["Job Title"]}
                    </div>

                    <div class="company-name">
                        {job["Company"]}
                    </div>

                    <div class="match-container">

                        <div class="match-label">
                            Compatibility Score
                        </div>

                        <div class="match-score">
                            {score:.0f}%
                        </div>

                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # JOB INFORMATION
        # ----------------------------------------------------

        info1, info2, info3, info4 = st.columns(4)


        with info1:

            st.markdown(
                dedent(
                    f"""
                    <div class="info-pill">

                        <div class="info-label">
                            📍 Location
                        </div>

                        <div class="info-value">
                            {job["Location"]}
                        </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )


        with info2:

            st.markdown(
                dedent(
                    f"""
                    <div class="info-pill">

                        <div class="info-label">
                            💼 Experience
                        </div>

                        <div class="info-value">
                            {job["Experience Level"]}
                        </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )


        with info3:

            st.markdown(
                dedent(
                    f"""
                    <div class="info-pill">

                        <div class="info-label">
                            🏢 Industry
                        </div>

                        <div class="info-value">
                            {job["Industry"]}
                        </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )


        with info4:

            st.markdown(
                dedent(
                    f"""
                    <div class="info-pill">

                        <div class="info-label">
                            💰 Salary
                        </div>

                        <div class="info-value">
                            {job["Salary"]:,.0f}
                        </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # MATCHING + GAP
        # ----------------------------------------------------

        col_a, col_b = st.columns(2)


        with col_a:

            matched_text = (
                ", ".join(sorted(matched))
                if matched
                else "No direct skill matches"
            )

            st.markdown(
                dedent(
                    f"""
                    <div class="skill-box">

                        <div class="box-title">
                            ✅ Matching Skills
                        </div>

                        <div class="skill-content">
                            {matched_text}
                        </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )


        with col_b:

            missing_text = (
                ", ".join(sorted(missing))
                if missing
                else "No additional skills identified"
            )

            st.markdown(
                dedent(
                    f"""
                    <div class="gap-box">

                        <div class="box-title">
                            📚 Skills to Develop
                        </div>

                        <div class="skill-content">
                            {missing_text}
                        </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # WHY THIS JOB
        # ----------------------------------------------------

        reason_text = "<br>• ".join(reasons)

        st.markdown(
            dedent(
                f"""
                <div class="reason-box">

                    <div class="reason-title">
                        💡 Why this job?
                    </div>

                    <div class="reason-content">
                        • {reason_text}
                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # SCORE BREAKDOWN
        # ----------------------------------------------------

        with st.expander("🔍 View recommendation score breakdown"):

            score_col1, score_col2 = st.columns(2)


            with score_col1:

                st.write(
                    f"**Skill similarity:** "
                    f"{job['Skill Similarity'] * 100:.1f}%"
                )

                st.progress(
                    float(job["Skill Similarity"])
                )


                st.write(
                    f"**Industry match:** "
                    f"{job['Industry Match'] * 100:.0f}%"
                )

                st.progress(
                    float(job["Industry Match"])
                )


            with score_col2:

                st.write(
                    f"**Experience match:** "
                    f"{job['Experience Match'] * 100:.0f}%"
                )

                st.progress(
                    float(job["Experience Match"])
                )


                st.write(
                    f"**Location match:** "
                    f"{job['Location Match'] * 100:.0f}%"
                )

                st.progress(
                    float(job["Location Match"])
                )


        st.write("")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    dedent("""
    <div class="custom-footer">

        <span class="footer-brand">
            AI Career Navigator
        </span>

        <br>

        Personalized career recommendations powered by
        similarity-based matching.

    </div>
    """),
    unsafe_allow_html=True
)
