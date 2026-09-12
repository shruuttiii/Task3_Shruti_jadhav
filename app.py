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

st.markdown(
    """
    <style>
    
    /* ---------- APP ---------- */
    
    .stApp {
        background: #f5f7fb;
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
    
    
    /* ---------- HERO ---------- */
    
    .hero-box {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 3.5rem 2rem;
        border-radius: 28px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(15, 23, 42, 0.18);
    }
    
    .hero-title {
        color: white;
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .hero-text {
        color: #dbeafe;
        font-size: 1.05rem;
        margin-top: 0.8rem;
        line-height: 1.6;
    }
    
    
    /* ---------- SECTION HEADINGS ---------- */
    
    .section-heading {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 1.5rem;
        margin-bottom: 0.3rem;
    }
    
    .section-subheading {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    
    
    /* ---------- METRIC CARDS ---------- */
    
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    }
    
    
    /* ---------- BUTTON ---------- */
    
    .stButton > button {
        width: 100%;
        height: 3.3rem;
        border-radius: 14px;
        border: none;
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        color: white;
        font-size: 1.05rem;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
    }
    
    .stButton > button:hover {
        color: white;
        transform: translateY(-2px);
    }
    
    
    /* ---------- JOB CARD ---------- */
    
    .job-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        padding: 1.5rem;
        margin-top: 1.3rem;
        box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
    }
    
    .job-rank {
        color: #2563eb;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .job-name {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }
    
    .company-name {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    
    
    /* ---------- MATCH SCORE ---------- */
    
    .score-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    
    .score-number {
        color: #2563eb;
        font-size: 2rem;
        font-weight: 900;
    }
    
    .score-label {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    
    /* ---------- INFO BOX ---------- */
    
    .info-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.8rem;
        margin-top: 0.8rem;
    }
    
    .info-label {
        color: #94a3b8;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    
    .info-value {
        color: #334155;
        font-size: 0.9rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    
    
    /* ---------- SKILLS ---------- */
    
    .skill-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 16px;
        padding: 1rem;
        margin-top: 1rem;
        min-height: 100px;
    }
    
    .gap-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 16px;
        padding: 1rem;
        margin-top: 1rem;
        min-height: 100px;
    }
    
    .box-title {
        font-weight: 800;
        font-size: 0.85rem;
        color: #334155;
        margin-bottom: 0.5rem;
    }
    
    .box-content {
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    
    
    /* ---------- WHY THIS JOB ---------- */
    
    .why-box {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-top: 1rem;
    }
    
    .why-title {
        color: #1e3a8a;
        font-weight: 800;
        font-size: 0.9rem;
    }
    
    
    /* ---------- FOOTER ---------- */
    
    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 3rem;
        padding: 2rem;
        font-size: 0.85rem;
    }
    
    </style>
    """,
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

    # Final weighted score
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
    """
    <div class="hero-box">
        <div class="hero-title">
            💼 AI Career Navigator
        </div>
        <div class="hero-text">
            Find career opportunities personalized to your
            skills, experience, interests, location and salary preferences.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROFILE
# ============================================================

st.markdown(
    '<div class="section-heading">👤 Build Your Career Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subheading">'
    'Tell us about yourself and let the recommendation engine find your best matches.'
    '</div>',
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


st.write("")


# ============================================================
# RECOMMENDATION BUTTON
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


    # --------------------------------------------------------
    # RESULTS HEADER
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">✨ Your Top Career Matches</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subheading">'
        'Your opportunities are ranked using skill similarity and your career preferences.'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    best_score = (
        recommendations.iloc[0]["Final Score"] * 100
    )

    average_score = (
        recommendations["Final Score"].mean() * 100
    )


    metric1, metric2, metric3, metric4 = st.columns(4)


    with metric1:

        st.metric(
            "Jobs analyzed",
            f"{len(df_model):,}"
        )


    with metric2:

        st.metric(
            "Top matches",
            len(recommendations)
        )


    with metric3:

        st.metric(
            "Best match",
            f"{best_score:.0f}%"
        )


    with metric4:

        st.metric(
            "Average match",
            f"{average_score:.0f}%"
        )


    # --------------------------------------------------------
    # JOB CARDS
    # --------------------------------------------------------

    for i, (_, job) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        # Skill analysis
        matched, missing = get_skill_match(
            user_skills,
            job["Skill_List"]
        )

        score = job["Final Score"] * 100


        # ====================================================
        # JOB HEADER
        # ====================================================

        st.markdown(
            f"""
            <div class="job-card">

                <div class="job-rank">
                    #{i} RECOMMENDED OPPORTUNITY
                </div>

                <div class="job-name">
                    {job["Job Title"]}
                </div>

                <div class="company-name">
                    {job["Company"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # SCORE
        # ====================================================

        score_col, blank_col = st.columns([1, 3])


        with score_col:

            st.markdown(
                f"""
                <div class="score-box">

                    <div class="score-label">
                        Match Score
                    </div>

                    <div class="score-number">
                        {score:.0f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # JOB INFORMATION
        # ====================================================

        info1, info2, info3, info4 = st.columns(4)


        with info1:

            st.markdown(
                f"""
                <div class="info-box">
                    <div class="info-label">📍 Location</div>
                    <div class="info-value">{job["Location"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with info2:

            st.markdown(
                f"""
                <div class="info-box">
                    <div class="info-label">💼 Experience</div>
                    <div class="info-value">{job["Experience Level"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with info3:

            st.markdown(
                f"""
                <div class="info-box">
                    <div class="info-label">🏢 Industry</div>
                    <div class="info-value">{job["Industry"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with info4:

            st.markdown(
                f"""
                <div class="info-box">
                    <div class="info-label">💰 Salary</div>
                    <div class="info-value">{job["Salary"]:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # SKILLS
        # ====================================================

        skill_col, gap_col = st.columns(2)


        with skill_col:

            matched_text = (
                ", ".join(sorted(matched))
                if matched
                else "No direct skill matches"
            )

            st.markdown(
                f"""
                <div class="skill-box">

                    <div class="box-title">
                        ✅ Matching Skills
                    </div>

                    <div class="box-content">
                        {matched_text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with gap_col:

            missing_text = (
                ", ".join(sorted(missing))
                if missing
                else "No additional skills identified"
            )

            st.markdown(
                f"""
                <div class="gap-box">

                    <div class="box-title">
                        📚 Skills to Develop
                    </div>

                    <div class="box-content">
                        {missing_text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # WHY THIS JOB
        # ====================================================

        reasons = []


        if (
            str(job["Industry"]).lower()
            == user_industry.lower()
        ):

            reasons.append(
                "Industry preference matches"
            )


        if (
            str(job["Experience Level"]).lower()
            == user_experience.lower()
        ):

            reasons.append(
                "Experience level matches"
            )


        if (
            str(job["Location"]).lower()
            == user_location.lower()
        ):

            reasons.append(
                "Preferred location matches"
            )


        if job["Salary"] >= user_min_salary:

            reasons.append(
                "Meets your minimum salary preference"
            )


        if matched:

            reasons.append(
                f"{len(matched)} skill(s) match the job requirements"
            )


        if not reasons:

            reasons.append(
                "Recommended based primarily on skill similarity"
            )


        reason_text = " • ".join(reasons)


        st.markdown(
            f"""
            <div class="why-box">

                <div class="why-title">
                    💡 Why this job?
                </div>

                <div>
                    {reason_text}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # SCORE BREAKDOWN
        # ====================================================

        with st.expander("🔍 View recommendation score breakdown"):

            breakdown1, breakdown2 = st.columns(2)


            with breakdown1:

                st.write(
                    f"**Skill similarity — "
                    f"{job['Skill Similarity'] * 100:.1f}%**"
                )

                st.progress(
                    float(job["Skill Similarity"])
                )


                st.write(
                    f"**Industry match — "
                    f"{job['Industry Match'] * 100:.0f}%**"
                )

                st.progress(
                    float(job["Industry Match"])
                )


                st.write(
                    f"**Experience match — "
                    f"{job['Experience Match'] * 100:.0f}%**"
                )

                st.progress(
                    float(job["Experience Match"])
                )


            with breakdown2:

                st.write(
                    f"**Location match — "
                    f"{job['Location Match'] * 100:.0f}%**"
                )

                st.progress(
                    float(job["Location Match"])
                )


                st.write(
                    f"**Salary score — "
                    f"{job['Salary Score'] * 100:.0f}%**"
                )

                st.progress(
                    float(job["Salary Score"])
                )


                st.info(
                    "Final Score = 50% Skills + 20% Industry + "
                    "15% Experience + 10% Location + 5% Salary"
                )


        st.write("")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>AI Career Navigator</b>
        <br>
        Personalized career recommendations powered by
        similarity-based matching.
    </div>
    """,
    unsafe_allow_html=True
)
