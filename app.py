import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Career Navigator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PREMIUM UI
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.16), transparent 30%),
        radial-gradient(circle at 90% 15%, rgba(6,182,212,0.12), transparent 30%),
        linear-gradient(135deg, #070b1a 0%, #0b1024 45%, #10152e 100%);
    color: #f8fafc;
}

/* Main container */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Remove Streamlit header space */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Hero */
.hero {
    padding: 42px 10px 35px 10px;
    text-align: center;
}

.hero-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 999px;
    background: rgba(99,102,241,0.14);
    border: 1px solid rgba(129,140,248,0.35);
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 52px;
    line-height: 1.05;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(
        90deg,
        #ffffff 10%,
        #a5b4fc 45%,
        #67e8f9 90%
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 760px;
    margin: 18px auto 0 auto;
    color: #a8b1c7;
    font-size: 17px;
    line-height: 1.7;
}

/* Section titles */
.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #f8fafc;
    margin-top: 25px;
    margin-bottom: 7px;
}

.section-subtitle {
    color: #8f9ab2;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Input labels */
label {
    color: #dbe4f5 !important;
    font-weight: 600 !important;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stMultiSelect div[data-baseweb="select"],
.stSelectbox div[data-baseweb="select"] {
    background: rgba(15,23,42,0.82) !important;
    border: 1px solid rgba(148,163,184,0.18) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Multiselect tags */
.stMultiSelect span[data-baseweb="tag"] {
    background: linear-gradient(135deg, #6366f1, #06b6d4) !important;
    border: none !important;
    color: white !important;
}

/* Main button */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 13px;
    padding: 14px 22px;
    background: linear-gradient(135deg, #6366f1, #7c3aed, #06b6d4);
    color: white;
    font-size: 16px;
    font-weight: 750;
    box-shadow: 0 10px 30px rgba(99,102,241,0.25);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(99,102,241,0.38);
}

/* Cards */
.card {
    background: linear-gradient(
        145deg,
        rgba(20,27,53,0.95),
        rgba(12,18,39,0.95)
    );
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 18px;
    padding: 22px;
    margin: 10px 0;
    box-shadow: 0 15px 40px rgba(0,0,0,0.18);
}

/* Recommendation card */
.job-title {
    font-size: 20px;
    font-weight: 750;
    color: #ffffff;
}

.company-name {
    color: #a5b4fc;
    font-size: 14px;
    margin-top: 5px;
}

.score {
    font-size: 30px;
    font-weight: 800;
    color: #67e8f9;
}

.small-label {
    color: #7f8aa3;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}

/* Skill chips */
.skill-chip {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px 5px 4px 0;
    border-radius: 999px;
    background: rgba(99,102,241,0.14);
    border: 1px solid rgba(129,140,248,0.32);
    color: #c7d2fe;
    font-size: 12px;
    font-weight: 600;
}

.missing-chip {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px 5px 4px 0;
    border-radius: 999px;
    background: rgba(245,158,11,0.10);
    border: 1px solid rgba(245,158,11,0.30);
    color: #fcd34d;
    font-size: 12px;
    font-weight: 600;
}

/* Metric */
[data-testid="stMetric"] {
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(148,163,184,0.12);
    padding: 15px;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

/* Progress */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #06b6d4);
}

/* Expander */
.streamlit-expanderHeader {
    color: #e2e8f0 !important;
    font-weight: 650;
}

/* Divider */
hr {
    border-color: rgba(148,163,184,0.10) !important;
}

/* Footer */
.footer {
    text-align: center;
    padding: 35px 0 10px 0;
    color: #64748b;
    font-size: 12px;
}

.footer strong {
    color: #94a3b8;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_resource
def load_model():

    with open("job_data.pkl", "rb") as file:
        data = pickle.load(file)

    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    job_skill_matrix = load_npz("job_skill_matrix.npz")

    return data, vectorizer, job_skill_matrix


try:
    df_model, vectorizer, job_skill_matrix = load_model()

except Exception as e:

    st.error("⚠️ Model files could not be loaded.")

    st.code(str(e))

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_skill_match(user_skills, job_skills):

    user_skills = set(
        skill.lower().strip()
        for skill in user_skills
        if skill.strip()
    )

    job_skills = set(
        skill.lower().strip()
        for skill in job_skills
        if skill.strip()
    )

    matched = user_skills.intersection(job_skills)

    missing = job_skills - user_skills

    return list(matched), list(missing)


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

    user_vector = vectorizer.transform([user_skill_text])

    skill_scores = cosine_similarity(
        user_vector,
        job_skill_matrix
    ).flatten()

    results = df_model.copy()

    results["Skill Similarity"] = skill_scores

    results["Industry Match"] = (
        results["Industry"].str.lower()
        == industry.lower()
    ).astype(float)

    results["Experience Match"] = (
        results["Experience Level"].str.lower()
        == experience.lower()
    ).astype(float)

    results["Location Match"] = (
        results["Location"].str.lower()
        == location.lower()
    ).astype(float)

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

    recommendations = []

    for _, job in results.iterrows():

        matched, missing = get_skill_match(
            user_skills,
            job["Skill_List"]
        )

        recommendations.append({

            "Job Title": job["Job Title"],

            "Company": job["Company"],

            "Location": job["Location"],

            "Experience": job["Experience Level"],

            "Salary": job["Salary"],

            "Industry": job["Industry"],

            "Final Score": round(
                job["Final Score"],
                3
            ),

            "Skill Similarity": job["Skill Similarity"],

            "Industry Match": job["Industry Match"],

            "Experience Match": job["Experience Match"],

            "Location Match": job["Location Match"],

            "Salary Score": job["Salary Score"],

            "Matched Skills": matched,

            "Skills to Develop": missing

        })

    return pd.DataFrame(recommendations)


def skill_chips(skills, missing=False):

    if not skills:

        st.write("No skills found.")

        return

    css_class = "missing-chip" if missing else "skill-chip"

    chips = ""

    for skill in skills:

        chips += f"""
        <span class="{css_class}">
            {skill.title()}
        </span>
        """

    st.markdown(
        chips,
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-badge">
✦ AI-POWERED CAREER INTELLIGENCE
</div>

<div class="hero-title">
AI Career Navigator
</div>

<div class="hero-subtitle">
Discover career opportunities that actually match your
skills, experience, interests and preferences.
Our recommendation engine analyzes your profile and
ranks opportunities using personalized similarity scoring.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PROFILE SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Build Your Career Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Tell us what you are looking for and let the recommendation engine do the matching.'
    '</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)

with col1:

    skills_input = st.multiselect(
        "Your Skills",
        options=sorted(
            list(
                vectorizer.get_feature_names_out()
            )
        ),
        default=[
            skill
            for skill in [
                "python",
                "sql",
                "machine learning"
            ]
            if skill in vectorizer.get_feature_names_out()
        ],
        help="Select the skills you currently have."
    )

    experience = st.selectbox(
        "Experience Level",
        sorted(
            df_model["Experience Level"]
            .dropna()
            .unique()
        )
    )

    industry = st.selectbox(
        "Preferred Industry",
        sorted(
            df_model["Industry"]
            .dropna()
            .unique()
        )
    )


with col2:

    location = st.selectbox(
        "Preferred Location",
        sorted(
            df_model["Location"]
            .dropna()
            .unique()
        )
    )

    min_salary = st.number_input(
        "Minimum Salary Preference",
        min_value=0,
        value=80000,
        step=5000
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=15,
        value=10
    )


# ============================================================
# SELECTED SKILLS PREVIEW
# ============================================================

if skills_input:

    st.markdown(
        '<div class="small-label">YOUR CURRENT SKILL SET</div>',
        unsafe_allow_html=True
    )

    skill_chips(skills_input)


st.write("")


# ============================================================
# RECOMMEND BUTTON
# ============================================================

if st.button(
    "🚀 Find My Best Career Matches",
    use_container_width=True
):

    if len(skills_input) == 0:

        st.warning(
            "Please select at least one skill."
        )

        st.stop()

    with st.spinner(
        "Analyzing your profile and finding the best matches..."
    ):

        recommendations = recommend_jobs(
            skills=skills_input,
            experience=experience,
            industry=industry,
            location=location,
            min_salary=min_salary,
            top_n=top_n
        )


    if recommendations.empty:

        st.warning(
            "No recommendations found."
        )

        st.stop()


    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Your Recommendation Overview</div>',
        unsafe_allow_html=True
    )

    avg_score = (
        recommendations["Final Score"].mean()
        * 100
    )

    best_score = (
        recommendations["Final Score"].max()
        * 100
    )

    matching_jobs = len(recommendations)

    unique_companies = (
        recommendations["Company"]
        .nunique()
    )


    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "Jobs Matched",
            matching_jobs
        )

    with m2:

        st.metric(
            "Average Match",
            f"{avg_score:.1f}%"
        )

    with m3:

        st.metric(
            "Best Match",
            f"{best_score:.1f}%"
        )

    with m4:

        st.metric(
            "Companies",
            unique_companies
        )


    # ========================================================
    # TOP RECOMMENDATION
    # ========================================================

    best_job = recommendations.iloc[0]

    st.markdown(
        '<div class="section-title">🏆 Your Top Career Match</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        top_left, top_right = st.columns(
            [3, 1]
        )

        with top_left:

            st.markdown(
                f"""
                <div class="job-title">
                {best_job["Job Title"]}
                </div>

                <div class="company-name">
                {best_job["Company"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(
                f"📍 {best_job['Location']}   •   "
                f"💼 {best_job['Experience']}   •   "
                f"🏢 {best_job['Industry']}"
            )

        with top_right:

            st.markdown(
                '<div class="small-label">MATCH SCORE</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="score">'
                f'{best_job["Final Score"] * 100:.1f}%'
                f'</div>',
                unsafe_allow_html=True
            )

        st.progress(
            float(best_job["Final Score"])
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">✨ Recommended Opportunities</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Ranked according to your personalized recommendation score.'
        '</div>',
        unsafe_allow_html=True
    )


    for index, job in recommendations.iterrows():

        with st.container(border=True):

            left, right = st.columns(
                [4, 1]
            )

            with left:

                st.markdown(
                    f"""
                    <div class="job-title">
                    {index + 1}. {job["Job Title"]}
                    </div>

                    <div class="company-name">
                    {job["Company"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(
                    f"📍 {job['Location']}  •  "
                    f"💼 {job['Experience']}  •  "
                    f"🏢 {job['Industry']}  •  "
                    f"💰 {job['Salary']:,.0f}"
                )

            with right:

                st.markdown(
                    '<div class="small-label">MATCH</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="score">'
                    f'{job["Final Score"] * 100:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )


            st.progress(
                float(job["Final Score"])
            )


            # =================================================
            # SKILLS
            # =================================================

            skill_col1, skill_col2 = st.columns(2)

            with skill_col1:

                st.markdown(
                    '<div class="small-label">'
                    'MATCHING SKILLS'
                    '</div>',
                    unsafe_allow_html=True
                )

                if job["Matched Skills"]:

                    skill_chips(
                        job["Matched Skills"]
                    )

                else:

                    st.caption(
                        "No direct skill matches."
                    )


            with skill_col2:

                st.markdown(
                    '<div class="small-label">'
                    'SKILLS TO DEVELOP'
                    '</div>',
                    unsafe_allow_html=True
                )

                if job["Skills to Develop"]:

                    skill_chips(
                        job["Skills to Develop"],
                        missing=True
                    )

                else:

                    st.caption(
                        "Excellent! No major skill gaps."
                    )


            # =================================================
            # WHY THIS JOB?
            # =================================================

            with st.expander(
                "💡 Why is this job recommended?"
            ):

                reasons = []

                if job["Matched Skills"]:

                    reasons.append(
                        "Your skills have strong similarity with the job requirements."
                    )

                if job["Industry Match"] == 1:

                    reasons.append(
                        "The industry matches your preferred industry."
                    )

                if job["Experience Match"] == 1:

                    reasons.append(
                        "The experience level matches your profile."
                    )

                if job["Location Match"] == 1:

                    reasons.append(
                        "The location matches your preference."
                    )

                if job["Salary Score"] >= 1:

                    reasons.append(
                        "The salary meets your minimum preference."
                    )

                for reason in reasons:

                    st.write(
                        "✓ " + reason
                    )


            # =================================================
            # SCORE BREAKDOWN
            # =================================================

            with st.expander(
                "📊 View recommendation score breakdown"
            ):

                st.write(
                    "The final recommendation score combines:"
                )

                b1, b2, b3, b4, b5 = st.columns(5)

                with b1:

                    st.metric(
                        "Skills",
                        f"{job['Skill Similarity'] * 100:.1f}%"
                    )

                with b2:

                    st.metric(
                        "Industry",
                        f"{job['Industry Match'] * 100:.0f}%"
                    )

                with b3:

                    st.metric(
                        "Experience",
                        f"{job['Experience Match'] * 100:.0f}%"
                    )

                with b4:

                    st.metric(
                        "Location",
                        f"{job['Location Match'] * 100:.0f}%"
                    )

                with b5:

                    st.metric(
                        "Salary",
                        f"{job['Salary Score'] * 100:.0f}%"
                    )


    # ========================================================
    # CAREER GAP ANALYSIS
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🧠 Career Skill Gap Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'See which skills appear most frequently across your recommended opportunities.'
        '</div>',
        unsafe_allow_html=True
    )


    # Count missing skills across recommendations

    skill_frequency = {}

    for _, job in recommendations.iterrows():

        for skill in job["Skills to Develop"]:

            skill = skill.lower().strip()

            if skill:

                skill_frequency[skill] = (
                    skill_frequency.get(skill, 0) + 1
                )


    # Sort skills by frequency

    gap_df = pd.DataFrame(
        list(
            skill_frequency.items()
        ),
        columns=[
            "Skill",
            "Job Count"
        ]
    )


    if not gap_df.empty:

        gap_df = (
            gap_df
            .sort_values(
                "Job Count",
                ascending=False
            )
            .head(10)
        )


        gap_left, gap_right = st.columns(
            [1.4, 1]
        )


        with gap_left:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🎯 Skills Worth Developing"
            )

            for _, row in gap_df.iterrows():

                skill = row["Skill"]

                count = int(
                    row["Job Count"]
                )

                st.write(
                    f"**{skill.title()}**  — appears in "
                    f"**{count}** recommended job(s)"
                )

                st.progress(
                    count / len(recommendations)
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        with gap_right:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🚀 Career Insight"
            )

            top_gap = gap_df.iloc[0]["Skill"]

            top_gap_count = int(
                gap_df.iloc[0]["Job Count"]
            )

            st.markdown(
                f"""
                Your recommendations suggest that
                **{top_gap.title()}** could be a valuable
                skill to develop next.

                It appears in **{top_gap_count}**
                of your top recommended opportunities.
                """
            )

            st.write("")

            st.info(
                "Tip: Prioritize skills that appear repeatedly "
                "across multiple recommended roles."
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


    else:

        st.success(
            "🎉 Excellent! Your current skills already cover "
            "the requirements of your recommended jobs."
        )


    # ========================================================
    # HOW THE MODEL WORKS
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '⚙️ How AI Career Navigator Works'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A transparent recommendation pipeline built using similarity and preference matching.'
        '</div>',
        unsafe_allow_html=True
    )


    h1, h2, h3, h4 = st.columns(4)


    with h1:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown("### 01")
        st.markdown("**Profile Analysis**")
        st.caption(
            "Your skills, experience, industry, location and salary preference are collected."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with h2:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown("### 02")
        st.markdown("**Skill Similarity**")
        st.caption(
            "TF-IDF and cosine similarity compare your skills with job requirements."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with h3:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown("### 03")
        st.markdown("**Preference Matching**")
        st.caption(
            "Industry, experience, location and salary preferences contribute to the score."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with h4:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown("### 04")
        st.markdown("**Personalized Ranking**")
        st.caption(
            "Jobs are ranked using a weighted final recommendation score."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # FORMULA
    # ========================================================

    with st.expander(
        "🔍 View recommendation formula"
    ):

        st.markdown(
            """
            ### Final Recommendation Score

            **Final Score =**

            **0.50 × Skill Similarity**

            **+ 0.20 × Industry Match**

            **+ 0.15 × Experience Match**

            **+ 0.10 × Location Match**

            **+ 0.05 × Salary Score**

            The highest-scoring jobs are presented as the
            most relevant recommendations.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with <strong>Python</strong> •
        <strong>Streamlit</strong> •
        <strong>TF-IDF</strong> •
        <strong>Cosine Similarity</strong>
        <br><br>
        AI Career Navigator — Personalized Career Recommendation Prototype
    </div>
    """,
    unsafe_allow_html=True
)
