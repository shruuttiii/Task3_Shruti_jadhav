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
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PREMIUM WARM EDITORIAL THEME
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');


/* =========================================================
   GLOBAL
   ========================================================= */

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(196, 181, 253, 0.22),
            transparent 24%
        ),
        radial-gradient(
            circle at 95% 12%,
            rgba(251, 146, 60, 0.12),
            transparent 25%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(244, 114, 182, 0.10),
            transparent 30%
        ),
        #f7f4ef;

    color: #29252d;
}

.block-container {
    max-width: 1240px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    position: relative;
    padding: 48px 25px 55px 25px;
    text-align: center;
}

.hero-badge {
    display: inline-block;

    padding: 8px 16px;

    border-radius: 999px;

    background: #eee7ff;

    border: 1px solid #ddd0ff;

    color: #6841b7;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 1.6px;

    margin-bottom: 18px;
}

.hero-title {
    font-family: 'Playfair Display', serif;

    font-size: 58px;

    line-height: 1.05;

    font-weight: 700;

    margin: 0;

    color: #2d2633;
}

.hero-title span {
    color: #7950c7;
}

.hero-subtitle {
    max-width: 720px;

    margin: 18px auto 0 auto;

    color: #756e78;

    font-size: 16px;

    line-height: 1.75;
}


/* =========================================================
   SECTION HEADERS
   ========================================================= */

.section-title {
    font-family: 'Playfair Display', serif;

    font-size: 28px;

    font-weight: 700;

    color: #302934;

    margin-top: 25px;

    margin-bottom: 5px;
}

.section-subtitle {
    color: #817984;

    font-size: 14px;

    margin-bottom: 20px;
}


/* =========================================================
   INPUT AREA
   ========================================================= */

.profile-panel {
    background: rgba(255,255,255,0.78);

    border: 1px solid #e8e1dc;

    border-radius: 24px;

    padding: 28px;

    box-shadow:
        0 15px 45px rgba(65, 50, 70, 0.08);
}


/* Input labels */

label {
    color: #4a414d !important;

    font-weight: 650 !important;
}


/* Text input */

.stTextInput input,
.stNumberInput input {

    background: #fffdfb !important;

    border: 1px solid #ddd5d0 !important;

    border-radius: 12px !important;

    color: #302934 !important;
}


/* Select boxes */

.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {

    background: #fffdfb !important;

    border: 1px solid #ddd5d0 !important;

    border-radius: 12px !important;

    color: #302934 !important;
}


/* Multiselect chips */

.stMultiSelect span[data-baseweb="tag"] {

    background: #eee7ff !important;

    border: 1px solid #d8c9ff !important;

    color: #6240a9 !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {

    width: 100%;

    border: none;

    border-radius: 13px;

    padding: 15px 22px;

    background:
        linear-gradient(
            100deg,
            #7045b8,
            #9568d3,
            #e57c67
        );

    color: white;

    font-size: 15px;

    font-weight: 700;

    box-shadow:
        0 12px 25px rgba(112,69,184,0.20);

    transition: all 0.25s ease;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 16px 32px rgba(112,69,184,0.28);
}


/* =========================================================
   SKILL CHIPS
   ========================================================= */

.skill-chip {

    display: inline-block;

    padding: 7px 13px;

    margin: 4px 5px 4px 0;

    border-radius: 999px;

    background: #eee7ff;

    border: 1px solid #d8c9ff;

    color: #6742aa;

    font-size: 12px;

    font-weight: 650;
}

.missing-chip {

    display: inline-block;

    padding: 7px 13px;

    margin: 4px 5px 4px 0;

    border-radius: 999px;

    background: #fff0e7;

    border: 1px solid #ffd4bd;

    color: #bd633d;

    font-size: 12px;

    font-weight: 650;
}


/* =========================================================
   METRICS
   ========================================================= */

[data-testid="stMetric"] {

    background: rgba(255,255,255,0.82);

    border: 1px solid #e7dfda;

    border-radius: 16px;

    padding: 18px;

    box-shadow:
        0 8px 25px rgba(60,45,55,0.06);
}

[data-testid="stMetricLabel"] {

    color: #847a85 !important;

    font-size: 12px !important;
}

[data-testid="stMetricValue"] {

    color: #342c38 !important;

    font-weight: 750 !important;
}


/* =========================================================
   CARDS
   ========================================================= */

[data-testid="stVerticalBlockBorderWrapper"] {

    background: rgba(255,255,255,0.82);

    border: 1px solid #e7dfda !important;

    border-radius: 20px !important;

    box-shadow:
        0 10px 32px rgba(65,50,70,0.06);

    transition: all 0.2s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {

    box-shadow:
        0 15px 40px rgba(65,50,70,0.10);
}


/* =========================================================
   JOB TITLE
   ========================================================= */

.job-title {

    font-size: 20px;

    font-weight: 750;

    color: #342b38;
}

.company-name {

    color: #7950c7;

    font-size: 14px;

    margin-top: 4px;

    font-weight: 600;
}

.score {

    font-size: 30px;

    font-weight: 800;

    color: #7045b8;
}

.small-label {

    color: #938895;

    font-size: 10px;

    text-transform: uppercase;

    letter-spacing: 1.3px;

    font-weight: 750;
}


/* =========================================================
   PROGRESS
   ========================================================= */

.stProgress > div > div > div > div {

    background:
        linear-gradient(
            90deg,
            #7650c4,
            #c17ed8,
            #e98570
        );
}


/* =========================================================
   EXPANDERS
   ========================================================= */

.streamlit-expanderHeader {

    color: #4c424e !important;

    font-weight: 650;
}

[data-testid="stExpander"] {

    border-color: #e7dfda !important;

    background: rgba(255,255,255,0.55);
}


/* =========================================================
   INSIGHT BOX
   ========================================================= */

.insight-box {

    background:
        linear-gradient(
            135deg,
            #f0eaff,
            #fff1eb
        );

    border: 1px solid #dfd2f5;

    border-radius: 18px;

    padding: 22px;

    margin-top: 8px;
}

.insight-title {

    color: #57358d;

    font-weight: 750;

    font-size: 16px;
}


/* =========================================================
   FORMULA
   ========================================================= */

.formula-box {

    background: #2f2734;

    color: #f9f5ff;

    border-radius: 18px;

    padding: 25px;

    margin-top: 10px;

    line-height: 2;
}

.formula-box strong {

    color: #d9c5ff;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    text-align: center;

    padding: 45px 0 10px 0;

    color: #9a919b;

    font-size: 12px;
}

.footer strong {

    color: #6e6570;
}


/* =========================================================
   DIVIDER
   ========================================================= */

hr {

    border-color: #e7dfda !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("job_data.pkl", "rb") as file:
        data = pickle.load(file)

    with open("tfidf_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    job_skill_matrix = load_npz(
        "job_skill_matrix.npz"
    )

    return data, vectorizer, job_skill_matrix


try:

    df_model, vectorizer, job_skill_matrix = load_model()

except Exception as e:

    st.error(
        "The recommendation model files could not be loaded."
    )

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

    matched = user_skills.intersection(
        job_skills
    )

    missing = job_skills.difference(
        user_skills
    )

    return list(matched), list(missing)


# ============================================================
# RECOMMENDATION ENGINE
# SAME MODEL / SAME WEIGHTS
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

    user_skill_text = " ".join(
        user_skills
    )

    user_vector = vectorizer.transform(
        [user_skill_text]
    )

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

    # EXACT SAME RECOMMENDATION FORMULA

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

            "Job Title":
                job["Job Title"],

            "Company":
                job["Company"],

            "Location":
                job["Location"],

            "Experience":
                job["Experience Level"],

            "Salary":
                job["Salary"],

            "Industry":
                job["Industry"],

            "Final Score":
                round(
                    job["Final Score"],
                    3
                ),

            "Skill Similarity":
                job["Skill Similarity"],

            "Industry Match":
                job["Industry Match"],

            "Experience Match":
                job["Experience Match"],

            "Location Match":
                job["Location Match"],

            "Salary Score":
                job["Salary Score"],

            "Matched Skills":
                matched,

            "Skills to Develop":
                missing

        })

    return pd.DataFrame(
        recommendations
    )


# ============================================================
# SKILL CHIP FUNCTION
# ============================================================

def skill_chips(
    skills,
    missing=False
):

    if not skills:

        st.caption(
            "None identified."
        )

        return

    css_class = (
        "missing-chip"
        if missing
        else "skill-chip"
    )

    chips = ""

    for skill in skills:

        chips += (
            f'<span class="{css_class}">'
            f'{skill.title()}'
            f'</span>'
        )

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
Find work that <span>fits you.</span>
</div>

<div class="hero-subtitle">
AI Career Navigator analyzes your skills, experience,
industry interests, location and salary preferences
to discover personalized career opportunities.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PROFILE SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Build your career profile'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Tell us what you are looking for. '
    'We will handle the matching.'
    '</div>',
    unsafe_allow_html=True
)


with st.container(
    border=True
):

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        skills_input = st.multiselect(
            "Your Skills",
            options=sorted(
                list(
                    vectorizer
                    .get_feature_names_out()
                )
            ),
            default=[
                skill
                for skill in [
                    "python",
                    "sql",
                    "machine learning"
                ]
                if skill in
                vectorizer.get_feature_names_out()
            ],
            help="Select the skills you already have."
        )

        experience = st.selectbox(
            "Experience Level",
            sorted(
                df_model[
                    "Experience Level"
                ]
                .dropna()
                .unique()
            )
        )

        industry = st.selectbox(
            "Preferred Industry",
            sorted(
                df_model[
                    "Industry"
                ]
                .dropna()
                .unique()
            )
        )

    with col2:

        location = st.selectbox(
            "Preferred Location",
            sorted(
                df_model[
                    "Location"
                ]
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
            "Recommendations to show",
            min_value=3,
            max_value=15,
            value=10
        )


# ============================================================
# CURRENT SKILLS
# ============================================================

if skills_input:

    st.markdown(
        '<div class="small-label">'
        'YOUR CURRENT SKILLS'
        '</div>',
        unsafe_allow_html=True
    )

    skill_chips(
        skills_input
    )


st.write("")


# ============================================================
# RECOMMENDATION BUTTON
# ============================================================

if st.button(
    "✦ Discover My Career Matches",
    use_container_width=True
):

    if not skills_input:

        st.warning(
            "Please select at least one skill."
        )

        st.stop()


    with st.spinner(
        "Analyzing your career profile..."
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
            "No recommendations were found."
        )

        st.stop()


    # ========================================================
    # OVERVIEW
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        'Your career matches'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Your profile has been compared against the available opportunities.'
        '</div>',
        unsafe_allow_html=True
    )


    avg_score = (
        recommendations[
            "Final Score"
        ].mean()
        * 100
    )

    best_score = (
        recommendations[
            "Final Score"
        ].max()
        * 100
    )

    matching_jobs = len(
        recommendations
    )

    unique_companies = (
        recommendations[
            "Company"
        ].nunique()
    )


    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "Jobs matched",
            matching_jobs
        )

    with m2:

        st.metric(
            "Average match",
            f"{avg_score:.1f}%"
        )

    with m3:

        st.metric(
            "Best match",
            f"{best_score:.1f}%"
        )

    with m4:

        st.metric(
            "Companies",
            unique_companies
        )


    # ========================================================
    # TOP MATCH
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '✦ Best match for you'
        '</div>',
        unsafe_allow_html=True
    )


    best_job = recommendations.iloc[0]


    with st.container(
        border=True
    ):

        top_left, top_right = st.columns(
            [4, 1]
        )

        with top_left:

            st.markdown(
                f'<div class="job-title">'
                f'{best_job["Job Title"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="company-name">'
                f'{best_job["Company"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.write(
                f'📍 {best_job["Location"]}   '
                f'•   💼 {best_job["Experience"]}   '
                f'•   🏢 {best_job["Industry"]}'
            )

        with top_right:

            st.markdown(
                '<div class="small-label">'
                'MATCH SCORE'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="score">'
                f'{best_job["Final Score"] * 100:.1f}%'
                f'</div>',
                unsafe_allow_html=True
            )

        st.progress(
            float(
                best_job["Final Score"]
            )
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Recommended opportunities'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Ranked from highest to lowest personalized match.'
        '</div>',
        unsafe_allow_html=True
    )


    for rank, (_, job) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        with st.container(
            border=True
        ):

            left, right = st.columns(
                [4, 1]
            )

            with left:

                st.markdown(
                    f'<div class="job-title">'
                    f'{rank:02d} &nbsp; {job["Job Title"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="company-name">'
                    f'{job["Company"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    f'📍 {job["Location"]}  '
                    f'•  💼 {job["Experience"]}  '
                    f'•  🏢 {job["Industry"]}  '
                    f'•  💰 {job["Salary"]:,.0f}'
                )

            with right:

                st.markdown(
                    '<div class="small-label">'
                    'MATCH'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="score">'
                    f'{job["Final Score"] * 100:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )


            st.progress(
                float(
                    job["Final Score"]
                )
            )


            skill_col1, skill_col2 = st.columns(
                2
            )


            # ------------------------------------------------
            # MATCHING SKILLS
            # ------------------------------------------------

            with skill_col1:

                st.markdown(
                    '<div class="small-label">'
                    'SKILLS YOU ALREADY HAVE'
                    '</div>',
                    unsafe_allow_html=True
                )

                skill_chips(
                    job["Matched Skills"]
                )


            # ------------------------------------------------
            # SKILL GAPS
            # ------------------------------------------------

            with skill_col2:

                st.markdown(
                    '<div class="small-label">'
                    'SKILLS TO DEVELOP'
                    '</div>',
                    unsafe_allow_html=True
                )

                skill_chips(
                    job["Skills to Develop"],
                    missing=True
                )


            # ------------------------------------------------
            # WHY THIS JOB
            # ------------------------------------------------

            with st.expander(
                "Why was this recommended?"
            ):

                reasons = []


                if job["Matched Skills"]:

                    reasons.append(
                        "Your current skills have "
                        "similarity with the job requirements."
                    )


                if job["Industry Match"] == 1:

                    reasons.append(
                        "The industry matches "
                        "your preference."
                    )


                if job["Experience Match"] == 1:

                    reasons.append(
                        "The experience level matches "
                        "your profile."
                    )


                if job["Location Match"] == 1:

                    reasons.append(
                        "The location matches "
                        "your preference."
                    )


                if job["Salary Score"] >= 1:

                    reasons.append(
                        "The salary meets "
                        "your minimum preference."
                    )


                for reason in reasons:

                    st.write(
                        "✓ " + reason
                    )


            # ------------------------------------------------
            # SCORE BREAKDOWN
            # ------------------------------------------------

            with st.expander(
                "View score breakdown"
            ):

                st.caption(
                    "How this recommendation score was calculated."
                )

                b1, b2, b3, b4, b5 = st.columns(5)

                with b1:

                    st.metric(
                        "Skills",
                        f'{job["Skill Similarity"] * 100:.1f}%'
                    )

                with b2:

                    st.metric(
                        "Industry",
                        f'{job["Industry Match"] * 100:.0f}%'
                    )

                with b3:

                    st.metric(
                        "Experience",
                        f'{job["Experience Match"] * 100:.0f}%'
                    )

                with b4:

                    st.metric(
                        "Location",
                        f'{job["Location Match"] * 100:.0f}%'
                    )

                with b5:

                    st.metric(
                        "Salary",
                        f'{job["Salary Score"] * 100:.0f}%'
                    )


    # ========================================================
    # CAREER GAP ANALYSIS
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🧠 Your career skill gap'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Skills that appear repeatedly across your recommended opportunities.'
        '</div>',
        unsafe_allow_html=True
    )


    skill_frequency = {}


    for _, job in recommendations.iterrows():

        for skill in job[
            "Skills to Develop"
        ]:

            skill = (
                skill
                .lower()
                .strip()
            )

            if skill:

                skill_frequency[skill] = (
                    skill_frequency.get(
                        skill,
                        0
                    ) + 1
                )


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
            [1.5, 1]
        )


        with gap_left:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### Skills worth developing"
                )

                for _, row in gap_df.iterrows():

                    skill = row["Skill"]

                    count = int(
                        row["Job Count"]
                    )

                    st.write(
                        f"**{skill.title()}** "
                        f"· {count} of "
                        f"{len(recommendations)} "
                        f"recommended jobs"
                    )

                    st.progress(
                        count /
                        len(recommendations)
                    )


        with gap_right:

            with st.container(
                border=True
            ):

                st.markdown(
                    "### ✦ Career insight"
                )

                top_gap = gap_df.iloc[0]["Skill"]

                top_gap_count = int(
                    gap_df.iloc[0]["Job Count"]
                )

                st.markdown(
                    f"""
                    <div class="insight-box">

                    <div class="insight-title">
                    Your next high-value skill
                    </div>

                    <br>

                    <b>{top_gap.title()}</b>

                    <br><br>

                    This skill appears in
                    <b>{top_gap_count}</b>
                    of your recommended opportunities.

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                st.caption(
                    "Prioritizing frequently requested "
                    "skills can help you prepare for "
                    "more of your target roles."
                )


    else:

        st.success(
            "🎉 Your current skills already cover "
            "the requirements of your recommended opportunities."
        )


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        'How your recommendation works'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A transparent recommendation pipeline combining '
        'skill similarity with your career preferences.'
        '</div>',
        unsafe_allow_html=True
    )


    h1, h2, h3, h4 = st.columns(4)


    with h1:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 01"
            )

            st.markdown(
                "**Understand your profile**"
            )

            st.caption(
                "Your skills, experience, "
                "industry, location and "
                "salary preference are collected."
            )


    with h2:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 02"
            )

            st.markdown(
                "**Compare your skills**"
            )

            st.caption(
                "TF-IDF and cosine similarity "
                "compare your skills with "
                "job requirements."
            )


    with h3:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 03"
            )

            st.markdown(
                "**Match preferences**"
            )

            st.caption(
                "Industry, experience, location "
                "and salary preferences "
                "contribute to the score."
            )


    with h4:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 04"
            )

            st.markdown(
                "**Rank opportunities**"
            )

            st.caption(
                "Jobs are ranked using the "
                "personalized final recommendation score."
            )


    # ========================================================
    # FORMULA
    # ========================================================

    with st.expander(
        "See the recommendation formula"
    ):

        st.markdown(
            """
            <div class="formula-box">

            <strong>Final Recommendation Score</strong>

            <br><br>

            0.50 × Skill Similarity

            <br>

            + 0.20 × Industry Match

            <br>

            + 0.15 × Experience Match

            <br>

            + 0.10 × Location Match

            <br>

            + 0.05 × Salary Score

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    AI Career Navigator

    <br>

    Personalized career recommendation prototype

    <br><br>

    Built with
    <strong>Python</strong> ·
    <strong>Streamlit</strong> ·
    <strong>TF-IDF</strong> ·
    <strong>Cosine Similarity</strong>

    </div>
    """,
    unsafe_allow_html=True
)
