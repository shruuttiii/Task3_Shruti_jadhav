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
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PALETTE
# ============================================================

ALABASTER = "#EFE8DF"
MAROON = "#7F0303"
MIDNIGHT = "#0F414A"
LIGHT_BLUE = "#96C0CE"
TAN = "#D8BA98"

DARK_TEXT = "#243338"
MUTED_TEXT = "#667477"
WHITE = "#FFFFFF"


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    f"""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap'
);


/* ==========================================================
   GLOBAL
   ========================================================== */

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
}}

.stApp {{

    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(150, 192, 206, 0.25),
            transparent 28%
        ),

        radial-gradient(
            circle at 100% 10%,
            rgba(216, 186, 152, 0.22),
            transparent 25%
        ),

        {ALABASTER};

    color: {DARK_TEXT};
}}

.block-container {{
    max-width: 1220px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}}

header[data-testid="stHeader"] {{
    background: transparent;
}}


/* ==========================================================
   HERO
   ========================================================== */

.hero {{
    padding: 45px 20px 50px 20px;
    text-align: center;
}}

.hero-badge {{

    display: inline-block;

    padding: 8px 16px;

    border-radius: 999px;

    background: {LIGHT_BLUE};

    color: {MIDNIGHT};

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 1.5px;

    margin-bottom: 20px;
}}

.hero-title {{

    font-family: 'Playfair Display', serif;

    font-size: 60px;

    line-height: 1.05;

    font-weight: 700;

    color: {MIDNIGHT};

    margin: 0;
}}

.hero-title span {{
    color: {MAROON};
}}

.hero-subtitle {{

    max-width: 720px;

    margin: 20px auto 0 auto;

    color: {MUTED_TEXT};

    font-size: 16px;

    line-height: 1.75;
}}


/* ==========================================================
   SECTION TITLES
   ========================================================== */

.section-title {{

    font-family: 'Playfair Display', serif;

    font-size: 29px;

    font-weight: 700;

    color: {MIDNIGHT};

    margin-top: 25px;

    margin-bottom: 5px;
}}

.section-subtitle {{

    color: {MUTED_TEXT};

    font-size: 14px;

    margin-bottom: 20px;
}}


/* ==========================================================
   CONTAINERS / CARDS
   ========================================================== */

[data-testid="stVerticalBlockBorderWrapper"] {{

    background: rgba(255,255,255,0.75);

    border: 1px solid rgba(15,65,74,0.12) !important;

    border-radius: 20px !important;

    box-shadow:
        0 10px 30px rgba(15,65,74,0.07);

    transition: all 0.2s ease;
}}

[data-testid="stVerticalBlockBorderWrapper"]:hover {{

    box-shadow:
        0 15px 38px rgba(15,65,74,0.12);

    transform: translateY(-1px);
}}


/* ==========================================================
   INPUTS
   ========================================================== */

label {{

    color: {MIDNIGHT} !important;

    font-weight: 700 !important;
}}

.stTextInput input,
.stNumberInput input {{

    background: {WHITE} !important;

    border: 1px solid #d6d2cc !important;

    border-radius: 12px !important;

    color: {DARK_TEXT} !important;
}}

.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {{

    background: {WHITE} !important;

    border: 1px solid #d6d2cc !important;

    border-radius: 12px !important;

    color: {DARK_TEXT} !important;
}}


/* Multiselect tags */

.stMultiSelect span[data-baseweb="tag"] {{

    background: {LIGHT_BLUE} !important;

    border: none !important;

    color: {MIDNIGHT} !important;

    font-weight: 600;
}}


/* ==========================================================
   MAIN BUTTON
   ========================================================== */

.stButton > button {{

    width: 100%;

    border: none;

    border-radius: 13px;

    padding: 15px 22px;

    background:
        linear-gradient(
            100deg,
            {MAROON},
            #9d1b1b
        );

    color: white;

    font-size: 15px;

    font-weight: 800;

    box-shadow:
        0 12px 25px rgba(127,3,3,0.20);

    transition: all 0.25s ease;
}}

.stButton > button:hover {{

    background:
        linear-gradient(
            100deg,
            #650000,
            {MAROON}
        );

    transform: translateY(-2px);

    box-shadow:
        0 16px 30px rgba(127,3,3,0.28);
}}


/* ==========================================================
   SKILL CHIPS
   ========================================================== */

.skill-chip {{

    display: inline-block;

    padding: 7px 13px;

    margin: 4px 5px 4px 0;

    border-radius: 999px;

    background: {LIGHT_BLUE};

    color: {MIDNIGHT};

    font-size: 12px;

    font-weight: 700;
}}

.missing-chip {{

    display: inline-block;

    padding: 7px 13px;

    margin: 4px 5px 4px 0;

    border-radius: 999px;

    background: {TAN};

    color: {MAROON};

    font-size: 12px;

    font-weight: 700;
}}


/* ==========================================================
   JOB TITLES
   ========================================================== */

.job-title {{

    font-size: 20px;

    font-weight: 800;

    color: {MIDNIGHT};
}}

.company-name {{

    color: {MAROON};

    font-size: 14px;

    margin-top: 4px;

    font-weight: 700;
}}

.score {{

    font-size: 30px;

    font-weight: 800;

    color: {MAROON};
}}

.small-label {{

    color: {MUTED_TEXT};

    font-size: 10px;

    text-transform: uppercase;

    letter-spacing: 1.3px;

    font-weight: 800;
}}


/* ==========================================================
   METRICS
   ========================================================== */

[data-testid="stMetric"] {{

    background: rgba(255,255,255,0.78);

    border: 1px solid rgba(15,65,74,0.12);

    border-radius: 16px;

    padding: 18px;

    box-shadow:
        0 7px 22px rgba(15,65,74,0.06);
}}

[data-testid="stMetricLabel"] {{

    color: {MUTED_TEXT} !important;

    font-size: 12px !important;
}}

[data-testid="stMetricValue"] {{

    color: {MIDNIGHT} !important;

    font-weight: 800 !important;
}}


/* ==========================================================
   PROGRESS BAR
   ========================================================== */

.stProgress > div > div > div > div {{

    background:
        linear-gradient(
            90deg,
            {MAROON},
            {TAN},
            {LIGHT_BLUE}
        );
}}


/* ==========================================================
   EXPANDERS
   ========================================================== */

[data-testid="stExpander"] {{

    border-color: rgba(15,65,74,0.14) !important;

    background: rgba(255,255,255,0.45);
}}

.streamlit-expanderHeader {{

    color: {MIDNIGHT} !important;

    font-weight: 700;
}}


/* ==========================================================
   CAREER INSIGHT
   ========================================================== */

.insight-box {{

    background:
        linear-gradient(
            135deg,
            rgba(150,192,206,0.42),
            rgba(216,186,152,0.35)
        );

    border-left: 5px solid {MAROON};

    border-radius: 15px;

    padding: 20px;

    color: {MIDNIGHT};
}}

.insight-title {{

    color: {MAROON};

    font-size: 16px;

    font-weight: 800;
}}


/* ==========================================================
   FORMULA
   ========================================================== */

.formula-box {{

    background: {MIDNIGHT};

    color: #f8f4ef;

    border-radius: 18px;

    padding: 25px;

    line-height: 2;
}}

.formula-box strong {{
    color: {TAN};
}}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {{

    text-align: center;

    padding: 45px 0 10px 0;

    color: #8b9292;

    font-size: 12px;
}}

.footer strong {{
    color: {MIDNIGHT};
}}


/* ==========================================================
   DIVIDER
   ========================================================== */

hr {{
    border-color: rgba(15,65,74,0.13) !important;
}}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL FILES
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
# SKILL MATCHING
# ============================================================

def get_skill_match(
    user_skills,
    job_skills
):

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
# ============================================================
# IMPORTANT:
# THIS IS THE SAME MODEL LOGIC AS BEFORE.
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

    results["Skill Similarity"] = (
        skill_scores
    )

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

    # ========================================================
    # SAME FINAL SCORE
    # ========================================================

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
# SKILL CHIP DISPLAY
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

st.markdown(
    """
<div class="hero">

    <div class="hero-badge">
        ✦ AI-POWERED CAREER INTELLIGENCE
    </div>

    <div class="hero-title">
        Find work that <span>fits you.</span>
    </div>

    <div class="hero-subtitle">
        AI Career Navigator analyzes your skills,
        experience, industry interests, location and
        salary preferences to discover personalized
        career opportunities.
    </div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# PROFILE
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
    'Our recommendation engine will do the matching.'
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
                vectorizer
                .get_feature_names_out()
            ]
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
        'Your profile has been compared against '
        'the available opportunities.'
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
    # RECOMMENDED JOBS
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
                    f'{rank:02d} &nbsp; '
                    f'{job["Job Title"]}'
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


            # ------------------------------------------------
            # SKILLS
            # ------------------------------------------------

            skill_col1, skill_col2 = st.columns(2)


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
        'Skills that appear repeatedly across '
        'your recommended opportunities.'
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
                    "Prioritize skills that appear "
                    "repeatedly across multiple "
                    "target roles."
                )


    else:

        st.success(
            "🎉 Your current skills already cover "
            "the requirements of your recommended jobs."
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
        'A transparent recommendation pipeline '
        'combining skill similarity with your '
        'career preferences.'
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
                "personalized final score."
            )


    # ========================================================
    # FORMULA
    # ========================================================

    with st.expander(
        "See the recommendation formula"
    ):

        st.markdown(
            f"""
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
    f"""
    <div class="footer">

        <strong>AI Career Navigator</strong>

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
