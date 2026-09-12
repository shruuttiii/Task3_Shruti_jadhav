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
# COLOR PALETTE
# ============================================================

ALABASTER = "#EFE8DF"
MAROON = "#7F0303"
MIDNIGHT = "#0F414A"
LIGHT_BLUE = "#96C0CE"
TAN = "#D8BA98"

TEXT = "#26383B"
MUTED = "#667477"
WHITE = "#FFFFFF"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap'
);

* {{
    font-family: 'DM Sans', sans-serif;
}}

.stApp {{
    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(150, 192, 206, 0.28),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 10%,
            rgba(216, 186, 152, 0.28),
            transparent 25%
        ),
        {ALABASTER};
    color: {TEXT};
}}

.block-container {{
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}}

header[data-testid="stHeader"] {{
    background: transparent;
}}


/* ------------------------------------------------------------
   HERO
------------------------------------------------------------ */

.hero-container {{
    background:
        linear-gradient(
            135deg,
            {MIDNIGHT} 0%,
            #174E59 55%,
            #256675 100%
        );

    border-radius: 28px;

    padding: 55px 50px;

    margin-bottom: 35px;

    box-shadow:
        0 20px 45px rgba(15, 65, 74, 0.20);

    position: relative;

    overflow: hidden;
}}

.hero-container::after {{
    content: "";

    position: absolute;

    width: 250px;
    height: 250px;

    border-radius: 50%;

    background: rgba(216,186,152,0.16);

    right: -80px;
    top: -90px;
}}

.hero-container::before {{
    content: "";

    position: absolute;

    width: 160px;
    height: 160px;

    border-radius: 50%;

    background: rgba(150,192,206,0.13);

    left: -60px;
    bottom: -70px;
}}

.hero-content {{
    position: relative;
    z-index: 2;
}}

.hero-kicker {{
    color: {TAN};

    font-size: 12px;

    font-weight: 800;

    letter-spacing: 2px;

    margin-bottom: 15px;
}}

.hero-title {{
    color: white;

    font-family: 'Playfair Display', serif;

    font-size: 54px;

    line-height: 1.05;

    margin: 0;
}}

.hero-title-accent {{
    color: {TAN};
}}

.hero-description {{
    color: rgba(255,255,255,0.78);

    max-width: 680px;

    font-size: 15px;

    line-height: 1.7;

    margin-top: 18px;
}}


/* ------------------------------------------------------------
   SECTION HEADERS
------------------------------------------------------------ */

.section-header {{
    font-family: 'Playfair Display', serif;

    font-size: 30px;

    color: {MIDNIGHT};

    font-weight: 700;

    margin-top: 28px;

    margin-bottom: 5px;
}}

.section-description {{
    color: {MUTED};

    font-size: 14px;

    margin-bottom: 18px;
}}


/* ------------------------------------------------------------
   CONTAINERS
------------------------------------------------------------ */

[data-testid="stVerticalBlockBorderWrapper"] {{
    background: rgba(255,255,255,0.72);

    border: 1px solid rgba(15,65,74,0.13) !important;

    border-radius: 20px !important;

    box-shadow:
        0 8px 25px rgba(15,65,74,0.07);
}}


/* ------------------------------------------------------------
   INPUTS
------------------------------------------------------------ */

label {{
    color: {MIDNIGHT} !important;

    font-weight: 700 !important;
}}

.stTextInput input,
.stNumberInput input {{
    background: white !important;

    color: {TEXT} !important;

    border: 1px solid #d5d0ca !important;

    border-radius: 12px !important;
}}

.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {{
    background: white !important;

    border-radius: 12px !important;

    border: 1px solid #d5d0ca !important;
}}

.stMultiSelect span[data-baseweb="tag"] {{
    background: {LIGHT_BLUE} !important;

    color: {MIDNIGHT} !important;

    border: none !important;

    font-weight: 700;
}}


/* ------------------------------------------------------------
   BUTTON
------------------------------------------------------------ */

.stButton > button {{
    background:
        linear-gradient(
            100deg,
            {MAROON},
            #9B1D1D
        );

    color: white;

    border: none;

    border-radius: 13px;

    padding: 15px 20px;

    font-size: 15px;

    font-weight: 800;

    box-shadow:
        0 10px 25px rgba(127,3,3,0.22);

    transition: all 0.25s ease;
}}

.stButton > button:hover {{
    transform: translateY(-2px);

    box-shadow:
        0 15px 30px rgba(127,3,3,0.30);
}}


/* ------------------------------------------------------------
   METRICS
------------------------------------------------------------ */

[data-testid="stMetric"] {{
    background: rgba(255,255,255,0.72);

    border: 1px solid rgba(15,65,74,0.12);

    border-radius: 17px;

    padding: 18px;

    box-shadow:
        0 7px 20px rgba(15,65,74,0.06);
}}

[data-testid="stMetricLabel"] {{
    color: {MUTED} !important;
}}

[data-testid="stMetricValue"] {{
    color: {MIDNIGHT} !important;

    font-weight: 800 !important;
}}


/* ------------------------------------------------------------
   PROGRESS
------------------------------------------------------------ */

.stProgress > div > div > div > div {{
    background:
        linear-gradient(
            90deg,
            {MAROON},
            {TAN},
            {LIGHT_BLUE}
        );
}}


/* ------------------------------------------------------------
   JOB CARD
------------------------------------------------------------ */

.job-number {{
    color: {MAROON};

    font-size: 12px;

    font-weight: 800;

    letter-spacing: 1px;
}}

.job-title {{
    color: {MIDNIGHT};

    font-size: 21px;

    font-weight: 800;

    margin-top: 4px;
}}

.company {{
    color: {MAROON};

    font-size: 14px;

    font-weight: 700;
}}

.match-score {{
    color: {MAROON};

    font-size: 30px;

    font-weight: 800;

    text-align: right;
}}

.match-label {{
    color: {MUTED};

    font-size: 10px;

    font-weight: 800;

    letter-spacing: 1px;

    text-align: right;
}}


/* ------------------------------------------------------------
   CHIPS
------------------------------------------------------------ */

.skill-chip {{
    display: inline-block;

    background: {LIGHT_BLUE};

    color: {MIDNIGHT};

    border-radius: 999px;

    padding: 6px 12px;

    margin: 3px 4px 3px 0;

    font-size: 11px;

    font-weight: 700;
}}

.missing-chip {{
    display: inline-block;

    background: {TAN};

    color: {MAROON};

    border-radius: 999px;

    padding: 6px 12px;

    margin: 3px 4px 3px 0;

    font-size: 11px;

    font-weight: 700;
}}


/* ------------------------------------------------------------
   INSIGHT
------------------------------------------------------------ */

.insight {{
    background:
        linear-gradient(
            135deg,
            rgba(150,192,206,0.30),
            rgba(216,186,152,0.32)
        );

    border-left: 5px solid {MAROON};

    border-radius: 15px;

    padding: 20px;
}}


/* ------------------------------------------------------------
   FORMULA
------------------------------------------------------------ */

.formula {{
    background: {MIDNIGHT};

    color: white;

    border-radius: 18px;

    padding: 25px;

    line-height: 2;
}}

.formula strong {{
    color: {TAN};
}}


/* ------------------------------------------------------------
   FOOTER
------------------------------------------------------------ */

.footer {{
    text-align: center;

    color: #7C8585;

    font-size: 12px;

    padding-top: 45px;
}}

.footer strong {{
    color: {MIDNIGHT};
}}

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

    job_skill_matrix = load_npz(
        "job_skill_matrix.npz"
    )

    return data, vectorizer, job_skill_matrix


try:

    df_model, vectorizer, job_skill_matrix = load_model()

except Exception as e:

    st.error(
        "Could not load the recommendation model."
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
# RECOMMENDATION MODEL
# ============================================================
# EXACT SAME MODEL LOGIC
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
# CHIP FUNCTION
# ============================================================

def display_chips(
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

    html = ""

    for skill in skills:

        html += (
            f'<span class="{css_class}">'
            f'{skill.title()}'
            f'</span>'
        )

    # IMPORTANT:
    # No indentation before HTML.
    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
<div class="hero-container">
<div class="hero-content">

<div class="hero-kicker">
✦ AI-POWERED CAREER INTELLIGENCE
</div>

<div class="hero-title">
Find work that <span class="hero-title-accent">fits you.</span>
</div>

<div class="hero-description">
AI Career Navigator analyzes your skills, experience,
industry interests, location and salary preferences
to discover personalized career opportunities.
</div>

</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# PROFILE SECTION
# ============================================================

st.markdown(
    '<div class="section-header">Build your career profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Tell us what you are looking for and let the recommendation '
    'engine find your strongest matches.'
    '</div>',
    unsafe_allow_html=True
)


with st.container(border=True):

    left, right = st.columns(
        2,
        gap="large"
    )


    with left:

        skill_options = sorted(
            list(
                vectorizer
                .get_feature_names_out()
            )
        )

        default_skills = [
            skill
            for skill in [
                "python",
                "sql",
                "machine learning"
            ]
            if skill in skill_options
        ]

        skills_input = st.multiselect(
            "Your Skills",
            options=skill_options,
            default=default_skills,
            placeholder="Select your skills..."
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


    with right:

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
            "Number of Recommendations",
            min_value=3,
            max_value=15,
            value=10
        )


# ============================================================
# SELECTED SKILLS
# ============================================================

if skills_input:

    st.caption(
        "YOUR CURRENT SKILLS"
    )

    display_chips(
        skills_input
    )


st.write("")


# ============================================================
# BUTTON
# ============================================================

find_matches = st.button(
    "✦ Discover My Career Matches",
    use_container_width=True
)


# ============================================================
# RECOMMENDATIONS
# ============================================================

if find_matches:

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
    # RESULTS HEADER
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-header">'
        'Your career matches'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Opportunities ranked from highest to lowest '
        'personalized match.'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # METRICS
    # ========================================================

    avg_score = (
        recommendations[
            "Final Score"
        ].mean() * 100
    )

    best_score = (
        recommendations[
            "Final Score"
        ].max() * 100
    )

    job_count = len(
        recommendations
    )

    company_count = (
        recommendations[
            "Company"
        ].nunique()
    )


    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.metric(
            "Jobs matched",
            job_count
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
            company_count
        )


    # ========================================================
    # BEST MATCH
    # ========================================================

    st.markdown(
        '<div class="section-header">'
        '✦ Best match for you'
        '</div>',
        unsafe_allow_html=True
    )


    best_job = recommendations.iloc[0]


    with st.container(border=True):

        a, b = st.columns(
            [4, 1]
        )


        with a:

            st.markdown(
                f'<div class="job-title">'
                f'{best_job["Job Title"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="company">'
                f'{best_job["Company"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.write(
                f'📍 {best_job["Location"]}  '
                f'•  💼 {best_job["Experience"]}  '
                f'•  🏢 {best_job["Industry"]}'
            )


        with b:

            st.markdown(
                '<div class="match-label">'
                'MATCH SCORE'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="match-score">'
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
    # JOB LIST
    # ========================================================

    st.markdown(
        '<div class="section-header">'
        'Recommended opportunities'
        '</div>',
        unsafe_allow_html=True
    )


    for rank, (_, job) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        with st.container(border=True):

            left, right = st.columns(
                [4, 1]
            )


            with left:

                st.markdown(
                    f'<div class="job-number">'
                    f'OPPORTUNITY {rank:02d}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="job-title">'
                    f'{job["Job Title"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="company">'
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
                    '<div class="match-label">'
                    'MATCH'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="match-score">'
                    f'{job["Final Score"] * 100:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )


            st.progress(
                float(
                    job["Final Score"]
                )
            )


            # =================================================
            # SKILLS
            # =================================================

            skill1, skill2 = st.columns(2)


            with skill1:

                st.caption(
                    "SKILLS YOU ALREADY HAVE"
                )

                display_chips(
                    job["Matched Skills"]
                )


            with skill2:

                st.caption(
                    "SKILLS TO DEVELOP"
                )

                display_chips(
                    job["Skills to Develop"],
                    missing=True
                )


            # =================================================
            # WHY THIS JOB
            # =================================================

            with st.expander(
                "Why was this recommended?"
            ):

                if job["Matched Skills"]:

                    st.write(
                        "✓ Your skills have similarity "
                        "with the job requirements."
                    )

                if job["Industry Match"] == 1:

                    st.write(
                        "✓ Industry matches your preference."
                    )

                if job["Experience Match"] == 1:

                    st.write(
                        "✓ Experience level matches."
                    )

                if job["Location Match"] == 1:

                    st.write(
                        "✓ Location matches your preference."
                    )

                if job["Salary Score"] >= 1:

                    st.write(
                        "✓ Salary meets your minimum preference."
                    )


            # =================================================
            # SCORE BREAKDOWN
            # =================================================

            with st.expander(
                "View recommendation score"
            ):

                c1, c2, c3, c4, c5 = st.columns(5)


                with c1:

                    st.metric(
                        "Skills",
                        f'{job["Skill Similarity"] * 100:.1f}%'
                    )


                with c2:

                    st.metric(
                        "Industry",
                        f'{job["Industry Match"] * 100:.0f}%'
                    )


                with c3:

                    st.metric(
                        "Experience",
                        f'{job["Experience Match"] * 100:.0f}%'
                    )


                with c4:

                    st.metric(
                        "Location",
                        f'{job["Location Match"] * 100:.0f}%'
                    )


                with c5:

                    st.metric(
                        "Salary",
                        f'{job["Salary Score"] * 100:.0f}%'
                    )


    # ========================================================
    # SKILL GAP
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-header">'
        '🧠 Your career skill gap'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Skills that appear frequently across your '
        'recommended opportunities.'
        '</div>',
        unsafe_allow_html=True
    )


    skill_frequency = {}


    for _, job in recommendations.iterrows():

        for skill in job[
            "Skills to Develop"
        ]:

            skill = skill.lower().strip()

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

            with st.container(border=True):

                st.subheader(
                    "Skills worth developing"
                )


                for _, row in gap_df.iterrows():

                    skill = row["Skill"]

                    count = int(
                        row["Job Count"]
                    )

                    st.write(
                        f"**{skill.title()}** "
                        f"· appears in {count} "
                        f"recommended jobs"
                    )

                    st.progress(
                        count /
                        len(recommendations)
                    )


        with gap_right:

            with st.container(border=True):

                st.subheader(
                    "✦ Career insight"
                )


                top_skill = (
                    gap_df.iloc[0]["Skill"]
                )

                top_count = int(
                    gap_df.iloc[0]["Job Count"]
                )


                st.markdown(
                    f"""
<div class="insight">

<b>Priority skill</b>

<br><br>

<span style="font-size:24px;">
{top_skill.title()}
</span>

<br><br>

This skill appears in
<b>{top_count}</b>
of your recommended opportunities.

</div>
""",
                    unsafe_allow_html=True
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
        '<div class="section-header">'
        'How AI Career Navigator works'
        '</div>',
        unsafe_allow_html=True
    )


    h1, h2, h3, h4 = st.columns(4)


    with h1:

        with st.container(border=True):

            st.markdown("### 01")

            st.markdown(
                "**Build profile**"
            )

            st.caption(
                "Your skills, experience, "
                "industry, location and "
                "salary preferences."
            )


    with h2:

        with st.container(border=True):

            st.markdown("### 02")

            st.markdown(
                "**Analyze skills**"
            )

            st.caption(
                "TF-IDF and cosine similarity "
                "compare your skills with "
                "job requirements."
            )


    with h3:

        with st.container(border=True):

            st.markdown("### 03")

            st.markdown(
                "**Match preferences**"
            )

            st.caption(
                "Industry, experience, "
                "location and salary "
                "contribute to the score."
            )


    with h4:

        with st.container(border=True):

            st.markdown("### 04")

            st.markdown(
                "**Rank jobs**"
            )

            st.caption(
                "Opportunities are ranked "
                "using the final personalized score."
            )


    # ========================================================
    # FORMULA
    # ========================================================

    with st.expander(
        "See the recommendation formula"
    ):

        st.markdown(
            """
<div class="formula">

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

<strong>AI Career Navigator</strong>

<br><br>

Personalized career recommendation prototype

<br><br>

Built with Python · Streamlit · TF-IDF · Cosine Similarity

</div>
""",
    unsafe_allow_html=True
)
