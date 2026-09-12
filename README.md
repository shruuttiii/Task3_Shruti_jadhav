# 💼 AI Career Navigator

### AI-Powered Personalized Career Recommendation System

AI Career Navigator is a personalized career recommendation system that helps users discover relevant job opportunities based on their **skills, experience level, preferred industry, location, and salary expectations**.

The system uses **TF-IDF, Cosine Similarity, and weighted preference matching** to rank job opportunities according to how well they match the user's profile.

---

## 🚀 Why AI Career Navigator?

Finding the right job is not only about searching for a job title.

A suitable opportunity depends on multiple factors:

- What skills does the candidate have?
- What industry are they interested in?
- What experience level do they have?
- Where do they want to work?
- What salary are they looking for?

AI Career Navigator combines these factors into a personalized recommendation score instead of relying only on simple keyword filtering.

---

## ✨ Key Features

### 🤖 Personalized Recommendations
Generates job recommendations based on an individual's career profile.

### 🧠 Skill-Based Matching
Uses **TF-IDF Vectorization and Cosine Similarity** to compare user skills with job-required skills.

### 🎯 Multi-Factor Ranking
Combines skill similarity with industry, experience, location, and salary preferences.

### 📚 Skill Gap Analysis
Shows which skills match a recommended job and which skills the user may need to develop.

### 📊 Match Score
Each recommendation receives a compatibility score to help users compare opportunities.

### 🖥️ Interactive Web Application
Built with Streamlit so users can interact with the recommendation engine through a simple web interface.

---

# 🧠 Recommendation Engine

The system calculates a final recommendation score using the following weighted formula:

```text
Final Score =
0.50 × Skill Similarity
+ 0.20 × Industry Match
+ 0.15 × Experience Match
+ 0.10 × Location Match
+ 0.05 × Salary Score
