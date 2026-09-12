# 💼 AI Career Navigator

### AI-Powered Personalized Job Recommendation System

AI Career Navigator is a personalized job recommendation system that helps users discover career opportunities based on their **skills, experience level, preferred industry, location, and salary preference**.

The system analyzes the user's profile and ranks job opportunities according to how well they match the user's preferences.

---

## 🚀 Live Demo

👉 [Launch AI Career Navigator](https://task3shrutijadhav-ejyooh9obevk8srfbflsc8.streamlit.app/)

Try the application directly through the deployed Streamlit app.

---

## 🎯 Project Objective

The objective of this project is to build a recommendation system that provides personalized career opportunities instead of simply displaying a list of jobs.

The system considers multiple user preferences and generates a ranked list of job recommendations.

It also provides:

- Personalized job recommendations
- Skill similarity analysis
- Matching skills
- Skills the user may need to develop
- Recommendation score
- Explanation of why a job was recommended

---

## 🧠 Recommendation Approach

The recommendation engine combines **text similarity and preference matching**.

### 1. Skill Similarity

The required skills for each job are converted into numerical representations using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The user's skills are then compared with job requirements using **Cosine Similarity**.

This determines how closely the user's skill profile matches each job.

### 2. Preference Matching

The system checks whether the job matches the user's:

- Industry
- Experience level
- Location
- Salary preference

### 3. Final Recommendation Score

The final score is calculated using weighted components:

**Final Score =**

- 50% × Skill Similarity
- 20% × Industry Match
- 15% × Experience Match
- 10% × Location Match
- 5% × Salary Score

Jobs with higher final scores are ranked higher in the recommendation list.

---

## 🔍 Example User Profile

A user can provide information such as:

**Skills**
- Python
- SQL
- Machine Learning

**Experience**
- Entry Level

**Industry**
- Software

**Location**
- Bangalore

**Minimum Salary Preference**
- 80,000

The system then analyzes the available job opportunities and returns the most relevant matches.

---

## ✨ Key Features

### 🎯 Personalized Recommendations
Jobs are ranked according to the user's individual preferences.

### 🧩 Skill Matching
The system identifies skills that match the requirements of recommended jobs.

### 📈 Skill Gap Analysis
Users can see skills they may need to develop for potential opportunities.

### 💡 Recommendation Explanation
The application explains why a particular job was recommended.

### 📊 Recommendation Score
Each opportunity receives a personalized match score.

### 🖥️ Interactive Web Application
The recommendation system is deployed using Streamlit and can be accessed through a web browser.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Streamlit
- TF-IDF
- Cosine Similarity

---

## 📂 Project Structure

```text
AI-Career-Navigator/
│
├── app.py
│   └── Streamlit application and recommendation engine
│
├── AI_Career_Navigator.ipynb
│   └── Model development and experimentation
│
├── job_recommendation_dataset.csv
│   └── Job recommendation dataset
│
├── job_data.pkl
│   └── Processed job data
│
├── tfidf_vectorizer.pkl
│   └── Trained TF-IDF vectorizer
│
├── job_skill_matrix.npz
│   └── TF-IDF job skill matrix
│
└── requirements.txt
    └── Python dependencies
```

---

## 👩‍💻 Project

**AI Career Navigator**

An AI-powered personalized job recommendation system built using Python, Scikit-learn, and Streamlit.
