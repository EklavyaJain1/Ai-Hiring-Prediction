# AI Hiring Prediction

An end-to-end machine learning project that predicts whether a candidate will be Hired or Rejected from resume attributes, built and compared across four classic classifiers.

# Overview

Given a candidate's profile (experience, education, skills, certifications, projects, salary expectation, AI screening score), the notebook trains and evaluates models to reproduce the recruiter's hire/reject decision. It also analyses which features drive the decision, and flags a data leakage issue around the AI Score column

# 📂 Repository Structure

Ai-Hiring-Prediction<br>
├── ai_hiring_prediction.ipynb  &emsp;#Full pipeline: EDA → features → models → tuning → importance


├── hiring_data.csv   &emsp;#Dataset (1,000 resumes, 11 columns)


└── README.md

#Dataset

hiring_data.csv has 1,000 candidate records.

<table>
  <tr>
    <th align="center">Column</th>
    <th align="center">Description</th>
  </tr>
   <tr><td><code>Resume_ID</code></td><td>Unique candidate ID</td></tr>
    <tr><td><code>Name</code></td><td>Candidate name</td></tr>
    <tr><td><code>Skills</code></td><td>Comma-separated skill list (e.g. <code>Python, SQL, NLP</code>)</td></tr>
    <tr><td><code>Experience (Years)</code></td><td>Years of experience (0–10)</td></tr>
    <tr><td><code>Education</code></td><td>B.Sc, B.Tech, M.Tech, MBA, PhD</td></tr>
    <tr><td><code>Certifications</code></td><td>Deep Learning Specialization, AWS Certified, Google ML (274 missing = no certification)</td></tr>
    <tr><td><code>Job Role</code></td><td>AI Researcher, Data Scientist, Cybersecurity Analyst, Software Engineer</td></tr>
    <tr><td><code>Recruiter Decision</code></td><td><strong>Target</strong>: <code>Hire</code> (812) / <code>Reject</code> (188)</td></tr>
    <tr><td><code>Salary Expectation ($)</code></td><td>Expected salary (~$40K–$120K)</td></tr>
    <tr><td><code>Projects Count</code></td><td>Number of projects (0–10)</td></tr>
    <tr><td><code>AI Score (0-100)</code></td><td>Automated resume-screening score</td></tr>
</table>


# Pipeline


1.Load & explore — head, info, describe<br>


2.Data quality — missing-value check, class balance<br>


3.Feature engineering<br>

4.Skills → 

&emsp;a.<mark>Skills_Count</mark>

&emsp;b.<mark>Certifications (missing) → Has_Certification (0/1)</mark>

&emsp;c.<mark>Education and Job Role → one-hot encoded</mark>

&emsp;d.<mark>Recruiter Decision → Target (Hire = 1, Reject = 0)</mark>

5.Train/test split — 80/20, random_state=42, stratified<br>

6.Baseline — Logistic Regression (with StandardScaler)<br>

7.Model comparison — Logistic Regression, KNN, Decision Tree, Random Forest<br>

8.Hyperparameter tuning — GridSearchCV (5-fold, F1) on the best model<br>

9.Feature importance analysis<br>

# Results

Evaluated on the 200-row held-out test set:
<img width="1883" height="835" alt="Results_ Model Performance Table" src="https://github.com/user-attachments/assets/e15e8baa-913b-4f29-929e-51b28c0adfb8" />

## Feature importance (Random Forest)

<table>
  <tr>
    <th align="center">Feature</th>
    <th align="center">Importance</th>
  </tr>
   <tr><td><code>AI Score (0-100)</code></td><td>	0.637</td></tr>
    <tr><td><code>Experience (Years)</code></td><td>	0.240</td></tr>
    <tr><td><code>Projects Count</code></td><td>	0.080</td></tr>
    <tr><td><code>Salary Expectation</code></td><td>0.019</td></tr>
    <tr><td><code>Skills_Count</code></td><td>	0.006</td></tr>
    <tr><td><code>Has_Certification</code></td><td>0.005</td></tr>

</table>

## Key Findings

The near-perfect scores are a sign of leakage, not a great model.

AI Score has a 0.84 correlation with the hiring decision, and the two classes barely overlap on it: rejected candidates have scores of at most 60, while hired candidates average ~92. The model is mostly learning a score threshold.

AI Score is itself a pre-computed screening output, so using it as a feature largely reproduces the recruiter's decision.

Recommended next step: drop AI Score and retrain, to see how well the model predicts from raw resume attributes alone (experience, projects, skills, etc.). Expect metrics to drop to more realistic levels.

After AI Score, experience and project count are the strongest signals.

## Getting Started

Requirements: Python 3.9+

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

Run locally
git clone https://github.com/EklavyaJain1/Ai-Hiring-Prediction.git

cd Ai-Hiring-Prediction

jupyter notebook ai_hiring_prediction.ipynb

<mark>Run on Google Colab: upload hiring_data.csv to the session (or mount Drive) and update the path in the first data-loading cell.</mark>

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter
