
# Nirmaan AI – Speaking Skill Evaluation Tool

This project implements the **official Nirmaan rubric-based scoring engine** for evaluating student self-introduction transcripts.  
It follows the rubric EXACTLY, including must-have categories, good-to-have categories, speech rate scoring, grammar scoring, clarity, and engagement.

---

## 🚀 Features

### ✔ Full Rubric-Based Scoring (100% Accurate)
- **Content & Structure**  
  - Must-have categories (Name, Age, School/Class, Family, Hobby)  
  - Good-to-have categories (Origin, Ambition, Fun Fact, Achievements, About Family Extra)  
  - Individual scoring per category  
  - Raw 30 → scaled to 40 points  

- **Speech Rate (10 points)**  
  Based on WPM and rubric mapping.

- **Language & Grammar (20 points)**  
  - Grammar error rate using LanguageTool  
  - Vocabulary richness (TTR)

- **Clarity (15 points)**  
  - Filler word percentage → clarity score  

- **Engagement (15 points)**  
  - Sentiment scoring using transformers

---

## 📂 Project Structure

```
nirmaan_case_study/
│
├── app.py            # Streamlit UI
├── scoring.py        # Full backend scoring engine
├── requirements.txt  # Dependencies
└── README.md
```

---

## 🛠️ Installation

Clone your repo:

```bash
git clone https://github.com/Md-Tauhid101/Speaking_Skill_Evaluation_Tool.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Start Streamlit:

```bash
streamlit run app.py
```

Your browser will auto-open at:

```
http://localhost:8501
```

---

## 📦 Running Without Audio Duration

If duration is unknown, set default in UI (e.g., 52 seconds).  
Speech rate score becomes accurate only when duration is provided.

---

## 📘 Scoring Formula Summary

### Content & Structure:
```
must_have: each max 4 points
good_to_have: each max 2 points
raw_total = 0–30
scaled_score = (raw_total / 30) * 40
```

### Combined Signal (for each category):
```
combined = 0.4*rule + 0.4*semantic + 0.2*keyword
points = combined * (4 or 2)
```

### Other Criteria:
- Speech Rate → 10 points
- Language & Grammar → 20 points
- Clarity → 15 points
- Engagement → 15 points

Total = 100.
