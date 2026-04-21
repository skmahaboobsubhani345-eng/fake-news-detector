# 🔍 Automated Fake News Detection System

> A machine learning powered web application that detects whether a news article is **Real**, **Fake**, or **Uncertain** — in under 2 seconds.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?style=flat&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-96.07%25-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 🌐 Live Demo

🚀 **[Try it here → https://fake-news-detector.onrender.com](https://fake-news-detector.onrender.com)**

> ⏳ First load may take 30 seconds (free server wakes up from sleep)

---

## 📸 What It Looks Like

```
┌─────────────────────────────────────────────┐
│   🔍 Automated Fake News Detection          │
│                                             │
│  [ Paste Text ] [ URL ] [ Headline ]        │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │ Paste your news article here...     │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  [ Run Fake News Detection ]                │
│                                             │
│  ✅ REAL NEWS — 89% Confidence             │
│  Risk: LOW | Style: Professional            │
│  ████████████████░░░░ 89%                  │
└─────────────────────────────────────────────┘
```

---

## 🎯 What It Does

You paste any news article, WhatsApp forward, headline, or URL — and the system tells you:

| Result | Meaning |
|--------|---------|
| ✅ **REAL** | Content shows credibility signals — likely authentic |
| 🚨 **FAKE** | Misinformation patterns detected — likely false |
| ⚠️ **UNCERTAIN** | Mixed signals — needs manual verification |

Along with the verdict, you also get:
- **Confidence percentage** — how sure the model is
- **Risk level** — LOW / MEDIUM / HIGH / CRITICAL
- **Writing style** — Professional or Sensationalist
- **9 linguistic features** — word count, CAPS%, keywords etc.
- **Plain English explanation** — why it was classified this way
- **Recommendation** — what you should do next

---

## 🧠 How It Works

```
Your Input (Text / URL / Headline)
           ↓
    Text Preprocessing
    (lowercase, remove URLs, clean text)
           ↓
    TF-IDF Vectorization
    (converts text to 50,000 numbers)
           ↓
    ┌──────────────────┐    ┌──────────────────┐
    │  Logistic        │    │  Rule-Based      │
    │  Regression      │    │  Linguistic      │
    │  ML Model (60%)  │    │  Score (40%)     │
    └────────┬─────────┘    └────────┬─────────┘
             └──────────┬────────────┘
                        ↓
               HYBRID SCORING
           Final = ML×0.60 + Rules×0.40
                        ↓
          ✅ REAL / 🚨 FAKE / ⚠️ UNCERTAIN
```

### Why Hybrid Scoring?
The WELFake dataset contains political fake news written in formal language, which causes the ML model alone to sometimes misclassify credible science/finance articles as fake. The hybrid approach combines ML probability with rule-based linguistic analysis to prevent this — making predictions more reliable.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **96.07%** |
| **ROC-AUC** | **99.28%** |
| **F1-Score** | **0.96** |
| **Precision (Real)** | 0.97 |
| **Precision (Fake)** | 0.95 |
| Training Dataset | WELFake (72,075 articles) |
| Test Set | 14,415 articles |

### Comparison with Other Methods

| Method | Accuracy | GPU Needed |
|--------|----------|------------|
| Naive Bayes | 88.3% | No |
| SVM + TF-IDF | 92.1% | No |
| **Our Model (LR + Hybrid)** | **96.07%** | **No** |
| BERT | 97.2% | Yes (4+ hours) |

---

## 🛠️ Tech Stack

**Frontend**
- HTML5, CSS3, JavaScript (Vanilla)
- Dark theme UI with animated confidence bar
- Responsive — works on mobile and desktop

**Backend**
- Python 3.10+
- Flask (REST API)
- Scikit-learn (TF-IDF + Logistic Regression)
- NumPy, Requests, BeautifulSoup4

**Dataset**
- [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) — 72,075 labeled news articles

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10 or above
- pip
- 4GB RAM minimum

### Step 1 — Clone the repo
```bash
git clone https://github.com/skmahaboobsubhani345-eng/fake-news-detector.git
cd fake-news-detector
```

### Step 2 — Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### Step 3 — Install packages
```bash
pip install -r requirements.txt
```

### Step 4 — Download dataset
Download [WELFake_Dataset.csv](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) from Kaggle and place it in the project folder.

### Step 5 — Train the model
```bash
python model.py
```
This takes about 60 seconds. Creates `fake_news_model_tfidf.pkl` and `fake_news_model_clf.pkl`.

### Step 6 — Start the app
```bash
python app.py
```

### Step 7 — Open in browser
```
http://127.0.0.1:5000
```

---

## 🧪 Test Examples

**Paste this to test FAKE detection:**
```
SHOCKING!! Doctors HIDING miracle cure!! Big Pharma DOESNT want 
you to know!!! Government SUPPRESSING this!! Share before DELETED!!
100% GUARANTEED!!!
```

**Paste this to test REAL detection:**
```
Scientists at Johns Hopkins University published research in Nature 
Medicine showing combination therapy reduced tumor size in 67 percent 
of patients. The study involved 240 participants over 18 months. 
Dr Sarah Chen said Phase III trials are still needed before approval.
```

---

## 📁 Project Structure

```
fake-news-detector/
├── app.py                    ← Flask web server
├── model.py                  ← ML training engine
├── requirements.txt          ← Python dependencies
├── Procfile                  ← Deployment config
├── templates/
│   └── index.html            ← Web interface
├── fake_news_model_tfidf.pkl ← Trained vectorizer (generated)
├── fake_news_model_clf.pkl   ← Trained classifier (generated)
└── WELFake_Dataset.csv       ← Training dataset (download separately)
```

---

## 👥 Team

**Team No: 15 — Department of Artificial Intelligence and Machine Learning**

| Name | Roll Number |
|------|-------------|
| Shaik Mahaboob Subhani | 11523110068 |
| Raja Hemanth Sai | 11523110088 |
| Prujith K | 11523110062 |

**Guide:** R. Sivabalan, Assistant Professor

---

## 📚 References

1. Baarir N.F. & Djeffal A., *"Fake News Detection Using Machine Learning"*, IEEE IHSH 2021
2. Verma P. et al., *"WELFake: Word Embedding over Linguistic Features"*, IEEE TCSS 2021
3. Kula S. et al., *"BERT for Fake News Detection"*, MDPI Applied Sciences 2020
4. Zhou X. & Zafarani R., *"A Survey of Fake News"*, ACM Computing Surveys 2020

---

## 📄 License

This project is licensed under the MIT License — feel free to use it for academic purposes.

---

## ⭐ If this project helped you

Give it a star ⭐ on GitHub — it helps others find this project!

---

*Built with ❤️ for fighting misinformation — Department of AIML, 2025*
