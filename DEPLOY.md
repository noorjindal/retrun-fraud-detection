# Deploy dashboard (simplest — free)

Use **Streamlit Community Cloud**. No credit card. ~5 minutes.

## What gets deployed

- Live **dashboard** only (Live Scoring, Batch, Model Insights, Dataset Explorer)
- Model files (`model/fraud_model.pkl`) are included in the repo

The Flask API stays local unless you deploy it separately later.

---

## Step 1 — Push to GitHub

```bash
cd ~/Desktop/return-fraud
git init
git add .
git commit -m "Return fraud detection — Streamlit deploy ready"
```

Create a new repo on GitHub: https://github.com/new  
Name it `return-fraud-detection` (public).

```bash
git remote add origin https://github.com/YOUR_USERNAME/return-fraud-detection.git
git branch -M main
git push -u origin main
```

---

## Step 2 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with **GitHub**
3. Click **New app**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/return-fraud-detection`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **Deploy**

Wait 2–5 minutes. You get a public URL like:

`https://return-fraud-detection-xxxxx.streamlit.app`

---

## Step 3 — Share the link

Put the URL in your README and resume.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Model not found` | Ensure `model/fraud_model.pkl` is committed (`git add model/`) |
| Build fails on imports | `requirements.txt` is at repo root |
| App is slow first load | Normal — model loads once (~2MB) |

---

## Run locally (same as cloud)

```bash
cd ~/Desktop/return-fraud
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501
