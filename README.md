# 🛡️ RoadShield AI — Road Accident Severity Predictor

![RoadShield AI Banner](https://img.shields.io/badge/ML-Random%20Forest-22c55e?style=for-the-badge) 
![Status](https://img.shields.io/badge/Status-Active-3b82f6?style=for-the-badge)
![Group](https://img.shields.io/badge/Group-3-f59e0b?style=for-the-badge)

RoadShield AI is a machine learning web application that predicts road accident severity (Slight, Serious, or Fatal) based on real-world conditions. It leverages a pre-trained **Random Forest** model on the Ethiopian accident dataset and uses **Explainable AI (SHAP)** to break down exactly *why* a prediction was made.

---

## ✨ Key Features

- **🎯 Severity Prediction:** Accurately classifies accident outcomes based on driver, vehicle, and environmental conditions.
- **🧠 Explainable AI:** Uses SHAP (Shapley Additive exPlanations) to dynamically highlight top risk and mitigating factors contributing to the predicted severity.
- **🛡️ UX-Optimised Input:** Groups questions smartly, only requiring inputs realistically known *before* an accident, with an "Advanced Mode" for deep analysis.
- **📊 Real-time Dashboard:** Built-in Exploratory Data Analysis (EDA) views highlighting underlying dataset trends like chi-square associations and temporal frequencies.
- **🌙 Premium Dark Theme:** Fully immersive dark UI customized heavily beyond basic Streamlit defaults.

## 🚀 Running the App Locally

To deploy RoadShield AI securely on your own machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pranati243/RoadShieldAI.git
   cd RoadShieldAI
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Application:**
   ```bash
   streamlit run app.py
   ```

## ⚙️ Built With

- **[Python](https://python.org/)** — Core backbone logic
- **[Streamlit](https://streamlit.io/)** — Fast interactive frontend web application
- **[Scikit-Learn](https://scikit-learn.org/)** — Random Forest ML pipeline modeling
- **[SHAP](https://shap.readthedocs.io/)** — TreeExplainer game-theory explainability
- **[Plotly](https://plotly.com/)** — Highly interactive, responsive data visualization

## 👥 Meet the Team
This initiative is a Machine Learning academic project by **Group 3**, Department of Information Technology at Fr. C. Rodrigues Institute of Technology (2025-26):
*   **Pranati Arun** (5023141)
*   **Vaibhavi Rai** (5023143)
*   **Ishwari Shinde** (5023155)
