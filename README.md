# Modeling-Chatbot
🧠🔍 Universal Classification Studio · with Data-Aware Chatbot
Upload a dataset ➜ train a quick probability model ➜ talk with an LLM that understands the data, the metrics, and the scored output—all on one Streamlit page.

What it does
Upload any tabular CSV (numeric and/or categorical columns).

Pick the target column & algorithm (LogReg, Gradient Boosting, or Random Forest).

Train – the app auto-handles missing values, scaling, one-hot encoding, and a stratified train/validation split.

Review accuracy, ROC-AUC, and an interactive ROC curve (binary targets).

Download the scored CSV and the trained .pkl model.

Chat with a free OpenRouter model (default mistral-7b-instruct).

The bot receives small snapshots of your dataframe, the scored results, and key performance metrics as hidden context, so it can answer questions like:

“Which class has the worst precision?”

“Why is cluster 3 under-represented?”

“What threshold gives recall ≥ 0.9?”

Proof-of-concept only—no hyper-parameter tuning, fairness checks, or MLOps.
For enterprise-grade model factories and LLM guards, visit drtomharty.com/bio.

Demo GIF
(drop in your own GIF or screenshot here)

Key Features

Module	Highlights
Model builder	One-click pipelines (impute ▸ scale/OHE ▸ classifier).
Metrics	Accuracy & ROC-AUC shown as Streamlit metrics; ROC curve via Plotly.
Outputs	scored_data.csv + model.pkl download buttons.
LLM chat	Adjustable temperature, model selector, full session memory.
Context injection	Head of raw data + tail of scored data + metrics posted as hidden system messages on every turn—keeps token budget low but answers rich.
Secrets
Add your OpenRouter API key so the chatbot works.

Streamlit Cloud
java
Copy
Edit
⋯  →  Edit secrets
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxx"
Local dev
toml
Copy
Edit
# ~/.streamlit/secrets.toml
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxx"
—or—

bash
Copy
Edit
export OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
Quick Start (local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/universal-classification-chatbot.git
cd universal-classification-chatbot
python -m venv venv && source venv/bin/activate   # Win: venv\Scripts\activate
pip install -r requirements.txt
streamlit run unified_model_chat_app.py
Open http://localhost:8501, drop in a CSV, and start chatting.

Free Deployment on Streamlit Cloud
Push the repo (public or private) to GitHub.

Go to streamlit.io/cloud ➜ New app and select the repo/branch.

Add OPENROUTER_API_KEY in Secrets.

Click Deploy—done!

(The app uses only CPU-friendly libraries, so the free tier is enough.)

Requirements
shell
Copy
Edit
streamlit>=1.32
pandas
numpy
scikit-learn
plotly
requests
Repo Layout
vbnet
Copy
Edit
unified_model_chat_app.py   ← single-file app
requirements.txt
README.md
License
CC0 1.0 – public-domain dedication. Attribution appreciated but not required.

Acknowledgements
Streamlit – rapid data apps

scikit-learn – the ML workhorse

Plotly – interactive charts

OpenRouter – effortless LLM access

Enjoy building, scoring, and chatting! 🚀
