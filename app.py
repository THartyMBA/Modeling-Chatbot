# unified_model_chat_app.py
"""
🧠➕📈  Data-Aware Chatbot + Universal Classification Studio
-----------------------------------------------------------------------

One Streamlit file that lets users…

1. **Upload** any CSV (mixed numeric & categorical columns).  
2. **Select** a target column and algorithm, then **Train** a model.  
3. Review accuracy / ROC-AUC, inspect the scored data, **download** results.  
4. **Chat** with a memory-enabled OpenRouter model that can reference the
   uploaded data, the trained model’s metrics, and the scored output.

────────────────────────────────────────────────────────────
!!  REQUIRE  `OPENROUTER_API_KEY`  in env or st.secrets   !!
────────────────────────────────────────────────────────────
"""

# ╭─ Imports ────────────────────────────────────────────────────────────────╮
import os, requests, io
import pandas as pd, numpy as np, streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# ╭─ Keys & Config ──────────────────────────────────────────────────────────╮
API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
DEFAULT_MODEL = "mistralai/mistral-7b-instruct"  # free tier

# ╭─ OpenRouter Chat Helper ─────────────────────────────────────────────────╮
def openrouter_chat(messages, model=DEFAULT_MODEL, temperature=0.7):
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing.")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://portfolio.example",   # customise
        "X-Title": "ModelChatDemo",
    }
    body = {"model": model, "messages": messages, "temperature": temperature}
    r = requests.post(url, headers=headers, json=body, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ╭─ ML Utilities ───────────────────────────────────────────────────────────╮
def get_estimator(name):
    if name == "Gradient Boosting":
        return GradientBoostingClassifier()
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    return LogisticRegression(max_iter=2000, n_jobs=-1)

def build_pipeline(df, target, model_name):
    num_cols = df.drop(columns=[target]).select_dtypes(include="number").columns
    cat_cols = df.drop(columns=[target]).select_dtypes(exclude="number").columns

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("enc", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
    return Pipeline([("pre", pre), ("clf", get_estimator(model_name))])

def train_model(df, target, algo, test_size):
    X, y = df.drop(columns=[target]), df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          stratify=y, random_state=42)
    pipe = build_pipeline(df, target, algo)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)
    ypred = np.argmax(proba, axis=1) if proba.shape[1]>2 else (proba[:,1]>0.5)
    acc = accuracy_score(yte, ypred)
    try:
        auc = roc_auc_score(yte, proba, multi_class="ovr" if proba.shape[1]>2 else "raise")
    except ValueError:
        auc = float("nan")
    metrics = {"accuracy": acc, "roc_auc": auc, "classes": list(pipe.classes_)}
    roc_dat = None
    if proba.shape[1]==2:
        fpr, tpr, _ = roc_curve(yte, proba[:,1])
        roc_dat = (fpr, tpr)
    scored = df.copy()
    scored["pred_proba"] = (
        pipe.predict_proba(X)[:,1] if proba.shape[1]==2
        else pipe.predict_proba(X).max(axis=1)
    )
    return pipe, metrics, roc_dat, scored

def plot_roc(fpr, tpr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"),
                             name="Random"))
    fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=450, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# ╭─ Session Init ───────────────────────────────────────────────────────────╮
def init_session():
    if "chat" not in st.session_state:
        st.session_state.chat=[{"role":"system","content":
            "You are ModelChat, a helpful data scientist. "
            "If the user uploads data or trains a model, you can reference it via "
            "`df` (raw) or `scored_df`, and model metrics via `model_metrics`."}]
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("scored_df", None)
    st.session_state.setdefault("roc_plot", None)
    st.session_state.setdefault("model_metrics", "")

# ╭─ Streamlit UI ───────────────────────────────────────────────────────────╮
st.set_page_config(page_title="🧠 Data-Aware Model Chatbot", layout="wide")
init_session()

with st.sidebar:
    st.header("📂 Upload & Train")
    file = st.file_uploader("CSV file", type="csv")
    if file:
        st.session_state.df = pd.read_csv(file)
        st.success("Data loaded!")
        st.dataframe(st.session_state.df.head())
        target = st.selectbox("🎯 Target column", st.session_state.df.columns, key="target")
        algo = st.selectbox("⚙️ Algorithm", ["Gradient Boosting","Random Forest","Logistic Regression"])
        test_split = st.slider("Validation split %", 0.1, 0.5, 0.2, 0.05)
        if st.button("🚀 Train"):
            with st.spinner("Training…"):
                pipe, metrics, roc_dat, scored = train_model(
                    st.session_state.df, target, algo, test_split)
                st.session_state.scored_df = scored
                st.session_state.model_metrics = f"**Accuracy**: {metrics['accuracy']:.3f} &nbsp; | &nbsp; " \
                                                f"**ROC-AUC**: {'N/A' if np.isnan(metrics['roc_auc']) else f'{metrics['roc_auc']:.3f}'}"
                if roc_dat:
                    st.session_state.roc_plot = plot_roc(*roc_dat)
                else:
                    st.session_state.roc_plot = None
            st.success("Model ready! Discuss below.")

# Main – show ROC and metrics if present
st.title("🧠 Data-Aware Model Chatbot")
st.info(
    "🔔 **Demo Notice**  \n"
    "This app is a lightweight proof-of-concept. For brand-safe, enterprise-grade "
    "generative marketing systems, [contact me](https://drtomharty.com/bio).",
    icon="💡"
)

if st.session_state.roc_plot is not None:
    st.subheader("ROC Curve")
    st.plotly_chart(st.session_state.roc_plot, use_container_width=True)
    st.caption(st.session_state.model_metrics)

# Display chat history
for m in st.session_state.chat[1:]:
    st.chat_message(m["role"]).markdown(m["content"])

# Chat input
user_msg = st.chat_input("Ask me about your model or anything…")
if user_msg:
    st.session_state.chat.append({"role":"user","content":user_msg})
    st.chat_message("user").markdown(user_msg)

    # Add small context for grounding
    if st.session_state.df is not None:
        st.session_state.chat.append(
            {"role":"system",
             "content":"First rows of uploaded data:\n\n"+st.session_state.df.head().to_markdown(index=False)}
        )
    if st.session_state.scored_df is not None:
        st.session_state.chat.append(
            {"role":"system",
             "content":"Sample of scored data (last 5 rows):\n\n"+st.session_state.scored_df.tail(5).to_markdown(index=False)}
        )
        st.session_state.chat.append(
            {"role":"system",
             "content":"Model metrics summary:\n\n"+st.session_state.model_metrics}
        )

    with st.spinner("Thinking…"):
        try:
            reply = openrouter_chat(st.session_state.chat)
        except Exception as e:
            st.error(f"OpenRouter error: {e}")
            st.stop()

    st.session_state.chat.append({"role":"assistant","content":reply})
    st.chat_message("assistant").markdown(reply)
