from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Default Prediction", page_icon="📊", layout="wide")
PROJECT_DIR = Path(__file__).parent
SITE_DIR = PROJECT_DIR / "website"
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

@st.cache_data(show_spinner="Loading the public training dataset…")
def load_training_data():
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()
    X.columns = [str(column).strip() for column in X.columns]
    uci_names = [f"X{i}" for i in range(1, 24)]
    if set(uci_names).issubset(X.columns):
        X = X.rename(columns=dict(zip(uci_names, FEATURES)))
    elif not set(FEATURES).issubset(X.columns) and X.shape[1] == len(FEATURES):
        X.columns = FEATURES
    if not set(FEATURES).issubset(X.columns):
        raise ValueError(f"Expected the 23 UCI features; received {list(X.columns)}")
    return X[FEATURES], y.astype(int)

@st.cache_resource(show_spinner="Training the selected model on the public dataset…")
def train_model(model_name):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_training_data()
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric = [c for c in FEATURES if c not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    if model_name == "Logistic Regression":
        estimator = LogisticRegression(max_iter=2000, random_state=42)
    elif model_name == "Random Forest":
        estimator = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    else:
        from xgboost import XGBClassifier
        estimator = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=-1,
        )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipeline.fit(X, y)
    return pipeline

st.title("Credit Card Default Prediction")
st.caption("Enter an anonymized customer profile to estimate default probability with a machine learning model.")
st.info("Classroom demonstration only. This model uses historical Taiwan data from 2005 and must not be used for real credit or lending decisions.")

with st.form("customer_prediction_form"):
    model_name = st.selectbox("Prediction model", ["Logistic Regression", "Random Forest", "XGBoost"], index=1)
    st.subheader("Customer profile")
    profile = st.columns(5)
    with profile[0]:
        limit_balance = st.number_input("Credit limit (NT$)", 0, 10_000_000, 100_000, step=10_000)
    with profile[1]:
        gender_label = st.selectbox("Gender", ["Male", "Female"])
    with profile[2]:
        education_label = st.selectbox("Education", ["Graduate school", "University", "High school", "Other"], index=1)
    with profile[3]:
        marital_label = st.selectbox("Marital status", ["Married", "Single", "Other"], index=1)
    with profile[4]:
        age = st.number_input("Age", 18, 100, 35)

    st.subheader("Repayment history")
    st.caption("−2 = no consumption · −1 = paid duly · 0 = revolving credit · 1–9 = months delayed")
    pay_columns = st.columns(6)
    pay_fields = [("PAY_0", "Most recent"), ("PAY_2", "Month −2"), ("PAY_3", "Month −3"), ("PAY_4", "Month −4"), ("PAY_5", "Month −5"), ("PAY_6", "Month −6")]
    history = {}
    for (field, label), column in zip(pay_fields, pay_columns):
        with column:
            history[field] = st.selectbox(label, list(range(-2, 10)), index=2, key=f"{field}_input")

    st.subheader("Monthly bills and payments (NT$)")
    st.caption("Enter the latest month first, followed by the five preceding months.")
    bill_columns = st.columns(6)
    bills = {}
    for month, column in enumerate(bill_columns, start=1):
        with column:
            bills[f"BILL_AMT{month}"] = st.number_input(f"Bill {month}", -10_000_000, 100_000_000, 5_000 if month == 1 else 0, step=1_000, key=f"bill_{month}")
    payment_columns = st.columns(6)
    payments = {}
    for month, column in enumerate(payment_columns, start=1):
        with column:
            payments[f"PAY_AMT{month}"] = st.number_input(f"Payment {month}", 0, 100_000_000, 2_000 if month == 1 else 0, step=1_000, key=f"payment_{month}")

    threshold = st.slider("Flag as higher risk at probability", 0.10, 0.90, 0.50, 0.05)
    submitted = st.form_submit_button("Estimate default risk", type="primary", use_container_width=True)

if submitted:
    st.session_state.pop("customer_prediction", None)
    values = {
        "LIMIT_BAL": limit_balance,
        "SEX": 1 if gender_label == "Male" else 2,
        "EDUCATION": {"Graduate school": 1, "University": 2, "High school": 3, "Other": 4}[education_label],
        "MARRIAGE": {"Married": 1, "Single": 2, "Other": 3}[marital_label],
        "AGE": age,
        **history, **bills, **payments,
    }
    try:
        model = train_model(model_name)
        customer = pd.DataFrame([[values[column] for column in FEATURES]], columns=FEATURES)
        probability = float(model.predict_proba(customer)[0, 1])
        st.session_state["customer_prediction"] = {
            "model": model_name, "probability": probability,
            "threshold": threshold, "high_risk": probability >= threshold,
        }
    except Exception as error:
        st.error("Could not load the public training dataset or train the model. Check Manage app → Logs for details.")
        with st.expander("Technical details"):
            st.code(str(error))

result = st.session_state.get("customer_prediction")
if result:
    st.subheader("Prediction result")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Estimated default probability", f"{result['probability']:.1%}")
    with right:
        if result["high_risk"]:
            st.warning(f"Higher-risk flag at the {result['threshold']:.0%} threshold")
        else:
            st.success(f"Below the {result['threshold']:.0%} threshold")
        st.caption(f"Model: {result['model']} · This is an estimate, not a guaranteed outcome.")
    st.progress(result["probability"])

st.divider()
st.subheader("Project dashboard")
html = (SITE_DIR / "index.html").read_text(encoding="utf-8")
css = (SITE_DIR / "styles.css").read_text(encoding="utf-8")
js = (SITE_DIR / "app.js").read_text(encoding="utf-8")
html = html.replace('<link rel="stylesheet" href="styles.css">', f"<style>\n{css}\n</style>")
html = html.replace('  <script src="app.js" defer></script>\n', "")
html = html.replace("</body>", f"<script>\n{js}\n</script>\n</body>")
st.iframe(html, height="content", alt="Interactive Credit Card Default Prediction project dashboard")
