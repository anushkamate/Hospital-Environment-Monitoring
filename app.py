import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Hospital Environment Monitor", page_icon="🏥", layout="wide")
st.title("🏥 Hospital Environment Monitoring")
st.caption("Machine Learning • Edge Inference Simulation • Room-wise Environmental Review")

FEATURES = ["NH3", "NO2", "CO", "PM2.5", "Temp", "Pressure", "Humidity", "O3"]
POLLUTANTS = ["NH3", "NO2", "CO", "PM2.5", "O3"]
CLASSES = ["Normal", "Warning", "Critical"]

st.warning(
    "Prototype only: labels are generated from configurable demonstration rules. "
    "They are not validated hospital safety limits, and predictions are not medical advice."
)

def normalize_columns(frame):
    """Map common variations in dataset column names to expected names."""
    df = frame.copy()
    normalized = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    aliases = {
        "nh3": ["nh3", "ammonia"],
        "no2": ["no2", "nitrogendioxide"],
        "co": ["co", "carbonmonoxide"],
        "pm2.5": ["pm2.5", "pm25", "pm2_5", "pm2,5"],
        "temp": ["temp", "temperature", "temperaturec"],
        "pressure": ["pressure", "atmosphericpressure"],
        "humidity": ["humidity", "relativehumidity"],
        "o3": ["o3", "ozone"],
    }
    rename = {}
    for target, names in aliases.items():
        for name in names:
            key = name.lower().replace(" ", "").replace("_", "")
            if key in normalized:
                rename[normalized[key]] = target
                break
    return df.rename(columns=rename)

def prepare_data(raw):
    df = normalize_columns(raw)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing) +
            ". Required columns: " + ", ".join(FEATURES)
        )
    df = df[FEATURES].copy()
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    if len(df) < 30:
        raise ValueError("At least 30 valid rows are recommended to train and test this demonstration.")
    return df

def make_labels(df):
    """Generate rule-based demonstration labels, not ground-truth safety labels."""
    labeled = df.copy()
    pollution = labeled[POLLUTANTS]
    warning_limits = pollution.quantile(0.75)
    critical_limits = pollution.quantile(0.90)

    def classify(row):
        warning_count = sum(row[c] > warning_limits[c] for c in POLLUTANTS)
        critical_count = sum(row[c] > critical_limits[c] for c in POLLUTANTS)
        temp, humidity = row["Temp"], row["Humidity"]

        if (critical_count >= 2 or temp < 18 or temp > 30 or humidity < 30 or humidity > 70):
            return "Critical"
        if (warning_count >= 2 or temp < 20 or temp > 27 or humidity < 35 or humidity > 65):
            return "Warning"
        return "Normal"

    labeled["label"] = labeled.apply(classify, axis=1)
    return labeled, warning_limits, critical_limits

@st.cache_data(show_spinner=False)
def train_and_evaluate(csv_bytes):
    import io
    raw = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    data = prepare_data(raw)
    labeled, warning_limits, critical_limits = make_labels(data)
    counts = labeled["label"].value_counts()
    if counts.min() < 2:
        raise ValueError(
            "The generated labels do not contain at least two examples per class. "
            "Try a different dataset or revise the demonstration thresholds."
        )

    X = labeled[FEATURES]
    y = labeled["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(
        y_test, pred, labels=CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    return labeled, model, accuracy, report, cm, warning_limits, critical_limits

uploaded = st.sidebar.file_uploader(
    "Upload your indoor air pollution CSV",
    type=["csv"],
    help="The CSV must contain NH3, NO2, CO, PM2.5, Temp, Pressure, Humidity and O3 columns."
)

if uploaded is None:
    st.info("Upload your Indoor Air Pollution CSV in the sidebar to train the demonstration model and use the dashboard.")
    st.markdown("### Required columns")
    st.code(", ".join(FEATURES))
    st.markdown("### How to deploy")
    st.markdown("1. Upload this repository to GitHub.\n2. In Streamlit Community Cloud, create an app from your repository and select `app.py`.\n3. Open the deployed app and upload the CSV.")
    st.stop()

try:
    with st.spinner("Preparing data and training the demonstration model..."):
        labeled, model, accuracy, report, cm, warning_limits, critical_limits = train_and_evaluate(uploaded.getvalue())
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"Dataset loaded: {len(labeled):,} valid records.")
m1, m2, m3 = st.columns(3)
m1.metric("Valid dataset rows", f"{len(labeled):,}")
m2.metric("Held-out test accuracy", f"{accuracy * 100:.2f}%")
m3.metric("Features used", len(FEATURES))

with st.expander("Model evaluation details"):
    st.write("Accuracy is measured against rule-generated demonstration labels, not verified hospital safety labels.")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    st.write("Confusion matrix (rows = true rule-generated labels; columns = predictions)")
    st.dataframe(cm_df, use_container_width=True)

st.divider()
st.subheader("Room readings — edge inference simulation")
st.write("Enter values below to classify one simulated room. Use the units represented by your source dataset.")

room = st.selectbox(
    "Room",
    ["General Ward", "ICU", "Operation Theatre", "Patient Room", "Laboratory", "Other"]
)

# Use medians of dataset columns as safe UI defaults, not safety targets.
medians = labeled[FEATURES].median()
with st.form("room_form"):
    cols = st.columns(4)
    values = {}
    for i, feature in enumerate(FEATURES):
        with cols[i % 4]:
            values[feature] = st.number_input(
                feature,
                value=float(medians[feature]),
                format="%.4f",
                key=f"val_{feature}"
            )
    submitted = st.form_submit_button("Classify room")

if submitted:
    sample = pd.DataFrame([values], columns=FEATURES)
    prediction = model.predict(sample)[0]
    st.session_state["last_result"] = {
        "room": room, **values, "prediction": prediction,
        "timestamp": pd.Timestamp.now().isoformat()
    }

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    st.markdown(f"### Result for {result['room']}")
    if result["prediction"] == "Critical":
        st.error("CRITICAL — review the room readings and verify with appropriate instruments.")
    elif result["prediction"] == "Warning":
        st.warning("WARNING — facility staff should review the readings.")
    else:
        st.success("NORMAL — no warning was produced by this demonstration model.")

    st.caption("This is a model output, not proof that a room is safe.")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(pd.DataFrame([{f: result[f] for f in FEATURES}]))
        st.metric("Model class probability", f"{np.max(probs) * 100:.1f}%")
        st.caption("Class probability is not the probability that the room is safe.")

st.divider()
st.subheader("Simulated room overview")
st.caption("Rows are sampled from the uploaded dataset and assigned fictional room names; they are not actual hospital sensor readings.")

rng = np.random.default_rng(42)
room_names = ["General Ward", "ICU", "Operation Theatre", "Patient Room", "Laboratory"]
indices = rng.choice(len(labeled), size=min(5, len(labeled)), replace=False)
overview = labeled.iloc[indices][FEATURES].copy().reset_index(drop=True)
overview.insert(0, "Room", room_names[:len(overview)])
overview["Predicted Condition"] = model.predict(overview[FEATURES])
st.dataframe(overview, use_container_width=True)

counts = overview["Predicted Condition"].value_counts().reindex(CLASSES, fill_value=0)
fig, ax = plt.subplots(figsize=(7, 3.5))
counts.plot(kind="bar", ax=ax)
ax.set_title("Predicted conditions in simulated room sample")
ax.set_xlabel("Predicted condition")
ax.set_ylabel("Number of rooms")
ax.tick_params(axis="x", rotation=0)
st.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(8, 3.5))
ax2.bar(overview["Room"], overview["Temp"])
ax2.set_title("Temperature by simulated room")
ax2.set_xlabel("Simulated room")
ax2.set_ylabel("Temperature (dataset units)")
ax2.tick_params(axis="x", rotation=20)
st.pyplot(fig2)

st.download_button(
    "Download simulated room report (CSV)",
    data=overview.to_csv(index=False).encode("utf-8"),
    file_name="hospital_environment_report.csv",
    mime="text/csv"
)

st.download_button(
    "Download labelled dataset (CSV)",
    data=labeled.to_csv(index=False).encode("utf-8"),
    file_name="hospital_environment_labelled_demo.csv",
    mime="text/csv"
)

st.sidebar.markdown("---")
st.sidebar.caption("Facility-monitoring prototype only. Configure and validate thresholds before any real-world use.")
