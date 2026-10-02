import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

st.set_page_config(page_title="Student Placement Prediction", layout="wide")

MODEL_OPTIONS = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
    "XGBoost": XGBClassifier,
}

REQUIRED_SCHEMAS = {
    "placementdata": {
        "columns": [
            "CGPA",
            "Internships",
            "Projects",
            "Workshops/Certifications",
            "AptitudeTestScore",
            "SoftSkillsRating",
            "ExtracurricularActivities",
            "PlacementTraining",
            "SSC_Marks",
            "HSC_Marks",
            "PlacementStatus",
        ],
        "target": "PlacementStatus",
    },
    "campusplacement": {
        "columns": [
            "ssc_p",
            "hsc_p",
            "degree_p",
            "workex",
            "etest_p",
            "mba_p",
            "status",
        ],
        "target": "status",
    },
}

FEATURE_COLUMNS = [
    "CGPA",
    "Degree Percentage",
    "10th Percentage",
    "12th Percentage",
    "Internship Experience",
    "Communication Skills",
    "Technical Skills",
]

TARGET_LABELS = {0: "Not Placed", 1: "Placed"}


@st.cache_data
def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        return None


def detect_schema(df: pd.DataFrame):
    if df is None:
        return None
    columns = set(df.columns.str.strip())
    if set(REQUIRED_SCHEMAS["placementdata"]["columns"]).issubset(columns):
        return "placementdata"
    if set(REQUIRED_SCHEMAS["campusplacement"]["columns"]).issubset(columns):
        return "campusplacement"
    return None


def clean_dataset(df: pd.DataFrame, schema: str):
    df = df.copy()
    if schema == "placementdata":
        df = df.rename(
            columns={
                "SSC_Marks": "10th Percentage",
                "HSC_Marks": "12th Percentage",
                "Internships": "Internship Experience",
                "SoftSkillsRating": "Communication Skills",
                "AptitudeTestScore": "Technical Skills",
                "PlacementStatus": "status",
            }
        )
        df["Degree Percentage"] = df["CGPA"] * 10
        df = df[FEATURE_COLUMNS + ["status"]].copy()
        df["status"] = df["status"].astype(str).str.strip().replace(
            {"Placed": 1, "NotPlaced": 0, "Not Placed": 0}
        )
    elif schema == "campusplacement":
        df = df.rename(
            columns={
                "ssc_p": "10th Percentage",
                "hsc_p": "12th Percentage",
                "degree_p": "Degree Percentage",
                "workex": "Internship Experience",
                "etest_p": "Communication Skills",
                "mba_p": "Technical Skills",
                "status": "status",
            }
        )
        df["CGPA"] = df["Degree Percentage"] / 10
        df["Internship Experience"] = df["Internship Experience"].astype(str).str.strip().replace(
            {"Yes": 1, "No": 0, "yes": 1, "no": 0}
        )
        df = df[FEATURE_COLUMNS + ["status"]].copy()
        df["status"] = df["status"].astype(str).str.strip().replace(
            {"Placed": 1, "Not Placed": 0, "NotPlaced": 0}
        )
    else:
        raise ValueError("Unknown schema")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df["status"] = df["status"].astype(int)
    return df


def validate_dataset(df: pd.DataFrame):
    errors = []
    if df is None:
        errors.append("Uploaded file cannot be read as CSV.")
        return None, errors
    schema = detect_schema(df)
    if schema is None:
        errors.append(
            "Dataset does not match the supported placement dataset schemas."
        )
        return None, errors
    cleaned = clean_dataset(df, schema)
    if cleaned.empty:
        errors.append("After cleaning, the dataset contains no valid rows.")
        return None, errors
    if cleaned["status"].nunique() < 2:
        errors.append("Dataset must contain both placed and not placed examples.")
    return cleaned, errors


def build_model(model_name: str):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=500)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42, n_estimators=100)
    elif model_name == "SVM":
        model = SVC(probability=True, random_state=42)
    elif model_name == "XGBoost":
        if XGBClassifier is None:
            raise ImportError(
                "XGBoost is not installed. Install dependencies or choose another model."
            )
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    else:
        raise ValueError("Unsupported model")

    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", model)])
    return pipeline


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def predict_single(model, row):
    proba = model.predict_proba(row)[0, 1]
    label = TARGET_LABELS[int(proba >= 0.5)]
    return label, float(proba)


def feature_importance(model, feature_names):
    clf = model.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_.reshape(-1))
    else:
        values = np.zeros(len(feature_names))
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def render_metrics(metrics):
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    st.metric("Precision", f"{metrics['precision']:.2%}")
    st.metric("Recall", f"{metrics['recall']:.2%}")
    st.metric("F1 Score", f"{metrics['f1_score']:.2%}")


def render_classification_report(y_test, y_pred):
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Not Placed", "Placed"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.round(2)
    report_df = report_df.rename(columns={
        "precision": "Precision",
        "recall": "Recall",
        "f1-score": "F1 Score",
        "support": "Support",
    })
    return report_df


def plot_heatmap(df: pd.DataFrame):
    corr = df[FEATURE_COLUMNS + ["status"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)


def plot_distribution(df: pd.DataFrame):
    counts = df["status"].map(TARGET_LABELS).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, palette="muted", ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Placement Distribution")
    st.pyplot(fig)


def plot_feature_importance(importances: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    st.pyplot(fig)


def make_report(df: pd.DataFrame, y_test, y_pred, y_proba):
    report_df = df.copy()
    report_df["Actual"] = y_test.map(TARGET_LABELS)
    report_df["Predicted"] = pd.Series(y_pred, index=report_df.index).map(TARGET_LABELS)
    report_df["Placement Probability"] = np.round(y_proba * 100, 2)
    return report_df


def main():
    st.title("Student Placement Prediction")
    st.write(
        "Upload a placement dataset, choose a model, and predict whether a student is likely to be placed."
    )

    with st.sidebar:
        st.header("Upload & Model Selection")
        uploaded_file = st.file_uploader(
            "Upload a placement dataset (.csv)", type=["csv"]
        )
        default_dataset = st.selectbox(
            "Or choose a sample dataset", ["None", "placementdata.csv", "Placement_Data_Full_Class.csv"]
        )
        selected_model = st.selectbox(
            "Select ML model", list(MODEL_OPTIONS.keys()), index=0
        )
        st.markdown("---")
        st.write("Upload a dataset first to enable dataset-aware student input ranges.")

    df = None
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
    elif default_dataset != "None":
        try:
            df = pd.read_csv(default_dataset)
        except Exception:
            df = None

    if df is None:
        st.warning(
            "Upload a supported placement dataset or select one of the sample datasets to continue."
        )
        return

    cleaned, errors = validate_dataset(df)
    if errors:
        st.error("Dataset validation failed")
        for err in errors:
            st.write(f"- {err}")
        return

    st.success("Dataset validated successfully.")
    st.write(f"Dataset rows: {cleaned.shape[0]}, features: {len(FEATURE_COLUMNS)}")
    st.data_editor(cleaned.head(8), use_container_width=True)

    # Determine schema and set dataset-aware input ranges
    schema = detect_schema(df)
    if schema == "placementdata":
        comm_min, comm_max, comm_step, comm_default = 0.0, 5.0, 0.1, round(cleaned["Communication Skills"].median(), 2)
        tech_min, tech_max, tech_step, tech_default = 0.0, 100.0, 1.0, round(cleaned["Technical Skills"].median(), 1)
        degree_default = round(cleaned["Degree Percentage"].median(), 1) if "Degree Percentage" in cleaned.columns else 70.0
    elif schema == "campusplacement":
        comm_min, comm_max, comm_step, comm_default = 0.0, 100.0, 1.0, round(cleaned["Communication Skills"].median(), 1) if "Communication Skills" in cleaned.columns else 60.0
        tech_min, tech_max, tech_step, tech_default = 0.0, 100.0, 1.0, round(cleaned["Technical Skills"].median(), 1) if "Technical Skills" in cleaned.columns else 70.0
        degree_default = round(cleaned["Degree Percentage"].median(), 1) if "Degree Percentage" in cleaned.columns else 70.0
    else:
        comm_min, comm_max, comm_step, comm_default = 0.0, 100.0, 1.0, 50.0
        tech_min, tech_max, tech_step, tech_default = 0.0, 100.0, 1.0, 50.0
        degree_default = 70.0

    # Render student input in sidebar using dataset-aware ranges
    with st.sidebar:
        st.markdown("## Student Input")
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=round(cleaned["CGPA"].median(), 2) if "CGPA" in cleaned.columns else 7.0, step=0.1)
        degree_pct = st.number_input(
            "Degree Percentage", min_value=0.0, max_value=100.0, value=degree_default, step=0.1
        )
        ssc_pct = st.number_input(
            "10th Percentage", min_value=0.0, max_value=100.0, value=round(cleaned["10th Percentage"].median(), 1) if "10th Percentage" in cleaned.columns else 75.0, step=0.1
        )
        hsc_pct = st.number_input(
            "12th Percentage", min_value=0.0, max_value=100.0, value=round(cleaned["12th Percentage"].median(), 1) if "12th Percentage" in cleaned.columns else 75.0, step=0.1
        )
        workex = st.selectbox("Internship Experience", ["No", "Yes"], index=0 if cleaned["Internship Experience"].median() < 0.5 else 1)
        communication = st.slider("Communication Skills", min_value=comm_min, max_value=comm_max, value=float(comm_default), step=comm_step)
        technical = st.slider("Technical Skills", min_value=tech_min, max_value=tech_max, value=float(tech_default), step=tech_step)
        st.markdown("---")
        st.write("The input fields are now dataset-aware — ranges reflect the loaded dataset.")

    X = cleaned[FEATURE_COLUMNS]
    y = cleaned["status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    try:
        model = build_model(selected_model)
    except ImportError as exc:
        st.error(str(exc))
        return

    with st.spinner(f"Training {selected_model}..."):
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    st.subheader("Model Evaluation")
    render_metrics(metrics)

    st.subheader("Evaluation Details")
    st.write("**Classification report:**")
    report_df = render_classification_report(y_test, metrics["y_pred"])
    st.dataframe(report_df, use_container_width=True)

    if st.button("Generate Prediction Report"):
        report_df = make_report(X_test, y_test, metrics["y_pred"], metrics["y_proba"])
        csv = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction report",
            data=csv,
            file_name="placement_prediction_report.csv",
            mime="text/csv",
        )

    st.subheader("Visualizations")
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        plot_distribution(cleaned)
    with plot_col2:
        plot_heatmap(cleaned)

    st.subheader("Feature Importance")
    importances = feature_importance(model, FEATURE_COLUMNS)
    plot_feature_importance(importances)

    st.subheader("Student Placement Prediction")
    if st.button("Predict Placement"):
        profile = np.array(
            [
                cgpa,
                degree_pct,
                ssc_pct,
                hsc_pct,
                1 if workex == "Yes" else 0,
                communication,
                technical,
            ]
        ).reshape(1, -1)
        prediction_label, probability = predict_single(model, profile)
        st.write(f"**Prediction:** {prediction_label}")
        st.write(f"**Placement probability:** {probability * 100:.2f}%")

        profile_df = pd.DataFrame(
            [
                {
                    "CGPA": cgpa,
                    "Degree Percentage": degree_pct,
                    "10th Percentage": ssc_pct,
                    "12th Percentage": hsc_pct,
                    "Internship Experience": workex,
                    "Communication Skills": communication,
                    "Technical Skills": technical,
                    "Predicted Status": prediction_label,
                    "Placement Probability (%)": round(probability * 100, 2),
                }
            ]
        )
        st.data_editor(profile_df, use_container_width=True)
        buffer = io.StringIO()
        profile_df.to_csv(buffer, index=False)
        st.download_button(
            "Download student prediction", data=buffer.getvalue().encode("utf-8"), file_name="student_prediction.csv", mime="text/csv",
        )

    st.markdown("---")
    st.caption(
        "This app supports training on uploaded placement datasets and predicts placement likelihood for a student profile."
    )


if __name__ == "__main__":
    main()
