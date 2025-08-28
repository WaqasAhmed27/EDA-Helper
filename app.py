import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             r2_score, mean_squared_error, mean_absolute_error)
from sklearn.impute import SimpleImputer

# -------------------------
# Session state defaults
# -------------------------
if "target_variable" not in st.session_state:
    st.session_state["target_variable"] = None
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "plot_key" not in st.session_state:
    st.session_state.plot_key = 0

# -------------------------
# Utilities
# -------------------------
def next_plot_key():
    st.session_state.plot_key += 1
    return f"plot_{st.session_state.plot_key}"

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        # Handle different delimiters
        if df.shape[1] == 1:
            file.seek(0)
            df = pd.read_csv(file, delimiter=';')
        if df.shape[1] == 1:
            file.seek(0)
            df = pd.read_csv(file, delimiter='\t')
        if df.shape[1] == 1:
            file.seek(0)
            df = pd.read_csv(file, delimiter='|')
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

def preprocess_for_ml(df, target_col, missing_strategy):
    """
    Returns X, y after basic preprocessing:
    - Optionally drop or impute missing values.
    - One-hot encodes categorical features (except the target).
    - Leaves numeric features as-is.
    """
    df_proc = df.copy()

    # Keep a copy to show how many rows/cols were removed if any
    original_shape = df_proc.shape

    # Handle missing values
    if missing_strategy == "Drop rows with NA":
        df_proc = df_proc.dropna()
    elif missing_strategy == "Simple Impute (mean/mode)":
        # numeric -> mean, categorical -> most_frequent
        for col in df_proc.columns:
            if col == target_col:
                continue
            if df_proc[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_proc[col]):
                    imp = SimpleImputer(strategy="mean")
                else:
                    imp = SimpleImputer(strategy="most_frequent")
                df_proc[[col]] = imp.fit_transform(df_proc[[col]])

    # Separate X and y
    if target_col not in df_proc.columns:
        raise ValueError(f"Target column '{target_col}' not present after preprocessing.")

    y = df_proc[target_col]
    X = df_proc.drop(columns=[target_col])

    # Convert object columns to dummies
    obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    # Drop any remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    return X, y, original_shape, df_proc.shape

# -------------------------
# Streamlit UI & Logic
# -------------------------
st.set_page_config(layout="wide", page_title="AutoEDA + ML")
st.title("📊 AutoEDA Explorer — now with ML")

with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.header("2. EDA / ML Settings")
    missing_strategy = st.selectbox("Missing value handling for ML", ("No action (report only)", "Drop rows with NA", "Simple Impute (mean/mode)"))
    test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=30, step=5)
    random_state = st.number_input("Random state (seed)", min_value=0, value=42, step=1)
    run_eda = st.button("Run EDA")
    run_ml = st.button("Run ML")

    st.markdown("---")
    st.write("Notes:")
    st.info("ML will only run when a target column is selected and 'Run ML' is clicked.")

# Load file
if uploaded_file is not None:
    if "uploaded_file_name" not in st.session_state or uploaded_file.name != st.session_state.uploaded_file_name:
        df_temp = load_data(uploaded_file)
        if not df_temp.empty:
            st.session_state.df = df_temp
            st.session_state.uploaded_file_name = uploaded_file.name
            # reset plot counter so keys are stable after new upload
            st.session_state.plot_key = 0
            st.experimental_rerun()

# Basic EDA
if not st.session_state.df.empty and run_eda:
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Data Preview")
    st.dataframe(st.session_state.df.head())

    st.subheader("Dataset Information")
    buffer = io.StringIO()
    st.session_state.df.info(buf=buffer)
    s = buffer.getvalue()
    n_rows, n_cols = st.session_state.df.shape
    n_missing_total = st.session_state.df.isnull().sum().sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", n_rows)
    col2.metric("Columns", n_cols)
    col3.metric("Total Missing Values", n_missing_total)

    st.markdown("**Detailed DataFrame Info:**")
    st.code(s, language="text")

    st.subheader("Missing Values by Column")
    missing_by_col = st.session_state.df.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    if not missing_by_col.empty:
        st.dataframe(missing_by_col.to_frame("missing_count"))
    else:
        st.write("No missing values detected.")

    st.subheader("Descriptive Statistics")
    st.dataframe(st.session_state.df.describe(include='all').transpose())

    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select a feature to visualize its distribution", options=(numeric_cols + categorical_cols))
    if feature_to_plot:
        if feature_to_plot in numeric_cols:
            fig = px.histogram(st.session_state.df, x=feature_to_plot, marginal="box")
        else:
            fig = px.histogram(st.session_state.df, x=feature_to_plot, color=feature_to_plot)
        st.plotly_chart(fig, use_container_width=True, key=next_plot_key())

    st.subheader("Correlation Analysis")
    if len(numeric_cols) > 1:
        corr = st.session_state.df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
        fig.update_layout(title="Correlation Matrix of Numeric Features")
        st.plotly_chart(fig, use_container_width=True, key=next_plot_key())
    else:
        st.warning("Not enough numeric columns for a correlation matrix.")

# ML Section
if not st.session_state.df.empty and run_ml:
    st.header("Machine Learning")

    # choose target
    columns = st.session_state.df.columns.tolist()
    target_col = st.selectbox("Select Target Variable (Y)", columns, key="ml_target")
    problem_type = st.selectbox("Problem Type", ("Auto-detect", "Classification", "Regression"), key="ml_problem")

    # Infer problem type if requested
    if problem_type == "Auto-detect":
        # If target is numeric with many unique values -> regression, else classification
        if pd.api.types.is_numeric_dtype(st.session_state.df[target_col]) and st.session_state.df[target_col].nunique() > 20:
            problem_type_real = "Regression"
        else:
            problem_type_real = "Classification"
    else:
        problem_type_real = problem_type

    st.write(f"Determined problem type: **{problem_type_real}**")

    try:
        X, y, orig_shape, new_shape = preprocess_for_ml(st.session_state.df, target_col, missing_strategy)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    st.write(f"Rows/Cols before preprocessing: {orig_shape} → after preprocessing: {new_shape}")
    if X.shape[0] == 0:
        st.error("No rows left after preprocessing. Try a different missing value strategy or choose a different target.")
        st.stop()

    # Train/test split
    test_frac = int(test_size) / 100.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=int(random_state), stratify=(y if problem_type_real == "Classification" else None))

    st.subheader("Training Models (baseline) — this may take a moment")

    results = []
    trained_models = {}

    if problem_type_real == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=int(random_state)),
            "Gradient Boosting": GradientBoostingClassifier(random_state=int(random_state))
        }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                # Try ROC AUC when possible (binary or probability-available)
                try:
                    if len(np.unique(y_test)) == 2:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        roc_auc = roc_auc_score(y_test, y_proba)
                    else:
                        # multiclass: use one-vs-rest average
                        y_proba = model.predict_proba(X_test)
                        roc_auc = roc_auc_score(pd.get_dummies(y_test), y_proba, average="weighted", multi_class="ovr")
                except Exception:
                    roc_auc = None

                results.append({
                    "Model": name,
                    "Accuracy": round(acc, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1": round(f1, 4),
                    "ROC AUC": round(roc_auc, 4) if roc_auc is not None else None
                })
                trained_models[name] = model
            except Exception as e:
                results.append({"Model": name, "Error": str(e)})

    else:  # Regression
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=int(random_state)),
            "Gradient Boosting": GradientBoostingRegressor(random_state=int(random_state))
        }

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                results.append({
                    "Model": name,
                    "R2": round(r2, 4),
                    "RMSE": round(rmse, 4),
                    "MAE": round(mae, 4)
                })
                trained_models[name] = model
            except Exception as e:
                results.append({"Model": name, "Error": str(e)})

    results_df = pd.DataFrame(results)
    st.subheader("Model Results")
    st.dataframe(results_df)

    # Show best model info
    try:
        if problem_type_real == "Classification":
            # prefer model with best F1 (or Accuracy if F1 missing)
            best = max([r for r in results if "F1" in r and r.get("F1") is not None], key=lambda x: x.get("F1", 0))
        else:
            best = max([r for r in results if "R2" in r], key=lambda x: x.get("R2", -999))
        st.success(f"Best Model: {best['Model']}")
        st.json(best)
    except Exception:
        st.info("Could not determine best model automatically.")

    # Optional: cross-validation
    st.subheader("Cross-validation (optional)")
    if st.button("Run 5-fold CV"):
        cv_scores = {}
        if problem_type_real == "Classification":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(random_state))
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=int(random_state))

        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=("r2" if problem_type_real == "Regression" else "accuracy"))
                cv_scores[name] = {"mean_score": float(np.mean(scores)), "std": float(np.std(scores))}
            except Exception as e:
                cv_scores[name] = {"error": str(e)}
        st.write(cv_scores)

    st.info("Tip: export trained models using joblib/pickle if you want to persist them locally.")

# If nothing uploaded
if st.session_state.df.empty:
    st.info("Awaiting for a CSV file to be uploaded.")
    st.markdown("""
    ### How to use this application:
    1.  **Upload your dataset** in CSV format using the sidebar.
    2.  Choose a **missing value strategy**, **test size**, and **random state**.
    3.  Click **Run EDA** to inspect data or **Run ML** to train baseline models.
    """)
