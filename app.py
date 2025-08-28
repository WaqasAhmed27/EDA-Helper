import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score 
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import io

# Initialize session state
if "target_variable" not in st.session_state:
    st.session_state["target_variable"] = None

keyRand = 1

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

# --- Streamlit UI and Logic ---

st.set_page_config(layout="wide", page_title="AutoEDA & ML Pipeline")
st.title("🚀 AutoEDA & ML Prediction Pipeline")

# Use session state to store dataframe and other variables
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.header("2. Configure Pipeline")
    missing_cols = []
    missing_strategy = None
    if uploaded_file is not None or not st.session_state.df.empty:
        if not st.session_state.df.empty:
            missing_cols = st.session_state.df.columns[st.session_state.df.isnull().any()].tolist()
            if missing_cols:
                st.markdown("### Missing Value Handling")
                missing_strategy = st.radio(
                    "How would you like to handle missing values?",
                    ("Impute", "Drop"),
                    key="missing_strategy"
                )
            else:
                missing_strategy = None

    if uploaded_file is not None:
        if "uploaded_file_name" not in st.session_state or uploaded_file.name != st.session_state.uploaded_file_name:
            df_temp = load_data(uploaded_file)
            if not df_temp.empty:
                st.session_state.df = df_temp
                st.session_state.uploaded_file_name = uploaded_file.name
                st.rerun()

    if not st.session_state.df.empty:
        columns = st.session_state.df.columns.tolist()
        target_variable = st.selectbox("Select the Target Variable (Y)", columns, key="target_variable")
        problem_type = st.selectbox("Select Problem Type", ["Classification", "Regression"], key="problem_type")
        run_button = st.button("Run Analysis & Prediction", type="primary")
    else:
        run_button = False
        target_variable = None
        problem_type = None

if not st.session_state.df.empty and run_button:
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Data Preview")
    st.dataframe(st.session_state.df.head())

    st.subheader("Dataset Information")
    buffer = io.StringIO()
    st.session_state.df.info(buf=buffer)
    s = buffer.getvalue()
    n_rows, n_cols = st.session_state.df.shape
    n_missing = st.session_state.df.isnull().sum().sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", n_rows)
    col2.metric("Columns", n_cols)
    col3.metric("Total Missing Values", n_missing)
    st.markdown("**Detailed DataFrame Info:**")
    st.code(s, language="text")

    st.subheader("Descriptive Statistics")
    st.dataframe(st.session_state.df.describe())

    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = st.session_state.df.select_dtypes(include='object').columns.tolist()

    if 'target_variable' in st.session_state and st.session_state.target_variable:
        st.subheader(f"Distribution of Target Variable: '{st.session_state.target_variable}'")
        if st.session_state.problem_type == "Classification":
            fig = px.histogram(st.session_state.df, x=st.session_state.target_variable, color=st.session_state.target_variable)
        else:
            fig = px.histogram(st.session_state.df, x=st.session_state.target_variable, marginal="box")
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{keyRand}")
        keyRand += 1

    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select a feature to visualize its distribution", numeric_cols + categorical_cols)
    if feature_to_plot:
        if feature_to_plot in numeric_cols:
            fig = px.histogram(st.session_state.df, x=feature_to_plot, marginal="box")
        else:
            fig = px.histogram(st.session_state.df, x=feature_to_plot, color=feature_to_plot)
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{keyRand}")
        keyRand += 1

    st.subheader("Correlation Analysis")
    if len(numeric_cols) > 1:
        corr = st.session_state.df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
        fig.update_layout(title="Correlation Matrix of Numeric Features")
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{keyRand}")
        keyRand += 1
    else:
        st.warning("Not enough numeric columns for a correlation matrix.")

    # --- Machine Learning ---
    st.header("Machine Learning Prediction")

    df_ml = st.session_state.df.copy()
    if 'missing_strategy' in st.session_state and missing_cols:
        if st.session_state.missing_strategy == "Impute":
            for col in missing_cols:
                strat = "mean" if pd.api.types.is_numeric_dtype(df_ml[col]) else "most_frequent"
                imputer = SimpleImputer(strategy=strat)
                df_ml[[col]] = imputer.fit_transform(df_ml[[col]])
        elif st.session_state.missing_strategy == "Drop":
            df_ml = df_ml.dropna()

    for col in df_ml.columns:
        if df_ml[col].dtype == 'object' and col != st.session_state.target_variable:
            try:
                df_ml[col] = pd.to_numeric(df_ml[col])
            except (ValueError, TypeError):
                df_ml = pd.get_dummies(df_ml, columns=[col], drop_first=True)

    if st.session_state.target_variable not in df_ml.columns:
        st.error(f"Target variable '{st.session_state.target_variable}' was removed during preprocessing.")
    else:
        X = df_ml.drop(st.session_state.target_variable, axis=1)
        y = df_ml[st.session_state.target_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.subheader("Model Training and Evaluation")
        results = []

        if st.session_state.problem_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    results.append({"Model": name, "Accuracy": accuracy})
                except Exception as e:
                    results.append({"Model": name, "Error": str(e)})

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            best_model = max(results, key=lambda x: x.get("Accuracy", 0))
            st.write(f"**Best Model:** {best_model['Model']} with Accuracy = {best_model.get('Accuracy', 0):.4f}")

        else:  # Regression
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42)
            }

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    results.append({"Model": name, "R²": r2})
                except Exception as e:
                    results.append({"Model": name, "Error": str(e)})

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            best_model = max(results, key=lambda x: x.get("R²", -999))
            st.write(f"**Best Model:** {best_model['Model']} with R² = {best_model.get('R²', 0):.4f}")

else:
    st.info("Awaiting for a CSV file to be uploaded and analysis to be run.")
    st.markdown("""
    ### How to use this application:
    1.  **Upload your dataset** in CSV format using the sidebar.
    2.  **Select your target variable** (the column you want to predict).
    3.  **Choose the problem type** (Classification or Regression).
    4.  Click the **'Run Analysis & Prediction'** button.

    The application will then perform an automated EDA, train baseline models, and show you the results.
    """)
