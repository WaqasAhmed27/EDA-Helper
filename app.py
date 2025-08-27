import streamlit as st
import pandas as pd
import numpy as np
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

st.set_page_config(layout="wide", page_title="AutoEDA")
st.title("📊 AutoEDA Explorer")

# Use session state to store dataframe
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.header("2. Configure EDA")

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
        run_button = st.button("Run EDA", type="primary")
    else:
        run_button = False
        target_variable = None

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
        if st.session_state.target_variable in numeric_cols:
            fig = px.histogram(st.session_state.df, x=st.session_state.target_variable, marginal="box")
        else:
            fig = px.histogram(st.session_state.df, x=st.session_state.target_variable, color=st.session_state.target_variable)
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

else:
    st.info("Awaiting for a CSV file to be uploaded and EDA to be run.")
    st.markdown("""
    ### How to use this application:
    1.  **Upload your dataset** in CSV format using the sidebar.
    2.  **Select your target variable** (optional).
    3.  Click the **'Run EDA'** button.

    The application will then perform automated EDA with descriptive statistics, distributions, and correlation analysis.
    """)
