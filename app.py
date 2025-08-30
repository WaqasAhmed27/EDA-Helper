import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score 
from xgboost import XGBClassifier, XGBRegressor  
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import plotly.express as px
import plotly.graph_objects as go
import io
import requests
import json
import time
from sklearn.impute import SimpleImputer

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

def get_gemini_recommendations(df, problem_type, target_variable, metrics, missing_strategy=None, missing_cols=None):
    st.info("🤖 Cooking up some recommendations... this may take a moment.")
    
    # Add missing value context to the prompt
    missing_context = ""
    if missing_cols:
        missing_context = f"""
        Columns with missing values: {missing_cols}
        User-selected missing value handling: {missing_strategy}
        Please recommend the best missing value handling strategy for this dataset and context, and justify your choice.
        """

    prompt = f"""
    Analyze the following dataset and machine learning model performance to provide expert recommendations.

    Dataset Preview (first 5 rows):
    ```
    {df.head().to_string()}
    ```
    
    Analysis Context:
    - Problem Type: {problem_type}
    - Target Variable: '{target_variable}'
    - Initial Model Performance:
    {metrics}
    {missing_context}

    Your Task:
    Based on the data, provide specific, actionable recommendations in a JSON object with the following structure:
    {{
      "missing_value_handling": {{
        "gemini_recommendation": "A markdown string with your recommended missing value handling strategy and justification."
      }},
      "feature_engineering": [
        {{
          "title": "A short, descriptive title",
          "description": "A description of the new feature or transformation.",
          "type": "new_feature" or "transformation",
          "code_snippet": "A Python code snippet to create the feature (e.g., df['new_col'] = ...)."
        }}
      ],
      "visualizations": [
        {{
          "title": "Title of the plot",
          "description": "What this plot will reveal.",
          "plot_type": "correlation_heatmap" or "box_plot" or "violin_plot",
          "code_snippet": "The Python code snippet to generate the plot using Plotly Express or Plotly Graph Objects."
        }}
      ],
      "model_selection": "A markdown string with model recommendations and justifications."
    }}
    Do not include any text outside of the JSON object in your response. Ensure the JSON is valid.
    """

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
        
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}]
            }]
        }
        
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        result = response.json()
        
        recommendations_str = result['candidates'][0]['content']['parts'][0]['text']
        # The API sometimes wraps JSON in markdown code blocks. We need to handle this.
        if recommendations_str.strip().startswith('```json'):
            recommendations_str = recommendations_str.strip()[7:-3].strip()
        
        recommendations = json.loads(recommendations_str)
        return recommendations

    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        st.error(f"Error fetching or parsing recommendations: {e}")
        return None

def display_and_interact(recommendations):
    st.header("🧠 Gemini-Powered Recommendations")
    if not recommendations:
        st.warning("No recommendations could be generated. Please check the API response.")
        return

    # Feature Engineering Section
    st.subheader("1. Feature Engineering")
    if "feature_engineering" in recommendations and recommendations["feature_engineering"]:
        for i, rec in enumerate(recommendations["feature_engineering"]):
            with st.expander(f"**{rec['title']}**"):
                st.markdown(f"**Description:** {rec['description']}")
                st.code(rec["code_snippet"], language='python')
                
                # Use a unique key for each button to avoid conflicts
                if rec["type"] == "new_feature":
                    if st.button(f"Add '{rec['title']}' to Data", key=f"add_feature_{i}"):
                        try:
                            local_vars = {"df": st.session_state.df}
                            exec(rec["code_snippet"], globals(), local_vars)
                            st.session_state.df = local_vars["df"]
                            st.success(f"Successfully added '{rec['title']}'!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error executing code snippet: {e}")
                else:
                    st.info("Transformations (e.g., scaling) are not applied with a button. You can use the code snippet to apply it manually.")
    else:
        st.info("No feature engineering recommendations were provided.")

    # Visualizations Section
    st.subheader("2. Additional Visualizations")
    if "visualizations" in recommendations and recommendations["visualizations"]:
        for i, viz in enumerate(recommendations["visualizations"]):
            with st.expander(f"**{viz['title']}**"):
                st.markdown(f"**Description:** {viz['description']}")
                try:
                    # Dynamically generate and display the plot
                    st.write("### Plot:")
                    if viz['plot_type'] == "correlation_heatmap":
                        df_numeric = st.session_state.df.select_dtypes(include=np.number)
                        corr = df_numeric.corr()
                        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
                        fig.update_layout(title="Correlation Matrix of Numeric Features")
                        st.plotly_chart(fig, use_container_width=True, key=6)

                    elif viz['plot_type'] == "box_plot":
                        if 'x_axis' in viz:
                            fig = px.box(st.session_state.df, x=st.session_state.target_variable, y=viz['x_axis'], 
                                         title=f"Box Plot of {viz['x_axis']} by {st.session_state.target_variable}")
                            st.plotly_chart(fig, use_container_width=True, key=7)
                        else:
                            st.warning("Visualization recommendation missing 'x_axis' field for box plot.")

                    elif viz['plot_type'] == "violin_plot":
                        if 'x_axis' in viz:
                            fig = px.violin(st.session_state.df, x=st.session_state.target_variable, y=viz['x_axis'], 
                                         title=f"Violin Plot of {viz['x_axis']} by {st.session_state.target_variable}")
                            st.plotly_chart(fig, use_container_width=True, key=8)
                        else:
                            st.warning("Visualization recommendation missing 'x_axis' field for violin plot.")

                    else:
                        st.warning(f"Plot type '{viz['plot_type']}' not recognized.")
                except Exception as e:
                    st.error(f"Could not generate plot: {e}")
                
                # Show the code snippet
                st.write("### Code Snippet:")
                st.code(viz["code_snippet"], language='python')
    else:
        st.info("No visualization recommendations were provided.")

    # Model Selection Section
    st.subheader("3. Model Selection")
    if "model_selection" in recommendations and recommendations["model_selection"]:
        st.markdown(recommendations["model_selection"])
    else:
        st.info("No model selection recommendations were provided.")

# --- Streamlit UI and Logic ---

st.set_page_config(layout="wide", page_title="AutoEDA & ML Pipeline")
st.title("🚀 AutoEDA & ML Prediction Pipeline")
st.write("with AI-Powered Recommendations")

# Use session state to store dataframe and other variables
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.header("2. Configure Pipeline")
    # --- Updated: Missing Value Handling ---
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
        # Only load if it's a new upload (not already stored)
        if "uploaded_file_name" not in st.session_state or uploaded_file.name != st.session_state.uploaded_file_name:
            df_temp = load_data(uploaded_file)
            if not df_temp.empty:
                st.session_state.df = df_temp
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.recommendations = None
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

if not st.session_state.df.empty and (run_button or 'recommendations' in st.session_state):
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Data Preview")
    st.dataframe(st.session_state.df.head())

    st.subheader("Dataset Information")
    buffer = io.StringIO()
    st.session_state.df.info(buf=buffer)
    s = buffer.getvalue()
    n_rows, n_cols = st.session_state.df.shape
    n_missing = st.session_state.df.isnull().sum().sum()
    # --- Improved visual formatting ---
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
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{keyRand}")
            keyRand = keyRand + 1

        else:
            fig = px.histogram(st.session_state.df, x=st.session_state.target_variable, marginal="box")
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{keyRand}")
            keyRand = keyRand + 1


    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select a feature to visualize its distribution", numeric_cols + categorical_cols)
    if feature_to_plot:
        if feature_to_plot in numeric_cols:
            fig = px.histogram(st.session_state.df, x=feature_to_plot, marginal="box")
        else:
            fig = px.histogram(st.session_state.df, x=feature_to_plot, color=feature_to_plot)
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{keyRand}")
            keyRand = keyRand + 1

    st.subheader("Correlation Analysis")
    if len(numeric_cols) > 1:
        corr = st.session_state.df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
        fig.update_layout(title="Correlation Matrix of Numeric Features")
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{keyRand}")
        keyRand = keyRand + 1

    else:
        st.warning("Not enough numeric columns for a correlation matrix.")

st.header("Machine Learning Prediction")

df_ml = st.session_state.df.copy()
# --- Updated: Handle missing values based on user selection ---
if 'missing_strategy' in st.session_state and missing_cols:
    if st.session_state.missing_strategy == "Impute":
        # Automatically select strategy per column
        for col in missing_cols:
            if pd.api.types.is_numeric_dtype(df_ml[col]):
                strat = "mean"
            else:
                strat = "most_frequent"
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

    if not all(col in df_ml.columns for col in X.columns):
        st.error("Mismatch in columns after preprocessing. Please check your data.")
    else:
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

            # show best model details
            best_model = max(results, key=lambda x: x.get("Accuracy", 0))
            st.write(f"**Best Model:** {best_model['Model']} with Accuracy = {best_model.get('Accuracy', 0):.4f}")

            metrics_text = results_df.to_string(index=False)

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

            # show best model details
            best_model = max(results, key=lambda x: x.get("R²", -999))
            st.write(f"**Best Model:** {best_model['Model']} with R² = {best_model.get('R²', 0):.4f}")

            metrics_text = results_df.to_string(index=False)

        # ✅ Only generate recommendations when user clicks button
        if run_button:
            recommendations = get_gemini_recommendations(
                st.session_state.df, 
                st.session_state.problem_type, 
                st.session_state.target_variable, 
                metrics_text,
                missing_strategy=st.session_state.get("missing_strategy"),
                missing_cols=missing_cols
            )
            st.session_state.recommendations = recommendations

        if 'recommendations' in st.session_state:
            # --- New: Show Gemini's missing value handling recommendation ---
            if (
                st.session_state.recommendations
                and "missing_value_handling" in st.session_state.recommendations
                and "gemini_recommendation" in st.session_state.recommendations["missing_value_handling"]
            ):
                st.subheader("Gemini Recommendation for Missing Value Handling")
                st.markdown(st.session_state.recommendations["missing_value_handling"]["gemini_recommendation"])
            display_and_interact(st.session_state.recommendations)
        else:
            st.info("Awaiting for a CSV file to be uploaded and analysis to be run.")
            st.markdown("""
            ### How to use this application:
            1.  **Upload your dataset** in CSV format using the sidebar.
            2.  **Select your target variable** (the column you want to predict).
            3.  **Choose the problem type** (Classification or Regression).
            4.  Click the **'Run Analysis & Prediction'** button.
            
            The application will then perform an automated EDA, train a baseline model, and provide you with AI-powered recommendations to improve your results!
            """)
