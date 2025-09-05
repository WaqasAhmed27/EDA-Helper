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

# Page configuration with custom styling
st.set_page_config(
    layout="wide", 
    page_title="AutoEDA & ML Pipeline",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .step-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "target_variable" not in st.session_state:
    st.session_state["target_variable"] = None
if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 1

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
    with st.spinner("🤖 Generating AI-powered recommendations..."):
        time.sleep(1)  # Visual feedback
        
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
    st.markdown('<div class="step-header"><h2>🧠 AI-Powered Recommendations</h2></div>', unsafe_allow_html=True)
    
    if not recommendations:
        st.warning("⚠️ No recommendations could be generated. Please check the API response.")
        return

    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Feature Engineering", "📊 Visualizations", "🎯 Model Selection", "💡 Missing Values"])
    
    with tab4:
        if (recommendations and "missing_value_handling" in recommendations 
            and "gemini_recommendation" in recommendations["missing_value_handling"]):
            st.markdown("### 🔍 Missing Value Handling Recommendation")
            st.info(recommendations["missing_value_handling"]["gemini_recommendation"])
        else:
            st.info("No specific missing value recommendations provided.")
    
    with tab1:
        if "feature_engineering" in recommendations and recommendations["feature_engineering"]:
            st.markdown("### 🛠️ Suggested Feature Engineering")
            
            for i, rec in enumerate(recommendations["feature_engineering"]):
                with st.expander(f"💡 {rec['title']}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {rec['description']}")
                        st.code(rec["code_snippet"], language='python')
                    
                    with col2:
                        if rec["type"] == "new_feature":
                            if st.button(f"✨ Apply Feature", key=f"add_feature_{i}", type="secondary"):
                                try:
                                    local_vars = {"df": st.session_state.df}
                                    exec(rec["code_snippet"], globals(), local_vars)
                                    st.session_state.df = local_vars["df"]
                                    st.success(f"✅ Successfully added '{rec['title']}'!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                        else:
                            st.info("💡 This is a transformation. Use the code snippet manually.")
        else:
            st.info("No feature engineering recommendations provided.")

    with tab2:
        if "visualizations" in recommendations and recommendations["visualizations"]:
            st.markdown("### 📈 Additional Visualizations")
            
            for i, viz in enumerate(recommendations["visualizations"]):
                with st.expander(f"📊 {viz['title']}", expanded=False):
                    st.markdown(f"**Insight:** {viz['description']}")
                    
                    try:
                        if viz['plot_type'] == "correlation_heatmap":
                            df_numeric = st.session_state.df.select_dtypes(include=np.number)
                            corr = df_numeric.corr()
                            fig = go.Figure(data=go.Heatmap(
                                z=corr.values, 
                                x=corr.columns, 
                                y=corr.columns, 
                                colorscale='Viridis',
                                text=np.round(corr.values, 2),
                                texttemplate="%{text}",
                                textfont={"size":10}
                            ))
                            fig.update_layout(
                                title="Correlation Matrix of Numeric Features",
                                width=600,
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        elif viz['plot_type'] == "box_plot":
                            if 'x_axis' in viz:
                                fig = px.box(st.session_state.df, x=st.session_state.target_variable, y=viz['x_axis'], 
                                           title=f"Box Plot of {viz['x_axis']} by {st.session_state.target_variable}")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Visualization recommendation missing 'x_axis' field for box plot.")

                        elif viz['plot_type'] == "violin_plot":
                            if 'x_axis' in viz:
                                fig = px.violin(st.session_state.df, x=st.session_state.target_variable, y=viz['x_axis'], 
                                             title=f"Violin Plot of {viz['x_axis']} by {st.session_state.target_variable}")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Visualization recommendation missing 'x_axis' field for violin plot.")

                    except Exception as e:
                        st.error(f"Could not generate plot: {e}")
                    
                    with st.expander("View Code", expanded=False):
                        st.code(viz["code_snippet"], language='python')
        else:
            st.info("No visualization recommendations provided.")

    with tab3:
        if "model_selection" in recommendations and recommendations["model_selection"]:
            st.markdown("### 🎯 Model Selection Guidance")
            st.markdown(recommendations["model_selection"])
        else:
            st.info("No model selection recommendations provided.")

# --- Main Application ---

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 AutoEDA & ML Pipeline</h1>
    <p>Powered by AI-driven insights and automated machine learning</p>
</div>
""", unsafe_allow_html=True)

# Progress indicator
progress_steps = ["Upload Data", "Configure", "Analyze", "Predict", "Optimize"]
current_step = st.session_state.get("current_step", 1)

col1, col2, col3, col4, col5 = st.columns(5)
for i, step in enumerate(progress_steps, 1):
    with [col1, col2, col3, col4, col5][i-1]:
        if i <= current_step:
            st.markdown(f"✅ **{step}**")
        else:
            st.markdown(f"⏳ {step}")

st.markdown("---")

# Use session state to store dataframe and other variables
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

# Sidebar with improved styling
with st.sidebar:
    st.markdown("## 📋 Configuration Panel")
    
    # Step 1: Data Upload
    with st.expander("📁 Step 1: Upload Data", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose your CSV file", 
            type=["csv"],
            help="Upload a CSV file with your dataset. Various delimiters are supported."
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
            st.session_state.current_step = max(st.session_state.current_step, 2)

    # Step 2: Configuration
    missing_cols = []
    missing_strategy = None
    
    if uploaded_file is not None or not st.session_state.df.empty:
        with st.expander("⚙️ Step 2: Configure Analysis", expanded=not st.session_state.df.empty):
            if not st.session_state.df.empty:
                missing_cols = st.session_state.df.columns[st.session_state.df.isnull().any()].tolist()
                
                if missing_cols:
                    st.markdown("### 🔧 Missing Value Strategy")
                    missing_strategy = st.radio(
                        "How to handle missing values?",
                        ("Impute", "Drop"),
                        key="missing_strategy",
                        help="Impute: Fill missing values with statistical measures. Drop: Remove rows with missing values."
                    )
                    st.info(f"Found missing values in: {', '.join(missing_cols[:3])}{' and more...' if len(missing_cols) > 3 else ''}")

    # Handle file upload
    if uploaded_file is not None:
        if "uploaded_file_name" not in st.session_state or uploaded_file.name != st.session_state.uploaded_file_name:
            with st.spinner("Loading data..."):
                df_temp = load_data(uploaded_file)
                if not df_temp.empty:
                    st.session_state.df = df_temp
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.recommendations = None
                    st.rerun()

    # Step 3: Target Variable Selection
    if not st.session_state.df.empty:
        with st.expander("🎯 Step 3: Select Target", expanded=True):
            columns = st.session_state.df.columns.tolist()
            target_variable = st.selectbox(
                "Target Variable (what to predict)", 
                columns, 
                key="target_variable",
                help="Select the column you want to predict"
            )
            problem_type = st.selectbox(
                "Problem Type", 
                ["Classification", "Regression"], 
                key="problem_type",
                help="Classification: Predicting categories. Regression: Predicting numbers."
            )
            
            if target_variable:
                st.session_state.current_step = max(st.session_state.current_step, 3)
            
            st.markdown("---")
            run_button = st.button(
                "🚀 Run Complete Analysis", 
                type="primary",
                use_container_width=True,
                disabled=not target_variable
            )
            
            if run_button:
                st.session_state.current_step = 4
    else:
        run_button = False
        target_variable = None
        problem_type = None

# Main content area
if st.session_state.df.empty:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 👋 Welcome to AutoEDA & ML Pipeline!
        
        **Get started in 3 simple steps:**
        
        1. 📁 **Upload** your CSV dataset
        2. 🎯 **Select** your target variable
        3. 🚀 **Run** automated analysis
        
        **Features:**
        - 📊 Automated Exploratory Data Analysis
        - 🤖 AI-powered recommendations
        - 🔧 Interactive feature engineering
        - 📈 Multiple ML model comparison
        - 💡 Intelligent missing value handling
        """)
        
        # Sample data option
        if st.button("📥 Try with Sample Data", type="secondary"):
            # Create sample data
            np.random.seed(42)
            sample_data = pd.DataFrame({
                'feature1': np.random.normal(100, 15, 200),
                'feature2': np.random.uniform(0, 50, 200),
                'feature3': np.random.choice(['A', 'B', 'C'], 200),
                'target': np.random.choice([0, 1], 200)
            })
            # Add some missing values
            sample_data.loc[10:15, 'feature1'] = np.nan
            st.session_state.df = sample_data
            st.session_state.current_step = 2
            st.rerun()

elif not st.session_state.df.empty and (run_button or 'recommendations' in st.session_state):
    # Main analysis section
    st.session_state.current_step = max(st.session_state.current_step, 3)
    
    # Data Overview Section
    st.markdown('<div class="step-header"><h2>📊 Dataset Overview</h2></div>', unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    n_rows, n_cols = st.session_state.df.shape
    n_missing = st.session_state.df.isnull().sum().sum()
    n_numeric = len(st.session_state.df.select_dtypes(include=np.number).columns)
    
    with col1:
        st.metric("📋 Rows", f"{n_rows:,}")
    with col2:
        st.metric("📊 Columns", n_cols)
    with col3:
        st.metric("❓ Missing Values", n_missing)
    with col4:
        st.metric("🔢 Numeric Features", n_numeric)

    # Data preview with improved styling
    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
    
    with st.expander("📈 Statistical Summary", expanded=False):
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
    
    # EDA Section
    st.markdown('<div class="step-header"><h2>🔍 Exploratory Data Analysis</h2></div>', unsafe_allow_html=True)
    
    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = st.session_state.df.select_dtypes(include='object').columns.tolist()

    # Target variable analysis
    if 'target_variable' in st.session_state and st.session_state.target_variable:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🎯 Target: {st.session_state.target_variable}")
            if st.session_state.problem_type == "Classification":
                fig = px.histogram(
                    st.session_state.df, 
                    x=st.session_state.target_variable, 
                    color=st.session_state.target_variable,
                    title="Target Distribution"
                )
            else:
                fig = px.histogram(
                    st.session_state.df, 
                    x=st.session_state.target_variable, 
                    marginal="box",
                    title="Target Distribution"
                )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📊 Feature Distribution")
            feature_to_plot = st.selectbox(
                "Select feature to analyze", 
                numeric_cols + categorical_cols,
                key="feature_selector"
            )
            if feature_to_plot:
                if feature_to_plot in numeric_cols:
                    fig = px.histogram(
                        st.session_state.df, 
                        x=feature_to_plot, 
                        marginal="box",
                        title=f"Distribution of {feature_to_plot}"
                    )
                else:
                    fig = px.histogram(
                        st.session_state.df, 
                        x=feature_to_plot, 
                        color=feature_to_plot,
                        title=f"Distribution of {feature_to_plot}"
                    )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    if len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Analysis")
        corr = st.session_state.df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, 
            x=corr.columns, 
            y=corr.columns, 
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size":10}
        ))
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # ML Section
    st.markdown('<div class="step-header"><h2>🤖 Machine Learning Analysis</h2></div>', unsafe_allow_html=True)
    
    df_ml = st.session_state.df.copy()
    
    # Handle missing values based on user selection
    if 'missing_strategy' in st.session_state and missing_cols:
        with st.spinner("Handling missing values..."):
            if st.session_state.missing_strategy == "Impute":
                for col in missing_cols:
                    if pd.api.types.is_numeric_dtype(df_ml[col]):
                        strat = "mean"
                    else:
                        strat = "most_frequent"
                    imputer = SimpleImputer(strategy=strat)
                    df_ml[[col]] = imputer.fit_transform(df_ml[[col]])
                st.success(f"✅ Imputed missing values in {len(missing_cols)} columns")
            elif st.session_state.missing_strategy == "Drop":
                original_rows = len(df_ml)
                df_ml = df_ml.dropna()
                st.warning(f"⚠️ Dropped {original_rows - len(df_ml)} rows with missing values")

    # Encode categorical variables
    for col in df_ml.columns:
        if df_ml[col].dtype == 'object' and col != st.session_state.target_variable:
            try:
                df_ml[col] = pd.to_numeric(df_ml[col])
            except (ValueError, TypeError):
                df_ml = pd.get_dummies(df_ml, columns=[col], drop_first=True)

    if st.session_state.target_variable not in df_ml.columns:
        st.error(f"❌ Target variable '{st.session_state.target_variable}' was removed during preprocessing.")
    else:
        X = df_ml.drop(st.session_state.target_variable, axis=1)
        y = df_ml[st.session_state.target_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.subheader("🏆 Model Performance Comparison")
        
        # Progress bar for model training
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []

        if st.session_state.problem_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }

            for i, (name, model) in enumerate(models.items()):
                status_text.text(f"Training {name}...")
                progress_bar.progress((i + 1) / len(models))
                
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    results.append({"Model": name, "Accuracy": f"{accuracy:.4f}", "Score": accuracy})
                except Exception as e:
                    results.append({"Model": name, "Accuracy": "Error", "Score": 0, "Error": str(e)})

            # Display results with styling
            results_df = pd.DataFrame(results)
            st.dataframe(results_df[["Model", "Accuracy"]], use_container_width=True)

            # Highlight best model
            best_model = max(results, key=lambda x: x.get("Score", 0))
            st.success(f"🏆 **Best Model:** {best_model['Model']} with Accuracy = {best_model.get('Score', 0):.4f}")

            metrics_text = results_df.to_string(index=False)

        else:  # Regression
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42)
            }

            for i, (name, model) in enumerate(models.items()):
                status_text.text(f"Training {name}...")
                progress_bar.progress((i + 1) / len(models))
                
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    results.append({"Model": name, "R²": f"{r2:.4f}", "Score": r2})
                except Exception as e:
                    results.append({"Model": name, "R²": "Error", "Score": -999, "Error": str(e)})

            # Display results with styling
            results_df = pd.DataFrame(results)
            st.dataframe(results_df[["Model", "R²"]], use_container_width=True)

            # Highlight best model
            best_model = max(results, key=lambda x: x.get("Score", -999))
            st.success(f"🏆 **Best Model:** {best_model['Model']} with R² = {best_model.get('Score', 0):.4f}")

            metrics_text = results_df.to_string(index=False)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Generate recommendations
        if run_button:
            st.session_state.current_step = 5
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
            display_and_interact(st.session_state.recommendations)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Made with ❤️ using Streamlit • Enhanced with AI-powered insights
</div>
""", unsafe_allow_html=True)
