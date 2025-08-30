# 📊 EDA Helper

An interactive **Streamlit** application that automates **Exploratory Data Analysis (EDA)** and builds baseline **Machine Learning models** (Classification & Regression) with **AI-powered recommendations** from Google Gemini.

This tool is designed to **help you quickly explore datasets, test ML models, and get expert recommendations** without writing code.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-🚀%20Try%20Now-brightgreen.svg)](https://auto-eda-ai.streamlit.app/)

---

## 🌐 Live Demo

**Try the app now**: [https://auto-eda-ai.streamlit.app/](https://auto-eda-ai.streamlit.app/)

No installation required! Upload your CSV file and start exploring your data with AI-powered insights immediately.

---

## ✨ Features

### 🧠 **AI-Powered Intelligence**
* **Gemini Integration** → Get expert recommendations for feature engineering, visualizations, and model selection
* **Smart Missing Value Strategy** → AI suggests optimal handling approaches for your specific dataset
* **Interactive Feature Creation** → Apply recommended feature engineering with one click
* **Dynamic Visualization Suggestions** → Get custom plot recommendations based on your data patterns

### 📥 **Data Management**
* **Automated Data Loading** → Handles CSV files with multiple delimiters (`,`, `;`, `\t`, `|`)
* **Data Preview & Summary** → Shape, missing values, and detailed dataframe info
* **Missing Value Handling** → Choose to **impute (mean/mode)** or **drop** missing values
* **Automatic Encoding** → Categorical variables are handled seamlessly

### 📈 **Exploratory Data Analysis**
* **Descriptive Statistics** → Summary of numeric features
* **Visualization Suite**:
  * Target variable distribution (histograms/boxplots)
  * Feature distributions (numeric & categorical)
  * Correlation heatmaps
  * Interactive plots with Plotly

### 🤖 **Machine Learning**
* **Model Training**:
  * **Classification**: Logistic Regression, Random Forest, Gradient Boosting
  * **Regression**: Linear Regression, Random Forest, Gradient Boosting
* **Enhanced Model Evaluation**:
  * Classification → Accuracy, Precision, Recall, F1-Score
  * Regression → R² score, MAE, MSE, RMSE
* **Best Model Selection** → Automatically identifies the top-performing model
* **AI Model Recommendations** → Get expert advice on model selection and hyperparameter tuning

---

## 🚀 Quick Start

### Option 1: Use the Live Demo (Recommended)
**🌐 [Try EDA Helper Online](https://auto-eda-ai.streamlit.app/)**

Simply visit the link above and start uploading your datasets immediately - no setup required!

### Option 2: Run Locally

#### Prerequisites
- Python 3.7 or higher
- pip package manager

#### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/eda-helper.git
cd eda-helper
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

#### Dependencies
```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
plotly>=5.15.0
seaborn>=0.11.0
matplotlib>=3.5.0
xgboost>=1.7.0
requests>=2.28.0
```

#### Configuration

**Important**: This app requires a Google Gemini API key for AI-powered recommendations.

1. **Get a Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key for the next step

2. **Set up Streamlit Secrets**:
   Create a `.streamlit/secrets.toml` file in your project root:
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key-here"
   ```

---

## 📖 How to Use

### Step-by-Step Guide

**Using the Live Demo** or **Local Installation**:

1. **📁 Upload Your Dataset**
   - Use the sidebar to upload a CSV file
   - The app automatically detects the delimiter

2. **🎯 Configure Analysis**
   - Select your target variable from the dropdown
   - Choose problem type: Classification or Regression
   - **Configure missing value handling**: Choose between imputation or dropping missing values

3. **🔍 Explore Your Data**
   - View data summary and statistics
   - Analyze feature distributions
   - Examine correlations between variables

4. **🤖 Train Models & Get AI Recommendations**
   - Click "Run Analysis & Prediction"
   - Compare model performances
   - **Receive AI-powered recommendations** for:
     - Missing value handling strategies
     - Feature engineering opportunities
     - Additional visualizations
     - Model selection advice

5. **⚡ Apply Recommendations**
   - **One-click feature creation** from AI suggestions
   - View recommended visualizations
   - Implement suggested improvements

6. **📊 Review Results**
   - Examine model metrics
   - Analyze feature importance (where available)
   - Export results for further use

---

## 💡 Example Workflow

**Dataset**: Titanic passenger data (`titanic.csv`)

1. Upload `titanic.csv`
2. Select `Survived` as target variable
3. Choose **Classification** problem type
4. Configure missing value handling (e.g., "Impute")
5. Run analysis

**Results**:
- 📈 Target distribution: 38% survived, 62% perished
- 🔗 Strong correlations: Passenger class, gender, age
- 🏆 **Best Model**: Random Forest with **85% accuracy**

**AI Recommendations**:
- 🧠 **Missing Values**: "Use median imputation for Age, mode for Embarked"
- ⚡ **Feature Engineering**: "Create family_size = SibSp + Parch + 1"
- 📊 **Visualizations**: "Add violin plot of Age by Survived"
- 🎯 **Model Selection**: "Try XGBoost with hyperparameter tuning"

**Model Performance**:
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.82 | 0.78 | 0.74 | 0.76 |
| Random Forest | **0.85** | **0.82** | **0.79** | **0.80** |
| Gradient Boosting | 0.84 | 0.80 | 0.77 | 0.78 |

---

## 📁 Project Structure

```
eda-helper/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── .streamlit/              # Streamlit configuration
│   └── secrets.toml        # API keys (not committed)
├── data/                    # Sample datasets (optional)
│   └── sample.csv
└── utils/                   # Helper functions (if any)
    └── preprocessing.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Seaborn, Matplotlib |
| **Machine Learning** | Scikit-learn, XGBoost |
| **AI Recommendations** | Google Gemini 2.5 Flash |
| **Development** | Python 3.7+ |

---

## 🔮 Roadmap

- [ ] **Advanced Models**: XGBoost, LightGBM, Neural Networks
- [x] **AI-Powered Recommendations**: Google Gemini integration ✅
- [x] **Smart Missing Value Handling**: AI-suggested strategies ✅
- [x] **Interactive Feature Engineering**: One-click feature creation ✅
- [ ] **Model Deployment**: Export trained models for production use
- [ ] **Time Series Support**: Specialized analysis for temporal data
- [ ] **Advanced Visualizations**: 3D plots, interactive dashboards
- [ ] **Data Export**: Download processed datasets and results
- [ ] **Hyperparameter Tuning**: Automated optimization with AI suggestions
- [ ] **Model Explainability**: SHAP values and feature importance analysis

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Submit** a pull request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/your-username/eda-helper.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Gemini API key in .streamlit/secrets.toml
echo 'GEMINI_API_KEY = "your-api-key-here"' > .streamlit/secrets.toml
```

---

## 🐛 Known Issues & Limitations

- Large datasets (>150MB) may cause performance issues
- Currently supports only CSV files
- Limited to tabular data (no support for images, text, etc.)
- **Gemini API calls may have rate limits** - recommendations might take a moment to generate
- AI recommendations require internet connection
- **API costs**: Gemini API usage may incur charges based on Google's pricing

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/waqasahmed27/eda-helper/issues)
- **Discussions**: [GitHub Discussions](https://github.com/waqasahmed27/eda-helper/discussions)
- **Email**: your.email@example.com

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Streamlit** team for the amazing framework
- **Scikit-learn** contributors for robust ML tools
- **Plotly** for interactive visualizations
- **Google Gemini** for AI-powered recommendations
- **XGBoost** developers for gradient boosting capabilities
- Open source community for inspiration and support

---

## ⭐ Show Your Support

If this project helped you, please consider giving it a ⭐ on GitHub!
