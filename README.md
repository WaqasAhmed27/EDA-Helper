---

# 📊 EDA Helper

An interactive **Streamlit** application that automates **Exploratory Data Analysis (EDA)** and builds baseline **Machine Learning models** (Classification & Regression) with minimal user effort.

This tool is designed to **help you quickly explore datasets and test ML models** without writing code.

---

## ✨ Features

* **Automated Data Loading** → Handles CSV files with multiple delimiters (`,`, `;`, `\t`, `|`).
* **Data Preview & Summary** → Shape, missing values, and detailed dataframe info.
* **Descriptive Statistics** → Summary of numeric features.
* **Visualization Suite**:

  * Target variable distribution (histograms/boxplots)
  * Feature distributions (numeric & categorical)
  * Correlation heatmaps
* **Missing Value Handling** → Choose to **impute (mean/mode)** or **drop** missing values.
* **Automatic Encoding** → Categorical variables are handled seamlessly.
* **Model Training**:

  * **Classification**: Logistic Regression, Random Forest, Gradient Boosting
  * **Regression**: Linear Regression, Random Forest, Gradient Boosting
* **Model Evaluation**:

  * Classification → Accuracy
  * Regression → R² score
* **Best Model Selection** → Automatically identifies the top-performing model.

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/eda-helper.git
cd eda-helper
pip install -r requirements.txt
```

**requirements.txt** should include:

```txt
streamlit
pandas
numpy
scikit-learn
plotly
```

---

## 🚀 Usage

Run the app locally:

```bash
streamlit run app.py
```

### Steps in the App

1. **Upload CSV** → Sidebar
2. **Select Target Variable** → Sidebar
3. **Choose Problem Type** → Classification / Regression
4. **Run Analysis & Prediction**
5. Explore:

   * EDA results
   * Feature distributions
   * Correlation heatmap
   * Model performances

---

## 📊 Example Workflow

* Upload dataset (e.g., `titanic.csv`)
* Select `Survived` as target variable
* Choose **Classification**
* Run analysis → App shows:

  * Target variable distribution
  * Correlation heatmap
  * Model performances: Logistic Regression, Random Forest, Gradient Boosting
* **Best Model Example** → Random Forest with Accuracy = **0.85**

---

## 📸 Screenshots (Optional)

*(Add screenshots here once you run the app locally — e.g., EDA preview, correlation heatmap, results table.)*

---

## 🛠️ Tech Stack

* **Frontend** → Streamlit
* **Data Processing** → Pandas, NumPy
* **Visualization** → Plotly (Express + Graph Objects)
* **Machine Learning** → Scikit-learn

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a new branch (`feature-new`)
3. Commit changes
4. Submit a pull request

---

## 📜 License

MIT License – feel free to use and modify.

---
