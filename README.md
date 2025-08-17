# 📊 ML4BankGrowth — Lending Rate Prediction  

This project applies **Machine Learning** (Ridge Regression with Recursive Feature Elimination) to predict **bank lending rates** based on regional and economic indicators.  

It provides a clean and interactive **Streamlit web app** for making predictions and visualizing the most important features influencing the model.  

---

## 🚀 Features
- **Interactive Web App** built with Streamlit.  
- **Predict Lending Rate** by adjusting regional & economic input features.  
- **Top 5 Important Features** shown in a simple bar chart for interpretability.  
- Model built using **Ridge Regression** with **RFE (Recursive Feature Elimination)** for feature selection.  
- **User-friendly UI** with sidebar inputs and styled prediction output.  

---

## 🏦 App Preview

### Main Interface  
- Sidebar inputs for all available features  
- Green success box showing prediction result  
- Bar chart of **Top 5 feature importances**  

*(You can add screenshots here after running the app, e.g., from Streamlit Community Cloud.)*

---

## ⚙️ Installation & Setup  

Clone the repository:  
```bash
git clone https://github.com/Humble-Librarian/ML4BankGrowth.git
cd ML4BankGrowth

```
### Create a virtual environment & install dependencies
```bash
pip install -r requirements.txt

```
### Run the Streamlit app locally
```bash
streamlit run app.py

```
## 📂 Repository Structure
ML4BankGrowth/
│── app.py # Streamlit web app
│── model.pkl # Trained Ridge Regression model
│── rfe_selector.pkl # Feature selector (RFE)
│── requirements.txt # Python dependencies
│── README.md # Project documentation
