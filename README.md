 🗄️ Database Query Cost Explosion Predictor

An advanced Machine Learning system that predicts **SQL query cost explosion**
and scalability risks by analyzing **query structure, data growth, and optimization factors**.

This project combines **Database Systems + Machine Learning + Explainable AI**
to simulate how real-world query optimizers reason about performance.

---

## 🚀 Key Features

### ✅ Machine Learning Cost Prediction
- Predicts estimated query execution time (ms)
- Trained using Random Forest regression

### ✅ Real SQL Query Parsing
- Uses `sqlparse` to extract:
  - Number of tables
  - JOIN operations
  - Aggregations (GROUP BY, COUNT, SUM)
  - WHERE clauses

### ✅ Query Risk Classification
- Low / Medium / High risk of cost explosion

### ✅ Optimization Score (0–100)
- Single metric representing query efficiency

### ✅ What-If Optimization Analysis
- Simulates:
  - Adding indexes
  - Reducing JOINs
  - Current vs optimized cost comparison

### ✅ Cost Explosion Threshold Detection
- Predicts the data size at which query becomes dangerous

### ✅ Cost Growth Curve Classification
- Detects Linear vs Exponential cost growth

### ✅ Explainable AI
- Feature importance visualization
- Shows why query becomes expensive

### ✅ History Logging
- Stores all predictions for analysis and auditing

---

## 🧠 System Architecture

SQL Query / Parameters
↓
Real SQL Parser (sqlparse)
↓
Feature Extraction
↓
ML Cost Prediction Model
↓
Risk Analysis + Optimization Intelligence
↓
Interactive Streamlit Dashboard

yaml
Copy code

---

## 📂 Project Structure

Query-Cost-Explosion-Predictor
│
├── data
│ ├── query_data.csv
│ └── history.csv
│
├── models
│ └── model.pkl
│
├── src
│ ├── train_model.py
│ └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore

yaml
Copy code

---

## ⚙️ Installation & Setup

```bash
pip install -r requirements.txt
python src/train_model.py
python -m streamlit run src/app.py
Open:

arduino
Copy code
http://localhost:8501
🧪 Example SQL Query
sql
Copy code
SELECT COUNT(o.id)
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN payments p ON p.order_id = o.id
WHERE o.amount > 500
GROUP BY c.country;
🎯 Use Cases
Database performance optimization

Query scalability analysis

ML-assisted query tuning

Educational database systems research

🏆 Why This Project Stands Out
✔ Combines Core CS + ML
✔ Uses real SQL parsing, not keyword matching
✔ Predicts future performance issues, not just current cost
✔ Includes explainability and optimization intelligence

This project demonstrates industry-level thinking and is suitable for
interviews, research portfolios, and shortlisting.

📌 Future Enhancements
Integration with real database EXPLAIN plans

Support for subqueries and nested SELECTs

Multi-model comparison (XGBoost, Linear Regression)

Cloud database cost analysis

👩‍💻 Author
Khadijah Mujibir Rahman
B.E. Computer Science & Engineering

yaml
Copy code

---

## 🔹 STEP 8: Commit README Changes

After saving README.md:

```bash
git add README.md
git commit -m "Add elite project README"
git push
