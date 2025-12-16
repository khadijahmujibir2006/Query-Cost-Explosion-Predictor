import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---- SQL PARSER IMPORTS ----
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Database Query Cost Explosion Predictor",
    layout="centered"
)

# ================= LOAD MODEL =================
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# ================= SQL PARSER =================

def parse_sql_query(query: str):
    parsed = sqlparse.parse(query)
    if not parsed:
        return 1, 0, 0, 0

    stmt = parsed[0]

    tables = set()
    joins = 0
    aggregation = 0
    where_clause = 0

    for token in stmt.tokens:
        # Detect JOINs
        if token.ttype is Keyword and "JOIN" in token.value.upper():
            joins += 1

        # Detect WHERE
        if token.ttype is Keyword and token.value.upper() == "WHERE":
            where_clause = 1

        # Detect aggregations
        if token.ttype is Keyword and token.value.upper() in [
            "COUNT", "SUM", "AVG", "MIN", "MAX"
        ]:
            aggregation = 1

        # Detect tables
        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                tables.add(identifier.get_real_name())

        if isinstance(token, Identifier):
            tables.add(token.get_real_name())

    return max(len(tables), 1), joins, aggregation, where_clause

# ================= HELPER FUNCTIONS =================

def risk_level(cost):
    if cost < 300:
        return "Low Risk 🟢"
    elif cost < 700:
        return "Medium Risk 🟡"
    return "High Risk 🔴"

def optimization_score(cost, joins, index):
    score = 100
    score -= joins * 10
    if index == 0:
        score -= 20
    if cost > 700:
        score -= 30
    return max(0, min(100, score))

def optimization_advice(joins, index, aggregation):
    tips = []
    if joins > 2:
        tips.append("Reduce JOINs or consider denormalization.")
    if index == 0:
        tips.append("Add indexes on JOIN and WHERE columns.")
    if aggregation == 1:
        tips.append("Optimize GROUP BY using indexed columns.")
    if not tips:
        tips.append("Query structure looks well optimized.")
    return tips

def find_explosion_threshold(model, base_features):
    for r in range(1000, 200000, 5000):
        test = base_features.copy()
        test[2] = r
        if model.predict([test])[0] > 700:
            return r
    return None

def growth_type(costs):
    if costs[-1] > costs[0] * 5:
        return "Exponential Growth 🔥"
    return "Linear Growth ✅"

def plot_feature_importance(model):
    features = ["Tables", "Joins", "Rows", "Index", "Aggregation"]
    importance = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_title("Feature Importance (Cost Drivers)")
    st.pyplot(fig)

def log_prediction(record):
    file = "data/history.csv"
    df = pd.DataFrame([record])
    try:
        old = pd.read_csv(file)
        df = pd.concat([old, df])
    except:
        pass
    df.to_csv(file, index=False)

# ================= UI =================

st.title("🗄️ Database Query Cost Explosion Predictor")

st.markdown(
    """
This system predicts **SQL query execution cost explosion** using **Machine Learning**
and a **real SQL parser**.  
It simulates scalability risks and provides optimization intelligence.
"""
)

# ---------------- SQL INPUT ----------------

st.markdown("## 🧾 SQL Query Analyzer (Real Parser)")

sql_query = st.text_area(
    "Paste your SQL query here",
    height=150,
    placeholder="SELECT COUNT(*) FROM orders o JOIN customers c ON o.id = c.id WHERE o.amount > 500"
)

use_sql = st.checkbox("Use SQL query for feature extraction", value=True)

if sql_query.strip() and use_sql:
    tables, joins, aggregation, where_clause = parse_sql_query(sql_query)
    st.success(
        f"Detected → Tables: {tables}, Joins: {joins}, "
        f"Aggregation: {aggregation}, WHERE: {where_clause}"
    )
else:
    st.markdown("### 🔧 Manual Query Parameters")
    tables = st.number_input("Number of tables", 1, 10, 1)
    joins = st.number_input("Number of joins", 0, 10, 0)
    aggregation = st.selectbox(
        "Aggregation used?",
        [0, 1],
        format_func=lambda x: "Yes" if x else "No"
    )

rows = st.number_input("Approximate number of rows", 100, 1_000_000, 1000, step=100)
index = st.selectbox(
    "Index present?",
    [0, 1],
    format_func=lambda x: "Yes" if x else "No"
)

# ================= PREDICTION =================

if st.button("🚀 Analyze Query"):
    features = [tables, joins, rows, index, aggregation]
    prediction = int(model.predict([features])[0])

    st.markdown("## 🔍 Analysis Result")
    st.metric("Estimated Execution Time (ms)", prediction)
    st.write("### Risk Level:", risk_level(prediction))
    st.metric("Optimization Score", f"{optimization_score(prediction, joins, index)}/100")

    # Optimization Advice
    st.markdown("## 🛠 Optimization Suggestions")
    for tip in optimization_advice(joins, index, aggregation):
        st.write("•", tip)

    # Feature Importance
    st.markdown("## 📊 Why this query is expensive")
    plot_feature_importance(model)

    # Cost Growth Simulation
    st.markdown("## 📈 Cost Growth Simulation")
    sim_rows = [rows, rows * 2, rows * 5, rows * 10]
    sim_costs = [
        model.predict([[tables, joins, r, index, aggregation]])[0]
        for r in sim_rows
    ]
    st.line_chart(
        pd.DataFrame(
            {"Predicted Cost (ms)": sim_costs},
            index=sim_rows
        )
    )
    st.write("Growth Pattern:", growth_type(sim_costs))

    # Explosion Threshold
    threshold = find_explosion_threshold(model, features.copy())
    if threshold:
        st.warning(f"⚠ Query cost likely explodes after ~{threshold} rows")
    else:
        st.success("No immediate cost explosion detected")

    # Logging
    log_prediction({
        "time": datetime.now(),
        "tables": tables,
        "joins": joins,
        "rows": rows,
        "index": index,
        "aggregation": aggregation,
        "predicted_cost": prediction
    })

    st.success("📁 Prediction logged successfully")

# ================= HISTORY =================

st.markdown("---")
st.markdown("## 🧾 Prediction History")

try:
    history = pd.read_csv("data/history.csv")
    st.dataframe(history.tail(10))
except:
    st.info("No history available yet.")
