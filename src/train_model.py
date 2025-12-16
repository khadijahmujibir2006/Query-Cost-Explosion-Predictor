
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/query_data.csv")

X = df.drop("execution_time", axis=1)
y = df["execution_time"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully")
