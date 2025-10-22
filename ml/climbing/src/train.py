import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path

# Chargement
df = pd.read_csv("ml/climbing/data/processed/bodyPerformance_clean.csv")

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.3f}")

# Sauvegarde
out = Path("apps/api/models/climbing/1.0.0")
out.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out / "model.pkl")

print("✅ Modèle sauvegardé dans:", out)
