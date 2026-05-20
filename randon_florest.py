import pandas as pd # type: ignore
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

url = "./creditcard.csv"
df = pd.read_csv(url)
df["Amount_log"] = np.log1p(df["Amount"])

#Preparando o ambiente de treino e teste para o modelo
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount_log"]])

x = df.drop(["Class"], axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.3, random_state=42
)
rf = RandomForestClassifier(
n_estimators=50, 
max_depth=10,
class_weight="balanced",
n_jobs=-1,
random_state=42
)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
print(f"Sem pipeline \n {classification_report(y_test, y_pred)}")

#usando pipeline
pipeline = Pipeline([
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

pipeline.fit(x_train, y_train)

# Probabilidades
y_probs = pipeline.predict_proba(x_test)[:, 1]

# Threshold de 0.3, para ser mais sensível a detecção de fraudes
threshold = 0.3

y_pred_custom = (y_probs > threshold).astype(int)

print(f"Com pipeline \n {classification_report(y_test, y_pred_custom)}")