import pandas as pd # type: ignore
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Importando e agrupando valores pelo LOG()
url = "./creditcard.csv"
df = pd.read_csv(url)
print(df["Class"].value_counts(normalize=True))
df["Amount_log"] = np.log1p(df["Amount"])

#Preparando o ambiente de treino e teste para o modelo
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount_log"]])

x = df.drop(["Class"], axis=1)
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.3, random_state=42
)

#utilizando o modelo
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))
