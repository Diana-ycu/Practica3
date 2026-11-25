import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar y limpiar datos
df = pd.read_csv("sensores_planta.csv")
df = df.dropna()

# 2. Separar características (X) y variable objetivo (y)
X = df[["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]]
y = df["estado"]

# 3. División 80% entrenamiento, 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Entrenar RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 5. Evaluar precisión
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"🎯 Precisión del modelo (Accuracy): {precision * 100:.2f}%")

# 6. Exportar modelo si cumple el umbral mínimo
if precision >= 0.80:
    joblib.dump(modelo, "modelo_mantenimiento.pkl")
    print("✅ Modelo exportado exitosamente como 'modelo_mantenimiento.pkl'")
else:
    print("⚠️ La precisión es menor al 80%. Revisa los datos antes de exportar.")