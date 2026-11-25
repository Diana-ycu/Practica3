import streamlit as st
import pandas as pd
import joblib
from groq import Groq
from datetime import datetime

# Configuración de página
st.set_page_config(
    page_title="Mantenimiento Predictivo IA",
    page_icon="⚙️",
    layout="wide"
)

# 1. Cargar modelo entrenado en caché
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_mantenimiento.pkl")

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"Error crítico al cargar 'modelo_mantenimiento.pkl': {e}")
    st.stop()

# Título principal
st.title("⚙️ Sistema Predictivo de Mantenimiento Industrial")
st.markdown("Monitoreo de telemetría en tiempo real y generación automática de Órdenes de Trabajo con IA.")

# 2. Controles en Sidebar
st.sidebar.header("🎛️ Telemetría de Sensores")
temperatura = st.sidebar.slider("Temperatura (°C)", min_value=40.0, max_value=120.0, value=60.0, step=0.5)
vibracion = st.sidebar.slider("Vibración (mm/s)", min_value=0.5, max_value=12.0, value=1.5, step=0.1)
horas = st.sidebar.number_input("Horas de Operación", min_value=0, max_value=10000, value=300, step=50)
presion = st.sidebar.slider("Presión (PSI)", min_value=70.0, max_value=170.0, value=92.0, step=0.5)

# 3. Tarjetas visuales st.metric
st.subheader("📊 Lecturas Actuales de la Línea")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", f"{temperatura} °C")
col2.metric("Vibración", f"{vibracion} mm/s")
col3.metric("Operación", f"{horas} hrs")
col4.metric("Presión", f"{presion} PSI")

st.divider()

# 4. Predicción de Machine Learning
entrada = pd.DataFrame([{
    "temperatura_c": temperatura,
    "vibracion_mms": vibracion,
    "horas_operacion": horas,
    "presion_psi": presion
}])

prediccion = int(modelo.predict(entrada)[0])
probabilidades = modelo.predict_proba(entrada)[0]
confianza = probabilidades[prediccion] * 100

st.subheader("🧠 Diagnóstico del Modelo Predictivo")

if prediccion == 0:
    st.success(f"**Estado 0: Operación Normal** — Confianza del modelo: **{confianza:.1f}%**")
    estado_texto = "Operación Normal"
elif prediccion == 1:
    st.warning(f"**Estado 1: Mantenimiento Preventivo Requerido** — Confianza del modelo: **{confianza:.1f}%**")
    estado_texto = "Mantenimiento Preventivo"
else:
    st.error(f"**Estado 2: FALLO CRÍTICO INMINENTE** — Confianza del modelo: **{confianza:.1f}%**")
    estado_texto = "Fallo Crítico"

# 5. Generación de Orden de Trabajo con Groq API
st.subheader("📋 Dictamen Técnico / Orden de Trabajo (IA)")

def generar_respuesta_groq(estado_id, estado_nombre, temp, vib, hrs, pres, conf):
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)

    # Detección dinámica de modelo activo
    modelos_activos = [
        m.id for m in client.models.list().data
        if not any(x in m.id.lower() for x in ["whisper", "guard", "tts", "playai", "embed"])
    ]
    modelo_groq = next((m for m in modelos_activos if "llama" in m.lower()), modelos_activos[0])

    system_prompt = (
        "Eres un Jefe de Mantenimiento Industrial experto en maquinaria automatizada, "
        "análisis de vibraciones, termografía y seguridad industrial. Responde siempre en español "
        "con lenguaje técnico, preciso y directo."
    )

    folio = f"OT-{datetime.now().strftime('%Y%m%d-%H%M')}"

    if estado_id == 0:
        user_prompt = (
            f"La máquina presenta los siguientes parámetros en rango estable ({estado_nombre}, confianza {conf:.1f}%):\n"
            f"- Temperatura: {temp} °C\n"
            f"- Vibración: {vib} mm/s\n"
            f"- Horas de operación: {hrs} hrs\n"
            f"- Presión: {pres} PSI\n\n"
            "Emite un dictamen breve (máximo 1 párrafo) confirmando la estabilidad operativa e indicando "
            "las recomendaciones rutinarias de monitoreo preventivo."
        )
    else:
        urgencia = "PROGRAMADA (Mantenimiento Preventivo)" if estado_id == 1 else "INMEDIATA / PARO DE EMERGENCIA (Fallo Crítico)"
        user_prompt = (
            f"Se ha detectado una anomalía de nivel: {estado_nombre} (Confianza: {conf:.1f}%).\n"
            f"Lecturas registradas:\n"
            f"- Temperatura: {temp} °C\n"
            f"- Vibración: {vib} mm/s\n"
            f"- Horas de operación: {hrs} hrs\n"
            f"- Presión: {pres} PSI\n\n"
            f"Genera una **Orden de Trabajo Técnico** formal en formato Markdown que incluya exactamente estas 4 secciones:\n"
            f"1. **Folio y Prioridad:** Usa el folio `{folio}` e indica prioridad {urgencia}.\n"
            f"2. **Diagnóstico Técnico:** Causa probable basada en los valores anómalos de temperatura, vibración, horas y presión.\n"
            f"3. **3 Acciones Técnicas:** Lista numerada de 3 intervenciones concretas a realizar.\n"
            f"4. **Equipo de Protección Personal (EPP):** Lista del EPP obligatorio para intervenir el equipo de forma segura."
        )

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=modelo_groq,
        temperature=0.3,
    )
    return chat_completion.choices[0].message.content

with st.spinner("Consultando al Jefe de Mantenimiento IA..."):
    try:
        reporte_ia = generar_respuesta_groq(
            prediccion, estado_texto, temperatura, vibracion, horas, presion, confianza
        )
        st.markdown(reporte_ia)
    except KeyError:
        st.warning("⚠️ No se encontró `GROQ_API_KEY` en `st.secrets`. El diagnóstico de Machine Learning sigue operativo, pero la generación de órdenes con IA está deshabilitada.")
    except Exception as e:
        st.warning(f"⚠️ No fue posible conectar con la API de Groq en este momento ({e}). El diagnóstico predictivo de Machine Learning sigue funcionando correctamente.")