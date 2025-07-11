import streamlit as st
from audio_recorder import record_audio
from inference import predict_emotion
from datetime import datetime
import pandas as pd
import os

st.set_page_config(page_title="Emothic Live", layout="centered")

st.title("🎤 Emothic Live – Emotion Detection")
st.write("Graba tu voz y detectaremos tu emoción en tiempo real.")

# Grabar audio
audio_bytes = record_audio()

if audio_bytes:
    filename = "recorded.wav"
    with open(filename, "wb") as f:
        f.write(audio_bytes)

    st.audio(audio_bytes, format="audio/wav")
    st.success("Audio grabado correctamente.")

    # Predicción
    with st.spinner("Analizando emoción..."):
        emotion = predict_emotion(filename)
        st.subheader(f"🧠 Emoción detectada: {emotion}")

        # Guardar en historial
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = pd.DataFrame([[timestamp, emotion]], columns=["timestamp", "emotion"])
        if os.path.exists("history.csv"):
            row.to_csv("history.csv", mode="a", header=False, index=False)
        else:
            row.to_csv("history.csv", index=False)

# Mostrar historial
if os.path.exists("history.csv"):
    st.markdown("---")
    st.subheader("📈 Historial emocional")
    df = pd.read_csv("history.csv")
    st.dataframe(df)

    chart = df["emotion"].value_counts().reset_index()
    chart.columns = ["Emotion", "Count"]
    st.bar_chart(data=chart.set_index("Emotion"))

    st.download_button("⬇️ Descargar historial CSV", data=df.to_csv(index=False),
                       file_name="historial_emothic.csv", mime="text/csv")
