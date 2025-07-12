import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import numpy as np
import io
import wave

# Configuración del cliente
client_settings = ClientSettings(
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

# Clase para procesar audio en vivo
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        # Convertir a numpy
        pcm = frame.to_ndarray().flatten()
        self.frames.append(pcm)
        return frame

def record_audio():
    ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDRECV,
        client_settings=client_settings,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if ctx.audio_processor and len(ctx.audio_processor.frames) > 0:
        pcm_data = np.concatenate(ctx.audio_processor.frames).astype(np.int16)

        # Crear archivo WAV en memoria
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(16000)
            wf.writeframes(pcm_data.tobytes())

        return buffer.getvalue()
    
    return None
