import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import numpy as np
import wave
import io

class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

def record_audio():
    webrtc_ctx = webrtc_streamer(
        key="send-audio",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        ),
        audio_processor_factory=AudioProcessor,
    )

    if "audio_buffer" not in st.session_state:
        st.session_state.audio_buffer = []

    if webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        for frame in audio_frames:
            audio = frame.to_ndarray()
            st.session_state.audio_buffer.append(audio)

    if st.button("Guardar audio"):
        if st.session_state.audio_buffer:
            audio_np = np.concatenate(st.session_state.audio_buffer, axis=0).astype(np.int16)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(audio_np.tobytes())
            buffer.seek(0)
            return buffer.read()
        else:
            st.warning("No se ha grabado audio.")
    return None
