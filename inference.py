from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import torchaudio

# Carga del modelo de Hugging Face
model_name = "superb/wav2vec2-base-superb-er"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
labels = model.config.id2label

def predict_emotion(audio_path):
    # Cargar el audio (mono, 16-bit, 16kHz)
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    inputs = extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=1).item()
    return labels[predicted_id]
