import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import os
import tempfile
import io
from scipy.io import wavfile

# Try importing additional libraries
try:
    import parselmouth
    PARSALMOUT_AVAILABLE = True
except ImportError:
    PARSALMOUT_AVAILABLE = False

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

# Load the trained model and scaler
try:
    model = joblib.load('pd_prediction_model.pkl')
    scaler = joblib.load('scaler (1).pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'pd_prediction_model.pkl' and 'scaler (1).pkl' are in the project folder.")
    st.stop()

# Function to extract features from audio bytes
def extract_features_from_bytes(audio_bytes):
    # Save bytes to temp WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        audio_path = tmp_file.name

    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError("Audio file not found")

        # Load audio file (44.1 kHz)
        y, sr = librosa.load(audio_path, sr=44100)

        # Extract MFCCs (84 coefficients to match Col_57–140)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=84)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Extract pitch-based features (approximates Baseline Features, e.g., Col_3–23)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.sum(pitches > 0) > 0 else 0
        pitch_std = np.std(pitches[pitches > 0]) if np.sum(pitches > 0) > 0 else 0

        # Extract intensity (approximates Col_24–26)
        intensity = librosa.feature.rms(y=y)[0]
        intensity_mean = np.mean(intensity)
        intensity_std = np.std(intensity)

        # Initialize features list
        features = [pitch_mean, pitch_std, intensity_mean, intensity_std]

        # Extract jitter and shimmer using parselmouth if available (approximates Vocal Fold features, Col_35–56)
        if PARSALMOUT_AVAILABLE:
            try:
                snd = parselmouth.Sound(audio_path)
                pitch = snd.to_pitch()
                pulses = parselmouth.praat.call(pitch, "To PointProcess")
                jitter_local = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) or 0.0
                shimmer_local = parselmouth.praat.call([snd, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 30) or 0.0
                features.extend([jitter_local, shimmer_local])
            except:
                features.extend([0.0, 0.0])  # Fallback
        else:
            features.extend([0.0, 0.0])  # Fallback

        # Extract formant frequencies if parselmouth available (approximates Col_27–30)
        if PARSALMOUT_AVAILABLE:
            try:
                snd = parselmouth.Sound(audio_path)
                formants = parselmouth.praat.call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
                formant1_mean = parselmouth.praat.call(formants, "Get mean", 1, 0, 0, "Hertz") or 0.0
                formant2_mean = parselmouth.praat.call(formants, "Get mean", 2, 0, 0, "Hertz") or 0.0
                formant3_mean = parselmouth.praat.call(formants, "Get mean", 3, 0, 0, "Hertz") or 0.0
                formant4_mean = parselmouth.praat.call(formants, "Get mean", 4, 0, 0, "Hertz") or 0.0
                features.extend([formant1_mean, formant2_mean, formant3_mean, formant4_mean])
            except:
                features.extend([0.0, 0.0, 0.0, 0.0])  # Fallback
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])  # Fallback

        # Extract Wavelet features if available (approximates Col_141–322, up to 182 features)
        wavelet_features = []
        if PYWT_AVAILABLE:
            try:
                coeffs = pywt.wavedec(y, 'db4', level=4)  # Daubechies wavelet, level 4
                for c in coeffs:
                    if len(c) > 0:
                        wavelet_features.extend([np.mean(c), np.std(c), np.min(c), np.max(c)])
                wavelet_features = wavelet_features[:182]
            except:
                pass
        while len(wavelet_features) < 182:
            wavelet_features.append(0.0)
        features.extend(wavelet_features)

        # Combine with MFCCs (84 features)
        features = np.concatenate([np.array(features), mfccs_mean])

        # Pad or truncate to match 753 features
        if len(features) < 753:
            features = np.pad(features, (0, 753 - len(features)), mode='constant')
        elif len(features) > 753:
            features = features[:753]

        return features

    finally:
        # Clean up temp file
        if os.path.exists(audio_path):
            os.unlink(audio_path)

# Streamlit app
st.title("Parkinson's Disease Speech Prediction")
st.write("Record or upload a WAV audio file (sustained /a/ vowel, 44.1 kHz, 3–5 seconds) to predict Parkinson's Disease.")

# Tabs for recording and uploading
tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

with tab1:
    st.write("Click below to record a sustained /a/ sound for 3–5 seconds.")
    audio_file = st.audio_input("Record your voice", sample_rate=44100)

    if audio_file is not None:
        # Read bytes from UploadedFile
        audio_bytes = audio_file.read()

        # Process audio bytes
        try:
            features = extract_features_from_bytes(audio_bytes)
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)[0]

            # Display results
            st.subheader("Prediction Results")
            st.write(f"**Class**: {'Parkinson\'s Disease' if prediction[0] == 1 else 'Healthy'}")
            # st.write(f"**Confidence (Healthy, PD)**: {probability[0]:.2f}, {probability[1]:.2f}")

            # Save results to CSV
            results = pd.DataFrame({
                'Audio_File': ['recorded_audio.wav'],
                'Prediction': ['PD' if prediction[0] == 1 else 'Healthy'],
                'Healthy_Probability': [probability[0]],
                'PD_Probability': [probability[1]]
            })
            results.to_csv('prediction_results.csv', index=False)
            st.success("Results saved to 'prediction_results.csv'")

            # Download results
            with open('prediction_results.csv', 'rb') as f:
                st.download_button(
                    label="Download Prediction Results",
                    data=f,
                    file_name='prediction_results.csv',
                    mime='text/csv'
                )

        except Exception as e:
            st.error(f"Error processing recorded audio: {e}")

with tab2:
    uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])
    if uploaded_file is not None:
        # Read bytes from UploadedFile
        audio_bytes = uploaded_file.read()

        # Process audio bytes
        try:
            features = extract_features_from_bytes(audio_bytes)
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)[0]

            # Display results
            st.subheader("Prediction Results")
            st.write(f"**Class**: {'Parkinson\'s Disease' if prediction[0] == 1 else 'Healthy'}")
            # st.write(f"**Confidence (Healthy, PD)**: {probability[0]:.2f}, {probability[1]:.2f}")

            # Save results to CSV
            results = pd.DataFrame({
                'Audio_File': [uploaded_file.name],
                'Prediction': ['PD' if prediction[0] == 1 else 'Healthy'],
                'Healthy_Probability': [probability[0]],
                'PD_Probability': [probability[1]]
            })
            results.to_csv('prediction_results.csv', index=False)
            st.success("Results saved to 'prediction_results.csv'")

            # Download results
            with open('prediction_results.csv', 'rb') as f:
                st.download_button(
                    label="Download Prediction Results",
                    data=f,
                    file_name='prediction_results.csv',
                    mime='text/csv'
                )

        except Exception as e:
            st.error(f"Error processing uploaded audio: {e}")
    else:
        st.info("Please upload a WAV audio file to proceed.")