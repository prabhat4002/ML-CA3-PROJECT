import pandas as pd
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
import os

# Check for praat-parselmouth
try:
    import parselmouth
    PARSALMOUT_AVAILABLE = True
    logging.info("Praat-parselmouth available")
except ImportError:
    PARSALMOUT_AVAILABLE = False
    logging.warning("Praat-parselmouth not installed, skipping jitter/shimmer/formants")

# Check for pywt
try:
    import pywt
    PYWT_AVAILABLE = True
    logging.info("PyWavelets available")
except ImportError:
    PYWT_AVAILABLE = False
    logging.warning("PyWavelets not installed, skipping wavelets")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Train on Selected Features (~115)
# Load dataset
try:
    data = pd.read_csv('pd_speech_features.csv')
    logging.info("Dataset loaded successfully")
except FileNotFoundError as e:
    logging.error(f"Dataset not found: {e}")
    raise

# Drop id and select easy features
data = data.drop('id', axis=1)

# Select features: Baseline (10), Intensity/Formant/Vocal (20), MFCCs (84), Wavelets (20)
baseline_cols = data.columns[0:10]  # Col_2–11 (gender to locAbsJitter)
time_domain_cols = data.columns[10:30]  # Col_12–31 (intensity, formant, vocal)
mfcc_cols = data.columns[55:139]  # Col_57–140 (84 MFCCs, index 55–138 after drop id)
wavelet_cols = data.columns[139:159]  # Col_141–160 (20 wavelets)
selected_cols = list(baseline_cols) + list(time_domain_cols) + list(mfcc_cols)
if PYWT_AVAILABLE:
    selected_cols += list(wavelet_cols)

X = data[selected_cols]
y = data['class']

logging.info(f"Selected {len(selected_cols)} features for training")
logging.info(f"Sample columns: {selected_cols[:5]} ... {selected_cols[-5:]}")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Training Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model, scaler, and selected columns
joblib.dump(model, 'simple_pd_model.pkl')
joblib.dump(scaler, 'simple_scaler.pkl')
joblib.dump(selected_cols, 'selected_features.pkl')
logging.info("Saved model, scaler, and feature list")

# Step 2: Feature Extraction Function
def extract_simple_features(audio_path, selected_cols_len):
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=44100)
        logging.info(f"Audio loaded: {audio_path}, sample rate: {sr}")

        # MFCCs (84)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=84)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Pitch (4, for Baseline)
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        pitch_min = np.min(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        pitch_max = np.max(pitches[pitches > 0]) if np.any(pitches > 0) else 0

        # Intensity (3)
        intensity = librosa.feature.rms(y=y)[0]
        intensity_mean = np.mean(intensity)
        intensity_std = np.std(intensity)
        intensity_max = np.max(intensity)

        # Initialize features
        features = [pitch_mean, pitch_std, pitch_min, pitch_max, intensity_mean, intensity_std, intensity_max]

        # Jitter/Shimmer/Formants (8)
        if PARSALMOUT_AVAILABLE:
            try:
                snd = parselmouth.Sound(audio_path)
                pitch_pm = snd.to_pitch()
                pulses = parselmouth.praat.call(pitch_pm, "To PointProcess")
                # Jitter
                jitter = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) or 0.0
                jitter_abs = parselmouth.praat.call(pulses, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3) or 0.0
                # Shimmer (use local percentage only)
                shimmer = parselmouth.praat.call([snd, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 30) or 0.0
                # Formants
                formants = parselmouth.praat.call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
                f1 = parselmouth.praat.call(formants, "Get mean", 1, 0, 0, "Hertz") or 0.0
                f2 = parselmouth.praat.call(formants, "Get mean", 2, 0, 0, "Hertz") or 0.0
                f3 = parselmouth.praat.call(formants, "Get mean", 3, 0, 0, "Hertz") or 0.0
                f4 = parselmouth.praat.call(formants, "Get mean", 4, 0, 0, "Hertz") or 0.0
                features.extend([jitter, jitter_abs, shimmer, f1, f2, f3, f4])
            except Exception as e:
                logging.warning(f"Failed to extract jitter/shimmer/formants: {e}")
                features.extend([0.0] * 7)  # Reduced to 7 features
        else:
            features.extend([0.0] * 7)
            logging.warning("Skipped jitter/shimmer/formants due to missing praat-parselmouth")

        # Wavelets (20)
        if PYWT_AVAILABLE:
            try:
                coeffs = pywt.wavedec(y, 'db4', level=4)
                wavelet_feats = []
                for c in coeffs:
                    if len(c) > 0:
                        wavelet_feats.extend([np.mean(c), np.std(c)])
                wavelet_feats = wavelet_feats[:20]
                if len(wavelet_feats) < 20:
                    wavelet_feats += [0.0] * (20 - len(wavelet_feats))
                features.extend(wavelet_feats)
            except Exception as e:
                logging.warning(f"Failed to extract wavelets: {e}")
                features.extend([0.0] * 20)
        else:
            features.extend([0.0] * 20)
            logging.warning("Skipped wavelets due to missing PyWavelets")

        # Combine with MFCCs
        features = np.concatenate([features, mfccs_mean])
        logging.info(f"Extracted {len(features)} features from audio")

        # Pad/truncate
        if len(features) < selected_cols_len:
            features = np.pad(features, (0, selected_cols_len - len(features)), 'constant')
            logging.warning(f"Padded {selected_cols_len - len(features)} features with zeros")
        elif len(features) > selected_cols_len:
            features = features[:selected_cols_len]
            logging.warning("Truncated features to match selected columns")

        return features

    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        raise

# Step 3: Predict on Audio
audio_path = 'audiomyaudio.wav'

try:
    selected_cols = joblib.load('selected_features.pkl')
except FileNotFoundError:
    logging.warning("Selected features file not found, using training columns")
    selected_cols = selected_cols  # Use from training

try:
    features = extract_simple_features(audio_path, len(selected_cols))
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    prob = model.predict_proba(features_scaled)[0]

    print("\nPrediction Results:")
    print("Class:", "Parkinson's Disease" if prediction[0] == 1 else "Healthy")
    print(f"Confidence (Healthy, PD): {prob[0]:.2f}, {prob[1]:.2f}")

    # Save results
    results = pd.DataFrame({
        'Audio_File': [audio_path],
        'Prediction': ['PD' if prediction[0] == 1 else 'Healthy'],
        'Healthy_Probability': [prob[0]],
        'PD_Probability': [prob[1]]
    })
    results.to_csv('simple_prediction_results.csv', index=False)
    print("\nResults saved to 'simple_prediction_results.csv'")
except Exception as e:
    logging.error(f"Error in prediction: {e}")
    raise