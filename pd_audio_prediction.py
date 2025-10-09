import librosa
import numpy as np
import pandas as pd
import joblib
import logging
import os

# Try importing additional libraries for enhanced features
try:
    import parselmouth
    PARSALMOUT_AVAILABLE = True
    print("Praat-parselmouth imported successfully for jitter/shimmer extraction.")
except ImportError:
    print("Warning: praat-parselmouth not installed. Jitter/shimmer features will be skipped.")
    PARSALMOUT_AVAILABLE = False

try:
    import pywt
    PYWT_AVAILABLE = True
    print("PyWavelets imported successfully for wavelet extraction.")
except ImportError:
    print("Warning: PyWavelets not installed. Wavelet features will be skipped.")
    PYWT_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model and scaler
try:
    model = joblib.load('pd_prediction_model.pkl')
    scaler = joblib.load('scaler (1).pkl')
    logging.info("Model and scaler loaded successfully")
except FileNotFoundError as e:
    logging.error(f"Error loading model or scaler: {e}")
    raise

# Function to extract features from audio
def extract_features(audio_path):
    try:
        # Verify audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio file (44.1 kHz, as per dataset)
        y, sr = librosa.load(audio_path, sr=44100)
        logging.info(f"Audio loaded: {audio_path}, sample rate: {sr}")

        # Extract MFCCs (84 coefficients to match Col_57–140)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=84)
        mfccs_mean = np.mean(mfccs, axis=1)
        logging.info(f"Extracted {len(mfccs_mean)} MFCC features")

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
                jitter_local = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                shimmer_local = parselmouth.praat.call([snd, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 30)
                features.extend([jitter_local, shimmer_local])
                logging.info("Extracted jitter and shimmer features")
            except Exception as e:
                logging.warning(f"Failed to extract jitter/shimmer: {e}. Using defaults.")
                features.extend([0.0, 0.0])  # Fallback
        else:
            features.extend([0.0, 0.0])  # Fallback
            logging.warning("Jitter/shimmer skipped due to missing parselmouth")

        # Extract formant frequencies if parselmouth available (approximates Col_27–30)
        if PARSALMOUT_AVAILABLE:
            try:
                snd = parselmouth.Sound(audio_path)
                formants = parselmouth.praat.call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
                formant1_mean = parselmouth.praat.call(formants, "Get mean", 1, 0, 0, "Hertz")
                formant2_mean = parselmouth.praat.call(formants, "Get mean", 2, 0, 0, "Hertz")
                formant3_mean = parselmouth.praat.call(formants, "Get mean", 3, 0, 0, "Hertz")
                formant4_mean = parselmouth.praat.call(formants, "Get mean", 4, 0, 0, "Hertz")
                features.extend([formant1_mean, formant2_mean, formant3_mean, formant4_mean])
                logging.info("Extracted formant features")
            except Exception as e:
                logging.warning(f"Failed to extract formants: {e}. Using defaults.")
                features.extend([0.0, 0.0, 0.0, 0.0])  # Fallback
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])  # Fallback
            logging.warning("Formants skipped due to missing parselmouth")

        # Extract Wavelet features if available (approximates Col_141–322, up to 182 features)
        wavelet_features = []
        if PYWT_AVAILABLE:
            try:
                coeffs = pywt.wavedec(y, 'db4', level=4)  # Daubechies wavelet, level 4
                for c in coeffs:
                    if len(c) > 0:  # Avoid empty coeffs
                        wavelet_features.extend([np.mean(c), np.std(c), np.min(c), np.max(c)])
                # Limit to approx 182 features (adjust as needed)
                wavelet_features = wavelet_features[:182]
                logging.info(f"Extracted {len(wavelet_features)} wavelet features")
            except Exception as e:
                logging.warning(f"Failed to extract wavelets: {e}. Using zeros.")
        # Pad wavelet to 182 if fewer
        while len(wavelet_features) < 182:
            wavelet_features.append(0.0)
        features.extend(wavelet_features)

        # Combine with MFCCs (84 features)
        features = np.concatenate([np.array(features), mfccs_mean])

        # Log extracted features count
        logging.info(f"Total extracted features before padding: {len(features)}")

        # Pad or truncate to match 753 features
        # Note: Still missing ~479 features (mostly TQWT, Col_323–754)
        if len(features) < 753:
            features = np.pad(features, (0, 753 - len(features)), mode='constant')
            logging.warning(f"Padded {753 - len(features)} features with zeros")
        elif len(features) > 753:
            features = features[:753]
            logging.warning("Truncated features to 753")

        return features

    except Exception as e:
        logging.error(f"Error in feature extraction: {e}")
        raise

# Path to your audio file (update if different)
audio_path = 'audiomyaudio.wav'

# Extract features
try:
    features = extract_features(audio_path)
    print("Extracted features shape:", features.shape)  # Should be (753,)
except Exception as e:
    logging.error(f"Failed to process audio: {e}")
    raise

# Scale features
try:
    features_scaled = scaler.transform([features])
except Exception as e:
    logging.error(f"Error scaling features: {e}")
    raise

# Predict
try:
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0]
except Exception as e:
    logging.error(f"Error in prediction: {e}")
    raise

# Output results
print("\nPrediction Results:")
print("Class:", "Parkinson's Disease" if prediction[0] == 1 else "Healthy")
print(f"Confidence (Healthy, PD): {probability[0]:.2f}, {probability[1]:.2f}")

# Save results to CSV
results = pd.DataFrame({
    'Audio_File': [audio_path],
    'Prediction': ['PD' if prediction[0] == 1 else 'Healthy'],
    'Healthy_Probability': [probability[0]],
    'PD_Probability': [probability[1]]
})
results.to_csv('prediction_results.csv', index=False)
print("\nResults saved to 'prediction_results.csv'")