import numpy as np
import librosa

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        rmse_mean = np.mean(rms)
        rmse_std = np.std(rms)

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        centroid_mean, centroid_std = np.mean(centroid), np.std(centroid)
        bandwidth_mean, bandwidth_std = np.mean(bandwidth), np.std(bandwidth)
        rolloff_mean, rolloff_std = np.mean(rolloff), np.std(rolloff)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = [np.mean(mfcc[i]) for i in range(13)]
        mfcc_stds = [np.std(mfcc[i]) for i in range(13)]

        # Gabungkan semua fitur
        features = [
            zcr_mean, zcr_std,
            rmse_mean, rmse_std,
            centroid_mean, centroid_std,
            bandwidth_mean, bandwidth_std,
            rolloff_mean, rolloff_std
        ]

        for mean, std in zip(mfcc_means, mfcc_stds):
            features.extend([mean, std])

        return np.array(features).reshape(1, -1)
    
    except Exception as e:
        print(f"❌ Error ekstraksi fitur: {e}")
        return None
