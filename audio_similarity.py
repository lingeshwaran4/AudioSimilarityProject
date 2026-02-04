import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load audio files
audio1, sr1 = librosa.load("OSR_us_000_0035_8k.wav", sr=None)
audio2, sr2 = librosa.load("OSR_us_000_0036_8k.wav", sr=None)

# Compute Spectrograms
S1 = np.abs(librosa.stft(audio1))
S2 = np.abs(librosa.stft(audio2))

# Plot Spectrograms
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
librosa.display.specshow(librosa.amplitude_to_db(S1, ref=np.max),
                         sr=sr1, x_axis='time', y_axis='log')
plt.title("Audio 1 Spectrogram")
plt.colorbar()

plt.subplot(1, 2, 2)
librosa.display.specshow(librosa.amplitude_to_db(S2, ref=np.max),
                         sr=sr2, x_axis='time', y_axis='log')
plt.title("Audio 2 Spectrogram")
plt.colorbar()

plt.tight_layout()
plt.show()

# Extract MFCCs
mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr2, n_mfcc=13)

# Average MFCCs over time
mfcc1_mean = np.mean(mfcc1, axis=1).reshape(1, -1)
mfcc2_mean = np.mean(mfcc2, axis=1).reshape(1, -1)

# Similarity score
similarity = cosine_similarity(mfcc1_mean, mfcc2_mean)

print(f"Similarity Score: {similarity[0][0]:.2f}")
