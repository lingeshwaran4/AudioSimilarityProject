Here’s the full **README.md** content in one single copy‑paste block for you:

```markdown
# 🎵 Audio Similarity Project

This project demonstrates how to **compare two audio files** using **spectrograms** and **MFCC (Mel-Frequency Cepstral Coefficients)** features.  
It leverages **Librosa**, **Matplotlib**, and **Scikit-learn** to visualize and compute similarity between audio signals.

---

## 🚀 Features
- Load and process audio files using **Librosa**
- Generate and visualize **spectrograms**
- Extract **MFCC features**
- Compute **cosine similarity** between audio samples
- Display a **similarity score**

---

## 📂 Project Structure
```
AudioSimilarityProject/
│── OSR_us_000_0035_8k.wav   # Sample audio file 1
│── OSR_us_000_0036_8k.wav   # Sample audio file 2
│── similarity.py            # Main Python script
│── README.md                # Project documentation
```

---

## 🛠️ Requirements
Install the required libraries before running the script:

```bash
pip install librosa matplotlib scikit-learn numpy
```

---

## 📊 How It Works
1. **Load Audio Files**
   ```python
   audio1, sr1 = librosa.load("OSR_us_000_0035_8k.wav", sr=None)
   audio2, sr2 = librosa.load("OSR_us_000_0036_8k.wav", sr=None)
   ```

2. **Compute Spectrograms**
   ```python
   S1 = np.abs(librosa.stft(audio1))
   S2 = np.abs(librosa.stft(audio2))
   ```

3. **Visualize Spectrograms**
   Spectrograms are plotted side by side for comparison.

4. **Extract MFCCs**
   ```python
   mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1, n_mfcc=13)
   mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr2, n_mfcc=13)
   ```

5. **Compute Similarity**
   ```python
   similarity = cosine_similarity(
       np.mean(mfcc1, axis=1).reshape(1, -1),
       np.mean(mfcc2, axis=1).reshape(1, -1)
   )
   print(f"Similarity Score: {similarity[0][0]:.2f}")
   ```

---

## 📈 Output
- **Spectrograms** of both audio files  
- **Similarity Score** (between 0 and 1, where 1 = identical)

Example:
```
Similarity Score: 0.87
```

---

## 🌟 Applications
- Speaker recognition  
- Music recommendation systems  
- Audio fingerprinting  
- Sound classification  

---

## 🤝 Contributing
Feel free to fork this repository, open issues, or submit pull requests to improve the project.

---

## 📜 License
This project is licensed under the MIT License.  
You are free to use, modify, and distribute it with attribution.
```

You can copy this entire block directly into your `README.md` file.
