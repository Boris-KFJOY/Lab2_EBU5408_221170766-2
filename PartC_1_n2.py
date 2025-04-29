#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part C (n_components=2) — Complete pipeline:
 1) Preprocess microphone recordings → preprocessed_n2.npy
 2) PCA (n_components=2, whiten) + FastICA → separate 2 sources
 3) Save separated sources in SeparatedSources_n2 with suffix _n2
 4) Visualize spectrograms of separated sources
 5) Play audio for listening

Directory structure assumed:
  .
  ├ MysteryAudioLab2/
  │   └ MysteryAudioLab2/
  │      ├ Microphone1.wav
  │      ├ Microphone2.wav
  │      └ Microphone3.wav
  └ PartC_n2.py   (this script)

Dependencies: numpy, scipy, librosa, soundfile, scikit-learn, matplotlib, tqdm, IPython (for Audio)
"""
import os
from pathlib import Path
import glob
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA, FastICA
from tqdm import tqdm

# ----------- Parameters -----------
TARGET_SR = 16000
HIGH_PASS_HZ = 20
N_COMPONENTS = 2
PREPROCESSED_FILE = "preprocessed_n2.npy"
SEPARATED_DIR = Path("SeparatedSources_n2")
SEPARATED_DIR.mkdir(exist_ok=True)

# ----------- Preprocessing -----------
def high_pass(sig, sr, cutoff=HIGH_PASS_HZ, order=5):
    sos = signal.butter(order, cutoff, btype='highpass', fs=sr, output='sos')
    return signal.sosfiltfilt(sos, sig)

def preprocess_and_save():
    # find all mic wavs
    raw_dir = Path("MysteryAudioLab2") / "MysteryAudioLab2"
    wav_paths = sorted(raw_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No .wav in {raw_dir}")
    # read lengths for alignment
    lengths = [sf.info(str(w)).frames for w in wav_paths]
    min_len = min(lengths)
    processed = []
    for p in tqdm(wav_paths, desc="Preprocessing"):  # type: Path
        data, sr = sf.read(str(p))
        # mono
        if data.ndim == 2:
            data = data.mean(axis=1)
        # resample
        if sr != TARGET_SR:
            data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        # detrend, highpass, normalize
        data = signal.detrend(data)
        data = high_pass(data, sr)
        data = data / np.max(np.abs(data)+1e-9)
        processed.append(data[:min_len])
    # stack (T, n_mics)
    X = np.stack(processed, axis=1)
    np.save(PREPROCESSED_FILE, X)
    print(f"[+] Preprocessed saved to {PREPROCESSED_FILE}, shape={X.shape}")
    return X

# ----------- Separation -----------
def separate(X):
    # PCA whiten
    pca = PCA(n_components=N_COMPONENTS, whiten=True, random_state=0)
    Xw = pca.fit_transform(X)
    print(f"PCA whiten done, explained var={pca.explained_variance_ratio_.sum():.3f}")
    # FastICA
    ica = FastICA(n_components=N_COMPONENTS, whiten=False, random_state=0)
    S = ica.fit_transform(Xw)
    print(f"FastICA done, iterations={ica.n_iter_}")
    return S

# ----------- Visualization & Listening -----------
def plot_spectrogram(y, sr, title="Spectrogram"):
    D = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    plt.figure(figsize=(6,4))
    librosa.display.specshow(DB, sr=sr, hop_length=256, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ----------- Main -----------
if __name__ == "__main__":
    # Preprocess
    X = preprocess_and_save()
    # Separate
    S = separate(X)
    # Save and visualize
    for i in range(N_COMPONENTS):
        y = S[:, i]
        y = y / np.max(np.abs(y)+1e-9)
        out_wav = SEPARATED_DIR / f"Source{i+1}_n2.wav"
        sf.write(str(out_wav), y, TARGET_SR)
        print(f"Saved separated source: {out_wav}")
        # plot spectrogram
        plot_spectrogram(y, TARGET_SR, title=f"Source {i+1} (n2) Spectrogram")
        # play audio
        print(f"Playing Source {i+1} (n2):")


    print("All done for n_components=2.")
