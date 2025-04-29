"""
Lab 2 – Part B  (Source Separation with FastICA)
------------------------------------------------
• 依赖:
  pip install numpy scikit-learn soundfile matplotlib librosa scipy tqdm
• 前置:
  - 保证 'preprocessed_X.npy' 已由 Part A 生成
  - 确保 TARGET_SR 与 Part A 中使用的采样率一致
"""

import os, warnings
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
from sklearn.decomposition import PCA, FastICA
from scipy import signal
from tqdm import tqdm

# ---------- Config ---------- #
PREPROCESSED_NPY = "preprocessed_X.npy"
TARGET_SR        = 16_000
OUT_DIR          = "SeparatedSources"
PLOT_EXAMPLE     = True        # True -> 画第1个源的波形&谱
N_COMPONENTS     = None        # None -> 自动等于通道数
# ---------------------------- #

os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

# 1. Load pre-processed matrix X (T × N)
X = np.load(PREPROCESSED_NPY)          # shape (T, n_mics)
T, N = X.shape
print(f"[INFO] Loaded {PREPROCESSED_NPY} shape={X.shape}")

# 2. Optional PCA whitening  (keeps numerical stability & lets you limit dimension)
n_components = N_COMPONENTS or N
pca = PCA(n_components=n_components, whiten=True, random_state=0)
X_white = pca.fit_transform(X)          # shape (T, n_components)
print(f"[INFO] After PCA whitening shape={X_white.shape}")

# 3. FastICA separation
ica = FastICA(
    n_components=n_components,
    whiten=False,               # 已经PCA-whiten，无需再白化
    max_iter=1000,
    tol=1e-4,
    fun="exp",                  # 对超高斯语音效果好，可改 'cube'
    random_state=0
)
S_est = ica.fit_transform(X_white)      # Independent sources  shape (T, n_components)
print("[INFO] FastICA converged:", ica.n_iter_ , "iterations")

# 4. Save separated signals as .wav
for idx in range(S_est.shape[1]):
    # 归一化防 clipping
    y = S_est[:, idx]
    y = y / np.max(np.abs(y)+1e-9)
    out_path = os.path.join(OUT_DIR, f"Source{idx+1}.wav")
    sf.write(out_path, y, TARGET_SR)
    print(f"[✓] Saved {out_path}")

# 5. (Optional) Visualise the first separated source
if PLOT_EXAMPLE:
    y = S_est[:, 0] / (np.max(np.abs(S_est[:, 0]))+1e-9)
    # 时域波形
    plt.figure(figsize=(10, 4))
    plt.plot(y, linewidth=0.5)
    plt.title("Separated Source 1 – Time Domain")
    plt.tight_layout()
    plt.show()

    # 频谱 (dB)
    f, Pxx = signal.welch(y, fs=TARGET_SR, nperseg=2048)
    plt.figure(figsize=(8, 4))
    plt.semilogy(f, Pxx)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title("Separated Source 1 – Power Spectral Density")
    plt.tight_layout()
    plt.show()

print("\n[ALL DONE] Part B separation finished.")
