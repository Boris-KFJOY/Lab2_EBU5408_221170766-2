# ---------- Config ---------- #
PREPROCESSED_NPY = "preprocessed_X.npy"
TARGET_SR        = 16_000
N_COMPONENTS     = None    # None -> 自动等于通道数
SHOW_SEC         = 5
N_FFT            = 1024
HOP              = 256
CMAP             = "magma"

MAX_ITER_LIST    = [10, 100, 1000]   # 这里设置要测试的max_iter
FUN_USED         = "exp"              # function固定
WHITEN           = True               # 一律先PCA白化

from sklearn.decomposition import PCA, FastICA
import os, glob, warnings, numpy as np, soundfile as sf, matplotlib.pyplot as plt, librosa, librosa.display
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ---------- load data ---------- #
X = np.load(PREPROCESSED_NPY)          # shape (T, n_mics)
T, N = X.shape
print(f"[INFO] Loaded {PREPROCESSED_NPY} shape={X.shape}")

# ---------- Whitening ---------- #
print(f"\n[INFO] Applying PCA whitening …")
n_components = N_COMPONENTS or N
pca = PCA(n_components=n_components, whiten=True, random_state=0)
X_white = pca.fit_transform(X)
print(f"[INFO] After whitening shape={X_white.shape}")

# ---------- Loop over different max_iter ---------- #
for max_iter in MAX_ITER_LIST:
    print(f"\n[INFO] === FastICA with max_iter={max_iter} ===")

    ica = FastICA(
        n_components=X_white.shape[1],
        whiten=False,          # 已经白化
        max_iter=max_iter,
        tol=1e-4,
        fun=FUN_USED,
        random_state=0
    )
    S_est = ica.fit_transform(X_white)   # shape (T, n_components)
    print("[INFO] FastICA actually used", ica.n_iter_, "iterations to converge.")

    # ---------- Save separated WAV ---------- #
    OUT_DIR_FUN = f'SeparatedSources_iter{max_iter}'
    os.makedirs(OUT_DIR_FUN, exist_ok=True)

    for idx in range(S_est.shape[1]):
        y = S_est[:, idx]
        y = y / (np.max(np.abs(y)) + 1e-9)
        out_path = os.path.join(OUT_DIR_FUN, f"Source{idx+1}.wav")
        sf.write(out_path, y, TARGET_SR)
        print(f"[✓] Saved {out_path}")

    # ---------- Visualization (Waveform + Spectrogram) ---------- #
    OUT_DIR_FIG = f'Figures_Q3.1_iter{max_iter}'
    os.makedirs(OUT_DIR_FIG, exist_ok=True)

    t = np.arange(T) / TARGET_SR
    idx_end = int(SHOW_SEC * TARGET_SR)
    t = t[:idx_end]
    S_est = S_est[:idx_end]

    def compute_spec(x):
        D = librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window='hann')
        return librosa.amplitude_to_db(np.abs(D) + 1e-9, ref=np.max)

    def plot_wave_and_spec(sig, title, fname):
        fig, ax = plt.subplots(2, 1, figsize=(10, 4), constrained_layout=True,
                               gridspec_kw={'height_ratios': [1, 2]})

        ax[0].plot(t, sig, linewidth=0.6)
        ax[0].set(title=title + ' – Waveform', xlabel='Time (s)', ylabel='Amp')
        ax[0].grid(alpha=0.3)

        S_db = compute_spec(sig)
        img = librosa.display.specshow(S_db, sr=TARGET_SR, hop_length=HOP,
                                       x_axis='time', y_axis='linear',
                                       cmap=CMAP, ax=ax[1])
        ax[1].set(title=title + ' – Spectrogram (dB)')
        fig.colorbar(img, ax=ax[1], format='%+2.0f dB')

        fig.savefig(os.path.join(OUT_DIR_FIG, fname), dpi=150)
        plt.close(fig)

    print('[INFO] Generating figures …')
    for j in tqdm(range(S_est.shape[1]), leave=False):
        plot_wave_and_spec(
            S_est[:, j] / (np.max(np.abs(S_est[:, j])) + 1e-9),
            f'ITER{max_iter} Source {j+1}',
            f'Source{j+1}.png'
        )

    print(f"[✓] All figures saved to \"{OUT_DIR_FIG}/\"")

print("\n[ALL DONE] loop over max_iter completed.")
