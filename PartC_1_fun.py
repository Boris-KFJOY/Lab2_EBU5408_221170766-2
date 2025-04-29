from sklearn.decomposition import PCA, FastICA
import os, glob, warnings, numpy as np, soundfile as sf, matplotlib.pyplot as plt, librosa, librosa.display
from tqdm import tqdm
# ---------- Config ---------- #
PREPROCESSED_NPY = "preprocessed_X.npy"
TARGET_SR        = 16_000
OUT_DIR_fun      = "SeparatedSources_fun"
N_COMPONENTS     = None        # None -> 自动等于通道数
# ---------------------------- #
MIX_NPY         = "preprocessed_X.npy"     # mixtures  (T × N_mics)
SRC_DIR         = "SeparatedSources"       # *.wav from Part B
OUT_DIR         = "Figures_Q3_1_fun"
SAMPLING_RATE   = 16_000                   # must match Part B
SHOW_SEC        = 5                        # plot first N seconds
N_FFT           = 1024                     # spectrogram params
HOP             = 256
CMAP            = "magma"
os.makedirs(OUT_DIR_fun, exist_ok=True)
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

# -----------------------------------------------------------
# 1. ICA + 保存 WAV
# -----------------------------------------------------------
FUN_LIST = ['cube', 'exp', 'logcosh']          # 要循环的 non-linearity
for cur_fun in FUN_LIST:
    print(f'\n[INFO] === FastICA fun=\"{cur_fun}\" ===')

    # 1.1 运行 ICA   -----------------------------------------
    ica = FastICA(
        n_components=n_components,
        whiten=False,          # 数据已 PCA-whiten
        max_iter=1000,
        tol=1e-4,
        fun=cur_fun,
        random_state=0
    )
    S_est = ica.fit_transform(X_white)         # (T × n_components)
    print("[INFO] converged:", ica.n_iter_, "iterations")

    # 1.2 保存 WAV   ----------------------------------------
    OUT_DIR_FUN = f'SeparatedSources_{cur_fun}'
    os.makedirs(OUT_DIR_FUN, exist_ok=True)

    for idx in range(S_est.shape[1]):
        y = S_est[:, idx]
        y = y / (np.max(np.abs(y)) + 1e-9)     # 归一化防 clipping
        out_path = os.path.join(OUT_DIR_FUN, f'Source{idx+1}.wav')
        sf.write(out_path, y, TARGET_SR)
        print(f'[✓] Saved {out_path}')

    # -------------------------------------------------------
    # 2. 读取刚存的结果，画波形 + 频谱图
    # -------------------------------------------------------
    MIX_NPY   = 'preprocessed_X.npy'
    SRC_DIR   = OUT_DIR_FUN                     # 用新的目录
    OUT_DIR   = f'Figures_Q3.1_{cur_fun}'       # 图目录也带后缀

    os.makedirs(OUT_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")

    # ----------  load data ----------
    X   = np.load(MIX_NPY)
    T   = X.shape[0]
    t   = np.arange(T) / TARGET_SR

    src_paths = sorted(glob.glob(os.path.join(SRC_DIR, '*.wav')))
    S_list = [sf.read(fp)[0] for fp in src_paths]
    S      = np.stack(S_list, axis=1)
    print(f'[INFO] mixtures shape={X.shape}, sources shape={S.shape}')

    # ----------  shorten for plotting ----------
    idx_end = int(SHOW_SEC * TARGET_SR)
    X = X[:idx_end]
    S = S[:idx_end]
    t = t[:idx_end]

    # ----------  plotting ----------
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

        fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
        plt.close(fig)

    print('[INFO] Generating figures …')
    for j in tqdm(range(S.shape[1]), leave=False):
        plot_wave_and_spec(
            S[:, j] / (np.max(np.abs(S[:, j])) + 1e-9),
            f'{cur_fun.upper()} Source {j+1}',
            f'Source{j+1}.png'
        )

    print(f'[✓] All figures saved to \"{OUT_DIR}/\"')

print('\n[ALL DONE] loop over fun completed.')

