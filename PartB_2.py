#  plot_Q22.py  –  Lab-2  Q2.2  Visualisation helper
# ---------------------------------------------------
# Requirements:
#   pip install numpy soundfile matplotlib librosa seaborn tqdm

import os, glob, warnings, numpy as np, soundfile as sf, matplotlib.pyplot as plt, librosa, librosa.display
from tqdm import tqdm

# ------------- CONFIG -------------
MIX_NPY         = "preprocessed_X.npy"     # mixtures  (T × N_mics)
SRC_DIR         = "SeparatedSources"       # *.wav from Part B
OUT_DIR         = "Figures_Q22"
SAMPLING_RATE   = 16_000                   # must match Part B
SHOW_SEC        = 5                        # plot first N seconds
N_FFT           = 1024                     # spectrogram params
HOP             = 256
CMAP            = "magma"
# ----------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

# ----------  load data ----------
X   = np.load(MIX_NPY)          # mixtures
T   = X.shape[0]
t   = np.arange(T) / SAMPLING_RATE

# load separated sources (read wavs so they are already normalised)
src_paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.wav")))
S_list = [sf.read(fp)[0] for fp in src_paths]          # list of 1-D arrays
S      = np.stack(S_list, axis=1)                      # (T × N_src)
print(f"[INFO] mixtures shape={X.shape}, sources shape={S.shape}")

# trim to SHOW_SEC seconds for cleaner figures
idx_end = int(SHOW_SEC * SAMPLING_RATE)
X = X[:idx_end]
S = S[:idx_end]
t = t[:idx_end]

# Utility – spectrogram (magnitude in dB)
def compute_spec(x):
    D = librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window="hann")
    S_db = librosa.amplitude_to_db(np.abs(D)+1e-9, ref=np.max)
    return S_db

# ----------  plot function ----------
def plot_wave_and_spec(sig, title, fname):
    fig, ax = plt.subplots(2, 1, figsize=(10, 4), constrained_layout=True,
                           gridspec_kw={"height_ratios": [1, 2]})

    # waveform
    ax[0].plot(t, sig, linewidth=0.6)
    ax[0].set(title=title + " – Waveform", xlabel="Time (s)", ylabel="Amp")
    ax[0].grid(alpha=0.3)

    # spectrogram
    S_db = compute_spec(sig)
    img = librosa.display.specshow(S_db, sr=SAMPLING_RATE, hop_length=HOP,
                                   x_axis='time', y_axis='linear',
                                   cmap=CMAP, ax=ax[1])
    ax[1].set(title=title + " – Spectrogram (dB)")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)

# ----------  iterate & save ----------
print("[INFO] Generating figures …")
for k in tqdm(range(X.shape[1])):
    plot_wave_and_spec(X[:, k] / (np.max(np.abs(X[:, k]))+1e-9),
                       f"Mixture Mic {k+1}", f"Mixture{k+1}.png")

for j in tqdm(range(S.shape[1])):
    plot_wave_and_spec(S[:, j] / (np.max(np.abs(S[:, j]))+1e-9),
                       f"Separated Source {j+1}", f"Source{j+1}.png")

print(f"[✓] All figures saved to “{OUT_DIR}/”")
