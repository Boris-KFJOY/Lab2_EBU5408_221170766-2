# ---------------------------------------------
#   Fast ICA on 3 mixtures in  ./myvoice/
#   single run, fun="exp"
# ---------------------------------------------
import os, glob, warnings
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import librosa.display
from scipy import signal
from sklearn.decomposition import PCA, FastICA
from tqdm import tqdm

# ---------- user settings ----------
RAW_DIR       = "myvoice"          # folder with mixture*.wav
TARGET_SR     = 16_000             # resample rate
HIGH_PASS_HZ  = 20                 # remove DC/rumble
SHOW_SEC      = 5                  # preview length
FUN           = "exp"              # ICA non-linearity
MAX_ITER      = 10000
TOL           = 1e-5
WHITEN        = True               # set False to skip PCA-whiten
# -----------------------------------

warnings.filterwarnings("ignore")
os.makedirs("SeparatedSources_myvoice", exist_ok=True)
os.makedirs("Figures_E",           exist_ok=True)

# ----------- helper -----------
def high_pass(x, sr, fc=20, order=5):
    sos = signal.butter(order, fc, btype="highpass", fs=sr, output='sos')
    return signal.sosfiltfilt(sos, x)

def load_and_preprocess(fp):
    y, sr = sf.read(fp)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    y = signal.detrend(y)
    y = high_pass(y, TARGET_SR, HIGH_PASS_HZ)
    y /= np.max(np.abs(y))+1e-9
    return y.astype(np.float32)

# ----------- read mixtures -----------
mix_paths = sorted(glob.glob(os.path.join(RAW_DIR, "mixture*.wav")))
if not mix_paths:
    raise FileNotFoundError("No mixture*.wav found in myvoice/")
print("[INFO] files:", *[os.path.basename(p) for p in mix_paths])

mixes = [load_and_preprocess(p) for p in mix_paths]
min_len = min(map(len, mixes))
X = np.stack([m[:min_len] for m in mixes], axis=1)   # (T,N)
print("[INFO] stacked matrix", X.shape)

# ----------- whitening (optional) -----------
if WHITEN:
    pca = PCA(whiten=True, random_state=0)
    Xw  = pca.fit_transform(X)
    print("[INFO] PCA-whitened shape", Xw.shape)
else:
    Xw = X

# ----------- ICA -----------
ica = FastICA(n_components=Xw.shape[1],
              whiten=False, fun=FUN,
              max_iter=MAX_ITER, tol=TOL,
              random_state=0)
S = ica.fit_transform(Xw)           # (T,N)
print(f"[INFO] converged in {ica.n_iter_} iterations")

# ----------- save wavs -----------
for k in range(S.shape[1]):
    y = S[:,k] / (np.max(np.abs(S[:,k]))+1e-9)
    sf.write(f"SeparatedSources_myvoice/Source{k+1}.wav", y, TARGET_SR)
    print("  ↳ Saved Source", k+1)

# ----------- quick visual preview -----------
t = np.arange(min_len) / TARGET_SR
idx_end = int(SHOW_SEC*TARGET_SR)
def db_spec(x):
    D = librosa.stft(x, n_fft=1024, hop_length=256)
    return librosa.amplitude_to_db(np.abs(D)+1e-9, ref=np.max)
for k in tqdm(range(S.shape[1]), desc="figs"):
    fig, ax = plt.subplots(2,1, figsize=(10,4), constrained_layout=True,
                           gridspec_kw={"height_ratios":[1,2]})
    ax[0].plot(t[:idx_end], S[:idx_end,k])
    ax[0].set(title=f"Source {k+1} – Waveform", xlabel="Time (s)", ylabel="Amp")
    ax[0].grid(alpha=.3)
    img = librosa.display.specshow(db_spec(S[:idx_end,k]),
                                   sr=TARGET_SR, hop_length=256,
                                   x_axis="time", y_axis="linear",
                                   cmap="magma", ax=ax[1])
    ax[1].set(title="Spectrogram (dB)")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    fig.savefig(f"Figures_E/Source{k+1}.png", dpi=150)
    plt.close(fig)
print("[✓] Done")
