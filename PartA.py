"""
Lab 2 – Part A (Q1.1)  预处理流水线
--------------------------------------------------
• 依赖：
  pip install soundfile librosa scipy numpy tqdm
• 目录结构假设：
  .
  ├── preprocess.py          # <--- 本脚本
  └── MysteryAudioLab2/
         ├── mic1.wav
         ├── mic2.wav
         └── ...
"""

import os, glob, warnings
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from tqdm import tqdm

# ---------------- 参数配置 ---------------- #
RAW_DIR = "MysteryAudioLab2"
TARGET_SR = 16_000  # 统一采样率 (Hz)
HIGH_PASS_HZ = 20  # 高通截止频率
SAVE_NPY = "preprocessed_X.npy"


# ---------------------------------------- #

def high_pass(sig, sr, cutoff=20, order=5):
    """Butterworth 高通滤波（去掉 DC & 超低频轰鸣）"""
    sos = signal.butter(order, cutoff, btype='highpass', fs=sr, output='sos')
    return signal.sosfiltfilt(sos, sig)


def process_one(path, target_sr):
    """读取单个 wav 并完成全部预处理"""
    data, sr = sf.read(path)  # data shape=(T,) or (T, C)
    if data.ndim == 2:  # 多通道转单声道
        data = np.mean(data, axis=1)
    if sr != target_sr:  # 重新采样
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    data = signal.detrend(data)  # 去直流趋势
    data = high_pass(data, sr, HIGH_PASS_HZ)  # 可选高通
    data = data / np.max(np.abs(data))  # 归一化幅度到 [-1,1]
    return data.astype(np.float32)


def main():
    wav_list = sorted(glob.glob(os.path.join(RAW_DIR,"**", "*.wav")))
    if not wav_list:
        raise FileNotFoundError(f"No wav files found in {RAW_DIR}")

    # 先找最短长度，后续裁剪保证同步
    lengths = []
    print("Scanning file lengths...")
    for fp in wav_list:
        lengths.append(sf.info(fp).frames)
    min_len = min(lengths)

    # 批量处理
    processed = []
    print("Processing & stacking...")
    for fp in tqdm(wav_list):
        sig = process_one(fp, TARGET_SR)
        processed.append(sig[:min_len])  # 裁剪到公共长度

    # 形状 (n_mics, T) 再转置成 (T, n_mics) 便于 sklearn-ICA
    X = np.stack(processed, axis=0).T
    np.save(SAVE_NPY, X)
    print(f"[✓] 预处理完成，保存到 {SAVE_NPY}，形状 {X.shape}")


if __name__ == "__main__":
    # 关闭杂项 warning（如采样率变化提示等）
    warnings.filterwarnings("ignore")
    main()
