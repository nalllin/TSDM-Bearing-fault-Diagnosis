import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import torch
import torch.nn.functional as F

def load_mat_signal(file_path: str) -> np.ndarray:
    """
    Load a vibration signal from a .mat file.
    Tries to find a key containing '_DE_time' (drive end signal).
    """
    mat = loadmat(file_path)
    for key in mat.keys():
        if "_DE_time" in key or "_de_time" in key:
            signal = mat[key].squeeze()
            return signal.astype(np.float32)
    for key in mat.keys():
        if not key.startswith("__"):
            signal = mat[key].squeeze()
            return signal.astype(np.float32)
    raise KeyError(f"No valid signal found in {file_path}")

def segment_signal(signal: np.ndarray, segment_length: int, step: int) -> list:
    """
    Segment a 1D signal into overlapping chunks.
    """
    segments = []
    N = len(signal)
    for start in range(0, N - segment_length + 1, step):
        seg = signal[start:start+segment_length]
        if len(seg) == segment_length:
            segments.append(seg.copy())
    return segments

def normalize_signals(data: np.ndarray) -> np.ndarray:
    """
    Normalize signals to zero mean and unit variance.
    """
    mean = np.mean(data)
    std = np.std(data)
    if std < 1e-8:
        std = 1e-8
    return (data - mean) / std

def plot_time_series(signal: np.ndarray, fs: int = 12000, title: str = "Vibration Signal", save_path: str = None):
    """
    Plot a time-domain signal.
    """
    duration = len(signal) / fs
    t = np.linspace(0, duration, len(signal), endpoint=False)
    plt.figure(figsize=(6, 3))
    plt.plot(t, signal, lw=1.0)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def compute_envelope_spectrum(signal: np.ndarray, fs: int = 12000):
    """
    Compute the envelope spectrum using the Hilbert transform.
    """
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    fft_vals = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), d=1.0/fs)
    return freqs, fft_vals
