import torch
import numpy as np
import matplotlib.pyplot as plt

from models.models import ImprovedUNet
from models.dataset import BearingSignalDataset
from models.train import sample
from models.utils import plot_time_series, compute_envelope_spectrum

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the trained model
model = ImprovedUNet(in_channels=1, base_channels=64, channel_mults=(1,2,4,8), time_emb_dim=256)
model_path = "output/trained_model.pth"  # Adjust path if needed
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully.")

# Load dataset and extract one sample per class
data_dir = "C:\\python_workspace\\DDPM\\raw"  # Update to your MAT files directory
dataset = BearingSignalDataset(data_dir=data_dir, segment_length=3000, segment_step=750)

class_examples = {0: None, 1: None, 2: None, 3: None}
for i in range(len(dataset)):
    x, label = dataset[i]
    if class_examples[label] is None:
        class_examples[label] = x.squeeze().numpy()

class_names = {0: "Healthy", 1: "IR", 2: "OR", 3: "B"}
print("Loaded examples:")
for cls, sig in class_examples.items():
    if sig is not None:
        print(f"Class {cls} ({class_names[cls]}): shape {sig.shape}")

# Generate synthetic samples
num_samples_to_generate = 1
synthetic_samples = sample(model, num_samples=num_samples_to_generate, device=device)
synthetic_sample = synthetic_samples[0]
print("Synthetic sample generated, shape:", synthetic_sample.shape)

# Plot real vs. synthetic time-domain signals for each class
for cls in class_examples.keys():
    real_signal = class_examples[cls]
    if real_signal is None:
        continue
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plot_time_series(real_signal, fs=12000, title=f"Real Signal - {class_names[cls]}")
    plt.subplot(1, 2, 2)
    plot_time_series(synthetic_sample, fs=12000, title=f"Synthetic Signal - {class_names[cls]}")
    plt.tight_layout()
    plt.savefig(f"real_vs_synthetic_{class_names[cls]}.png")
    plt.show()

# Compute and plot envelope spectra and differences
for cls in class_examples.keys():
    real_signal = class_examples[cls]
    if real_signal is None:
        continue
    fs = 12000
    freqs, spectrum_real = compute_envelope_spectrum(real_signal, fs=fs)
    _, spectrum_syn = compute_envelope_spectrum(synthetic_sample, fs=fs)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(freqs, spectrum_real, label='Real')
    plt.plot(freqs, spectrum_syn, label='Synthetic')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(f"Envelope Spectrum - {class_names[cls]}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(freqs, np.abs(spectrum_real - spectrum_syn))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Absolute Difference")
    plt.title(f"Spectrum Difference - {class_names[cls]}")
    plt.tight_layout()
    plt.savefig(f"env_spectrum_{class_names[cls]}.png")
    plt.show()
