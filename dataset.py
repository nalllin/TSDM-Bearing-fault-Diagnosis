import os
import numpy as np
from torch.utils.data import Dataset
from .utils import load_mat_signal, segment_signal, normalize_signals

class BearingSignalDataset(Dataset):
    """
    Dataset for bearing vibration signals.
    Loads .mat files, segments signals, and assigns labels.
    """
    def __init__(self, data_dir: str, segment_length: int = 3000, segment_step: int = 750, use_normal: bool = True):
        self.segment_length = segment_length
        self.segment_step = segment_step
        self.segments = []
        self.labels = []
        # Define label mapping: Healthy, IR, OR, B
        self.label_mapping = {"Healthy": 0, "IR": 1, "OR": 2, "B": 3}
        # Select files: use 007 files for faults and files containing "Normal" for healthy
        all_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
        selected_files = []
        for fname in all_files:
            if "007" in fname or ("Normal" in fname and use_normal):
                selected_files.append(os.path.join(data_dir, fname))
        selected_files.sort()
        for filepath in selected_files:
            signal = load_mat_signal(filepath)
            segs = segment_signal(signal, segment_length, segment_step)
            fname = os.path.basename(filepath)
            if "Normal" in fname:
                label = self.label_mapping["Healthy"]
            elif "IR" in fname:
                label = self.label_mapping["IR"]
            elif "OR" in fname:
                label = self.label_mapping["OR"]
            elif "B" in fname:
                label = self.label_mapping["B"]
            else:
                label = -1
            for seg in segs:
                self.segments.append(seg)
                self.labels.append(label)
        self.segments = np.array(self.segments, dtype=np.float32)
        self.segments = normalize_signals(self.segments)
        self.num_samples = self.segments.shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]
        import torch
        segment_tensor = torch.from_numpy(segment).unsqueeze(0)
        return segment_tensor, label
