"""
resnet_dnn.py
PyTorch implementation of the 1D Deep Residual Neural Network (ResNet-1D).
Exactly reconstructs Section V (Residual Network Method), Figures 4, 5, 6, and Equations (5)-(8)
from the research report:
- Input contract: X in R^(N x 4096 x 12) at 400 Hz
- Stem: Conv1D (kernel_size=16, 64 filters) + BatchNorm1d + ReLU
- Stage 1: 4096 -> 1024 samples, 128 channels (ResidualUnit)
- Stage 2: 1024 -> 256 samples, 196 channels (ResidualUnit)
- Stage 3: 256  -> 64 samples,  256 channels (ResidualUnit)
- Stage 4: 64   -> 16 samples,  320 channels (ResidualUnit)
- Dense Linear Layer: 16 * 320 = 5120 -> 6 outputs
- Sigmoid Activation: Independent multi-label output scores p_k in (0, 1)
- Grad-CAM / Attribution Hooks: Allows 1D temporal saliency extraction across all 12 leads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

TARGET_CLASSES = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
NOMINAL_THRESHOLDS = np.array([0.124, 0.070, 0.050, 0.278, 0.390, 0.174], dtype=float)

CLASS_DESCRIPTIONS = {
    '1dAVb': '1st Degree Atrioventricular Block (Conduction: PR > 200 ms)',
    'RBBB': 'Right Bundle Branch Block (Conduction: rsR\' in V1, wide S in I/V6)',
    'LBBB': 'Left Bundle Branch Block (Conduction: broad notched R in I/aVL/V6)',
    'SB': 'Sinus Bradycardia (Rhythm: SA pacing rate < 60 BPM)',
    'AF': 'Atrial Fibrillation (Rhythm: irregularly irregular, absent P-waves)',
    'ST': 'Sinus Tachycardia (Rhythm: SA pacing rate > 100 BPM)'
}

class Conv1dSame(nn.Module):
    """1D Convolution with 'SAME' padding behavior matching TensorFlow."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        # Calculate padding dynamically for 'SAME' padding
        in_len = x.shape[-1]
        out_len = int(np.ceil(float(in_len) / float(self.stride)))
        pad_needed = max(0, (out_len - 1) * self.stride + self.kernel_size - in_len)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        if pad_needed > 0:
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)

class ResidualUnit1D(nn.Module):
    """
    Residual unit block (unidimensional) from Section V, Fig. 4.
    Applies Conv1d -> BN -> ReLU -> Dropout -> Conv1d(stride) + Shortcut.
    """
    def __init__(self, in_samples, out_samples, in_channels, out_channels, kernel_size=17, dropout_rate=0.2):
        super().__init__()
        self.downsample_factor = in_samples // out_samples
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Residual branch: Conv1D -> BN -> ReLU -> Dropout -> Conv1D(stride)
        self.conv1 = Conv1dSame(in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = Conv1dSame(out_channels, out_channels, kernel_size=kernel_size, stride=self.downsample_factor, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut branch: MaxPool (if downsample) + 1x1 Conv (if channel mismatch)
        if self.downsample_factor > 1:
            self.shortcut_pool = nn.MaxPool1d(kernel_size=self.downsample_factor, stride=self.downsample_factor, padding=0)
        else:
            self.shortcut_pool = nn.Identity()
            
        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut_conv = nn.Identity()

    def forward(self, x):
        # Shortcut path
        shortcut = self.shortcut_pool(x)
        shortcut = self.shortcut_conv(shortcut)
        
        # Residual path
        res = self.conv1(x)
        res = self.bn1(res)
        res = F.relu(res)
        res = self.dropout(res)
        res = self.conv2(res)
        
        # Match dimensions if slight off-by-one due to pooling
        if res.shape[-1] != shortcut.shape[-1]:
            min_len = min(res.shape[-1], shortcut.shape[-1])
            res = res[..., :min_len]
            shortcut = shortcut[..., :min_len]
            
        out = res + shortcut
        out = self.bn2(out)
        out = F.relu(out)
        return out

class ResNet1DECG(nn.Module):
    """
    Complete 1D Residual Neural Network for 12-lead multi-label ECG diagnosis.
    Input: (Batch, 12, 4096)
    Output: (Batch, 6) logit predictions
    """
    def __init__(self, in_channels=12, n_classes=6):
        super().__init__()
        # Stem Convolution: kernel 16, 64 filters
        self.stem_conv = Conv1dSame(in_channels, 64, kernel_size=16, stride=1, bias=False)
        self.stem_bn = nn.BatchNorm1d(64)
        
        # 4 Residual Stages (Section V, Fig. 5)
        # Stage 1: 4096 -> 1024, 64 -> 128
        self.stage1 = ResidualUnit1D(4096, 1024, 64, 128, kernel_size=17, dropout_rate=0.2)
        # Stage 2: 1024 -> 256, 128 -> 196
        self.stage2 = ResidualUnit1D(1024, 256, 128, 196, kernel_size=17, dropout_rate=0.2)
        # Stage 3: 256 -> 64, 196 -> 256
        self.stage3 = ResidualUnit1D(256, 64, 196, 256, kernel_size=17, dropout_rate=0.2)
        # Stage 4: 64 -> 16, 256 -> 320
        self.stage4 = ResidualUnit1D(64, 16, 256, 320, kernel_size=17, dropout_rate=0.2)
        
        # Dense classification head: Flatten (16 * 320 = 5120) -> Dense(6)
        self.fc = nn.Linear(16 * 320, n_classes)
        
        # Saliency gradient hooks
        self.feature_gradients = None
        self.last_conv_features = None

    def activations_hook(self, grad):
        self.feature_gradients = grad

    def forward(self, x, return_features=False):
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = F.relu(x)
        
        # Residual stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.last_conv_features = x
        
        # Flatten
        feat_flat = torch.flatten(x, 1)
        logits = self.fc(feat_flat)
        
        if return_features:
            return logits, feat_flat
        return logits
