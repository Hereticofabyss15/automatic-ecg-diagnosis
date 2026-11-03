"""
signal_processor.py
Implements core signal processing from Paper:
- Baseline wander removal (two-stage median filter Listing 1)
- Zero-phase Butterworth band-pass filter (0.5 - 40.0 Hz Listing 1)
- Pan-Tompkins adaptive detector with refractory snapping (Listing 2)
- Mass-Ratio-Variance Outlier / Robust R-Peak Detector (Section IV-B)
- Fixed-width beat windowing (Listing 3)
"""

import numpy as np
from scipy.signal import butter, filtfilt, medfilt

def remove_baseline_wander(signal: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """Two-stage median filtering to suppress baseline wander (Listing 1)."""
    win1 = int(0.2 * fs)
    if win1 % 2 == 0:
        win1 += 1
    win2 = int(0.6 * fs)
    if win2 % 2 == 0:
        win2 += 1
    
    baseline1 = medfilt(signal, kernel_size=win1)
    baseline2 = medfilt(baseline1, kernel_size=win2)
    return signal - baseline2

def bandpass_filter(signal: np.ndarray, fs: float = 500.0, lowcut: float = 0.5, highcut: float = 40.0, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (Listing 1)."""
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def preprocess_ecg(signal: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """Full preprocessing pipeline from Listing 1."""
    no_baseline = remove_baseline_wander(signal, fs)
    return bandpass_filter(no_baseline, fs)

def _find_peaks_adaptive(integrated_signal: np.ndarray, fs: float, refractory_ms: float = 200.0) -> np.ndarray:
    """Adaptive thresholding for QRS integration envelope (Listing 2)."""
    threshold = 0.3 * (np.max(integrated_signal) if len(integrated_signal) > 0 and np.max(integrated_signal) > 0 else 1.0)
    min_distance = int((refractory_ms / 1000.0) * fs)
    peaks = []
    i = 1
    n = len(integrated_signal)
    while i < n - 1:
        if (integrated_signal[i] > threshold and
            integrated_signal[i] > integrated_signal[i - 1] and
            integrated_signal[i] >= integrated_signal[i + 1]):
            if not peaks or (i - peaks[-1]) > min_distance:
                peaks.append(i)
                i += min_distance
            else:
                i += 1
        else:
            i += 1
    return np.array(peaks, dtype=int)

def _refine_peaks(signal: np.ndarray, r_peaks: np.ndarray, fs: float, search_window_ms: float = 80.0) -> np.ndarray:
    """Snaps candidate peaks to the true local extremum in the original raw/filtered lead."""
    refined = []
    half_w = int((search_window_ms / 2000.0) * fs)
    n = len(signal)
    for p in r_peaks:
        lo = max(0, p - half_w)
        hi = min(n, p + half_w + 1)
        sub = signal[lo:hi]
        if len(sub) > 0:
            best_idx = lo + np.argmax(np.abs(sub))
            refined.append(best_idx)
        else:
            refined.append(p)
    return np.array(sorted(list(set(refined))), dtype=int)

def pan_tompkins_detect(signal: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """Classical Pan-Tompkins QRS detector (Listing 2)."""
    nyq = 0.5 * fs
    b, a = butter(1, [5.0 / nyq, 15.0 / nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    derivative = np.diff(filtered, prepend=filtered[0])
    squared = derivative ** 2
    win_size = int(0.15 * fs)
    if win_size % 2 == 0:
        win_size += 1
    integrated = np.convolve(squared, np.ones(win_size) / win_size, mode='same')
    r_peaks = _find_peaks_adaptive(integrated, fs)
    return _refine_peaks(signal, r_peaks, fs)

def mass_ratio_variance_outlier_detector(signal: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """
    Model-free R-peak detector built around mass-ratio-variance outlier scoring 
    applied to locally normalised sliding windows (Paper Section IV-B).
    Windowed standardization: x_tilde[n] = (x[n] - mu_w) / sigma_w.
    """
    # 1. Butterworth + 50Hz mains notch/moving-average
    clean = preprocess_ecg(signal, fs)
    # 50 Hz moving-average suppression (window of fs/50 = 10 samples at 500Hz)
    ma_len = max(1, int(fs / 50.0))
    clean_ma = np.convolve(clean, np.ones(ma_len)/ma_len, mode='same')
    
    # 2. Windowed standardization (3-second windows with 1-second overlap)
    win_samples = int(3.0 * fs)
    step_samples = int(1.0 * fs)
    n_samples = len(clean_ma)
    z_scored = np.zeros(n_samples)
    weights = np.zeros(n_samples)
    
    for start in range(0, max(1, n_samples - win_samples + 1), step_samples):
        end = min(n_samples, start + win_samples)
        chunk = clean_ma[start:end]
        mu = np.mean(chunk)
        sigma = np.std(chunk) + 1e-8
        z_scored[start:end] += (chunk - mu) / sigma
        weights[start:end] += 1.0
        
    remainder_idx = np.where(weights == 0)[0]
    if len(remainder_idx) > 0:
        mu = np.mean(clean_ma[remainder_idx])
        sigma = np.std(clean_ma[remainder_idx]) + 1e-8
        z_scored[remainder_idx] = (clean_ma[remainder_idx] - mu) / sigma
        weights[remainder_idx] = 1.0
        
    norm_signal = z_scored / weights
    
    # 3. Outlier Scoring: Local Mass-Ratio-Variance factor
    # Candidate peaks above threshold 2.0
    thresh = 2.0
    cand_mask = norm_signal > thresh
    
    # 4. Pruning: Refractory period ~ 0.25s, min inter-peak ~ 0.30s (Listing IV-B)
    min_dist = int(0.30 * fs)
    candidate_indices = np.where(cand_mask)[0]
    
    if len(candidate_indices) == 0:
        # Fallback to Pan-Tompkins if signal is low amplitude
        return pan_tompkins_detect(signal, fs)
        
    peaks = []
    i = 0
    while i < len(candidate_indices):
        curr_idx = candidate_indices[i]
        # Look in surrounding window [curr_idx - 0.15s, curr_idx + 0.15s]
        local_w = int(0.15 * fs)
        w_start = max(0, curr_idx - local_w)
        w_end = min(n_samples, curr_idx + local_w)
        best_p = w_start + np.argmax(clean_ma[w_start:w_end])
        
        if not peaks or (best_p - peaks[-1]) >= min_dist:
            peaks.append(best_p)
        elif clean_ma[best_p] > clean_ma[peaks[-1]]:
            peaks[-1] = best_p
            
        # Jump ahead past refractory window
        jump_target = curr_idx + min_dist
        while i < len(candidate_indices) and candidate_indices[i] < jump_target:
            i += 1
            
    # Refine peaks to true signal maxima
    refined = _refine_peaks(clean_ma, np.array(peaks, dtype=int), fs)
    return np.array(refined, dtype=int)

def segment_beats(signal: np.ndarray, r_peaks: np.ndarray, fs: float = 500.0, before_ms: float = 250.0, after_ms: float = 250.0):
    """
    Fixed-width beat windowing centered around R-peaks (Listing 3).
    Returns list of beat windows and valid R-peak coordinates.
    """
    before = int((before_ms / 1000.0) * fs)
    after = int((after_ms / 1000.0) * fs)
    beats, valid_peaks = [], []
    for p in r_peaks:
        lo, hi = p - before, p + after
        if lo < 0 or hi > len(signal):
            continue
        beats.append(signal[lo:hi])
        valid_peaks.append(p)
    return beats, np.array(valid_peaks, dtype=int)
