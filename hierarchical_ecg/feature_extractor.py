"""
feature_extractor.py
Implements the 99-feature hierarchical fusion vector described in Sections IV-D, IV-E, IV-F:
1. Stream 1: RR-Interval and HRV Features (19 features, Section IV-D)
2. Stream 2: Beat-Derived Features (8 features, Section IV-E)
3. Stream 3: Multi-Lead Descriptors across 12 leads (72 features = 12 leads x 6 stats, Section IV-F)
Total = 19 + 8 + 72 = 99 features.

Also implements the 54 beat-level features for the XGBoost beat classifier (Section IV-C).
"""

import numpy as np
from scipy import stats

AAMI_CLASSES = ['N', 'SVEB', 'VEB', 'F', 'Q']
RHYTHM_CLASSES = ['AF', 'AFIB', 'SA', 'SB', 'SR', 'ST', 'SVT']

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def extract_rr_hrv_features(r_peaks: np.ndarray, fs: float = 500.0) -> dict:
    """
    Computes 19 RR and HRV descriptors (Section IV-D, Equations 2-6):
    - mean_rr, std_rr (SDNN), min_rr, max_rr, median_rr, iqr_rr, cv_rr
    - mean_hr, std_hr, min_hr, max_hr
    - rmssd, pnn50, pnn100, sdsd, mean_diff_rr
    - entropy_rr, skew_rr, kurtosis_rr
    """
    feats = {}
    if len(r_peaks) < 2:
        # Default fallback for extreme flatlines or artifacts
        default_names = [
            'rr_mean', 'rr_sdnn', 'rr_min', 'rr_max', 'rr_median', 'rr_iqr', 'rr_cv',
            'hr_mean', 'hr_std', 'hr_min', 'hr_max',
            'rr_rmssd', 'rr_pnn50', 'rr_pnn100', 'rr_sdsd', 'rr_mean_diff',
            'rr_entropy', 'rr_skew', 'rr_kurtosis'
        ]
        return {k: 0.0 for k in default_names}
        
    rrs = np.diff(r_peaks) / fs  # in seconds
    hrs = 60.0 / (rrs + 1e-8)    # in bpm
    
    feats['rr_mean'] = float(np.mean(rrs))
    sdnn = float(np.std(rrs, ddof=1)) if len(rrs) > 1 else 0.0
    feats['rr_sdnn'] = sdnn
    feats['rr_min'] = float(np.min(rrs))
    feats['rr_max'] = float(np.max(rrs))
    feats['rr_median'] = float(np.median(rrs))
    q75, q25 = np.percentile(rrs, [75, 25])
    feats['rr_iqr'] = float(q75 - q25)
    feats['rr_cv'] = float(sdnn / (feats['rr_mean'] + 1e-8))
    
    feats['hr_mean'] = float(np.mean(hrs))
    feats['hr_std'] = float(np.std(hrs, ddof=1)) if len(hrs) > 1 else 0.0
    feats['hr_min'] = float(np.min(hrs))
    feats['hr_max'] = float(np.max(hrs))
    
    diff_rrs = np.diff(rrs) * 1000.0  # in ms
    if len(diff_rrs) > 0:
        feats['rr_rmssd'] = float(np.sqrt(np.mean(diff_rrs ** 2)))
        feats['rr_pnn50'] = float(100.0 * np.sum(np.abs(diff_rrs) > 50.0) / len(diff_rrs))
        feats['rr_pnn100'] = float(100.0 * np.sum(np.abs(diff_rrs) > 100.0) / len(diff_rrs))
        feats['rr_sdsd'] = float(np.std(diff_rrs, ddof=1)) if len(diff_rrs) > 1 else 0.0
        feats['rr_mean_diff'] = float(np.mean(np.abs(diff_rrs)))
    else:
        feats['rr_rmssd'] = 0.0
        feats['rr_pnn50'] = 0.0
        feats['rr_pnn100'] = 0.0
        feats['rr_sdsd'] = 0.0
        feats['rr_mean_diff'] = 0.0
        
    # Interval distribution entropy, skewness, kurtosis
    hist, _ = np.histogram(rrs, bins=min(10, max(2, len(rrs))), density=True)
    hist = hist[hist > 0]
    feats['rr_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-12))) if len(hist) > 0 else 0.0
    feats['rr_skew'] = float(stats.skew(rrs)) if len(rrs) > 2 else 0.0
    feats['rr_kurtosis'] = float(stats.kurtosis(rrs)) if len(rrs) > 3 else 0.0
    
    return feats

def extract_single_beat_54_features(beat_window: np.ndarray, pre_rr: float, post_rr: float, fs: float = 500.0) -> np.ndarray:
    """
    Computes 54 signal-processing descriptors for a single beat (Section IV-C):
    - Time-domain morphology (mean, std, skew, kurt, energy, min, max, peak-to-peak)
    - Local RR and Heart Rate context (pre_rr, post_rr, ratio, local HR)
    - QRS measurements (width, slopes, area, zero-crossings)
    - Wavelet / Sub-band energy approximations
    - Spectral content (FFT dominant freq, spectral centroid, spectral flux)
    """
    w = np.asarray(beat_window, dtype=float)
    L = len(w)
    feats = []
    
    # 1. Basic Stats (8)
    feats.append(np.mean(w))
    feats.append(np.std(w))
    feats.append(stats.skew(w) if L > 2 else 0.0)
    feats.append(stats.kurtosis(w) if L > 3 else 0.0)
    feats.append(np.sum(w ** 2))
    feats.append(np.min(w))
    feats.append(np.max(w))
    feats.append(np.max(w) - np.min(w))
    
    # 2. Local RR Interval Context (4)
    feats.append(pre_rr)
    feats.append(post_rr)
    feats.append(pre_rr / (post_rr + 1e-6))
    feats.append(60.0 / (pre_rr + 1e-6))
    
    # 3. Waveform Slopes and QRS shape (8)
    mid = L // 2
    q_region = w[max(0, mid - int(0.06*fs)): mid]
    s_region = w[mid: min(L, mid + int(0.06*fs))]
    
    up_slope = (w[mid] - np.min(q_region)) / (len(q_region) + 1e-6) if len(q_region) > 0 else 0.0
    down_slope = (w[mid] - np.min(s_region)) / (len(s_region) + 1e-6) if len(s_region) > 0 else 0.0
    feats.append(up_slope)
    feats.append(down_slope)
    feats.append(up_slope / (abs(down_slope) + 1e-6))
    # trapz in numpy 2.0+ is trapezoid
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    feats.append(float(trapz_fn(np.abs(w))) if trapz_fn else float(np.sum(np.abs(w))))
    feats.append(np.sum(np.diff(np.sign(w)) != 0)) # zero crossings
    feats.append(np.percentile(w, 25))
    feats.append(np.percentile(w, 75))
    feats.append(np.median(w))
    
    # 4. Segment energies (8 sub-windows)
    sub_size = max(1, L // 8)
    for i in range(8):
        sub = w[i*sub_size : (i+1)*sub_size]
        feats.append(np.sum(sub ** 2) if len(sub) > 0 else 0.0)
        
    # 5. Spectral & Frequency Descriptors (12)
    fft_vals = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(L, 1.0/fs)
    fft_vals[0] = 0.0 # exclude DC
    
    dom_freq = freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0.0
    spec_centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-8)
    spec_energy = np.sum(fft_vals ** 2)
    spec_std = np.sqrt(np.sum(((freqs - spec_centroid)**2) * fft_vals) / (np.sum(fft_vals) + 1e-8))
    
    feats.extend([dom_freq, spec_centroid, spec_energy, spec_std])
    
    # Energy in 8 frequency bands: 0-2Hz, 2-5Hz, 5-10Hz, 10-15Hz, 15-20Hz, 20-30Hz, 30-40Hz, 40+Hz
    band_edges = [0, 2, 5, 10, 15, 20, 30, 40, fs/2.0]
    for b in range(len(band_edges) - 1):
        idx = np.where((freqs >= band_edges[b]) & (freqs < band_edges[b+1]))[0]
        feats.append(np.sum(fft_vals[idx] ** 2) if len(idx) > 0 else 0.0)
        
    # 6. Derivative dynamics (14 features to bring total exactly to 54)
    dw = np.diff(w)
    feats.append(np.mean(dw))
    feats.append(np.std(dw))
    feats.append(np.max(dw))
    feats.append(np.min(dw))
    feats.append(np.sum(dw ** 2))
    d2w = np.diff(dw)
    feats.append(np.mean(d2w))
    feats.append(np.std(d2w))
    feats.append(np.max(d2w))
    feats.append(np.min(d2w))
    feats.append(np.sum(d2w ** 2))
    feats.append(float(np.argmax(w)) / L) # R-peak relative location
    feats.append(float(np.argmin(w)) / L)
    feats.append(float(np.sum(w > 0.5 * np.max(w)))) # peak width at half max
    feats.append(float(np.sum(np.abs(dw)))) # curve total variation
    
    return np.array(feats[:54], dtype=float)

def extract_beat_derived_features(beat_predictions: list) -> dict:
    """
    Computes 8 beat-derived features (Section IV-E, Equations 7-8):
    - Class burdens: beta_N, beta_SVEB, beta_VEB, beta_F, beta_Q
    - Ectopic burden: Be = (n_SVEB + n_VEB) / N
    - Class transition rate: T
    - Class entropy: H
    """
    N = len(beat_predictions)
    if N == 0:
        return {
            'beat_burden_N': 1.0,
            'beat_burden_SVEB': 0.0,
            'beat_burden_VEB': 0.0,
            'beat_burden_F': 0.0,
            'beat_burden_Q': 0.0,
            'ectopic_burden': 0.0,
            'transition_rate': 0.0,
            'class_entropy': 0.0
        }
        
    counts = {c: 0 for c in AAMI_CLASSES}
    for b in beat_predictions:
        counts[b] = counts.get(b, 0) + 1
        
    beta = {c: counts[c] / float(N) for c in AAMI_CLASSES}
    ectopic_burden = (counts['SVEB'] + counts['VEB']) / float(N)
    
    transitions = 0
    for i in range(N - 1):
        if beat_predictions[i] != beat_predictions[i+1]:
            transitions += 1
    transition_rate = (transitions / float(N - 1)) if N > 1 else 0.0
    
    entropy = 0.0
    for c in AAMI_CLASSES:
        p = beta[c]
        if p > 0:
            entropy -= p * np.log2(p)
            
    return {
        'beat_burden_N': float(beta['N']),
        'beat_burden_SVEB': float(beta['SVEB']),
        'beat_burden_VEB': float(beta['VEB']),
        'beat_burden_F': float(beta['F']),
        'beat_burden_Q': float(beta['Q']),
        'ectopic_burden': float(ectopic_burden),
        'transition_rate': float(transition_rate),
        'class_entropy': float(entropy)
    }

def extract_multilead_descriptors(ecg_12lead: np.ndarray, fs: float = 500.0) -> dict:
    """
    Computes 6 statistics across all 12 leads (Section IV-F, Equations 9-10):
    1. Mean: mu
    2. Standard deviation: sigma
    3. Root-mean-square: rms = sqrt( (1/L) * sum(x[n]^2) )
    4. Peak-to-peak amplitude: PP = max(x[n]) - min(x[n])
    5. Energy: E = sum(x[n]^2)
    6. Dominant frequency: f_d = (k* / L) * fs
    12 leads x 6 stats = 72 inputs.
    """
    feats = {}
    n_leads, L = ecg_12lead.shape
    
    for l_idx in range(12):
        lead_name = LEAD_NAMES[l_idx] if l_idx < len(LEAD_NAMES) else f"lead_{l_idx+1}"
        sig = ecg_12lead[l_idx] if l_idx < n_leads else np.zeros(L)
        
        feats[f'{lead_name}_mean'] = float(np.mean(sig))
        feats[f'{lead_name}_std'] = float(np.std(sig))
        feats[f'{lead_name}_rms'] = float(np.sqrt(np.mean(sig ** 2)))
        feats[f'{lead_name}_peak_to_peak'] = float(np.max(sig) - np.min(sig))
        feats[f'{lead_name}_energy'] = float(np.sum(sig ** 2))
        
        # Dominant frequency (excluding zero-frequency bin)
        fft_vals = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(L, 1.0 / fs)
        if len(fft_vals) > 1:
            fft_vals[0] = 0.0 # zero DC
            dom_f = freqs[np.argmax(fft_vals)]
        else:
            dom_f = 0.0
        feats[f'{lead_name}_dom_freq'] = float(dom_f)
        
    return feats

def build_99_feature_vector(rr_hrv_dict: dict, beat_derived_dict: dict, multilead_dict: dict) -> tuple[np.ndarray, list]:
    """Concatenates all 3 streams into the exact 99-dimensional fusion vector."""
    combined = {}
    combined.update(rr_hrv_dict)
    combined.update(beat_derived_dict)
    combined.update(multilead_dict)
    
    feature_names = list(combined.keys())
    values = np.array([combined[k] for k in feature_names], dtype=float)
    return values, feature_names
