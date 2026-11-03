"""
generate_samples.py
Creates realistic physiological 12-lead ECG sample records (.mat / .npy / .csv)
spanning all seven rhythm classes analyzed in the research paper:
- Sinus Rhythm (SR)
- Sinus Bradycardia (SB)
- Sinus Tachycardia (ST)
- Atrial Fibrillation (AFIB)
- Atrial Flutter (AF)
- Sinus Irregularity (SA)
- Supraventricular Tachycardia (SVT)
"""

import os
import numpy as np
import scipy.io as sio

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

def synthesize_12lead_ecg(rhythm_type: str = "SR", duration_sec: float = 10.0, fs: float = 500.0) -> np.ndarray:
    """Synthesizes realistic 12-lead waveforms mimicking electrophysiological properties."""
    total_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    
    # 1. Heart rate parameters & RR variability
    if rhythm_type == "SB":
        target_hr = 48.0
        rr_std = 0.03
    elif rhythm_type == "SR":
        target_hr = 72.0
        rr_std = 0.02
    elif rhythm_type == "ST":
        target_hr = 125.0
        rr_std = 0.015
    elif rhythm_type == "AFIB":
        target_hr = 105.0
        rr_std = 0.22  # irregularly irregular
    elif rhythm_type == "AF":
        target_hr = 100.0
        rr_std = 0.04
    elif rhythm_type == "SA":
        target_hr = 70.0
        rr_std = 0.15  # respiratory sinus arrhythmia
    elif rhythm_type == "SVT":
        target_hr = 168.0
        rr_std = 0.01
    else:
        target_hr = 75.0
        rr_std = 0.02

    # Generate R-peak timings
    mean_rr = 60.0 / target_hr
    current_time = 0.15
    peak_times = []
    
    while current_time < duration_sec - 0.2:
        peak_times.append(current_time)
        if rhythm_type == "SA":
            # Respiratory variation modulation (~0.25 Hz modulation)
            mod = 0.18 * np.sin(2 * np.pi * 0.25 * current_time)
            interval = mean_rr + mod + np.random.normal(0, 0.01)
        elif rhythm_type == "AFIB":
            # Chaotic gamma/exponential distributed intervals
            interval = np.random.gamma(shape=9.0, scale=mean_rr / 9.0)
        else:
            interval = np.random.normal(mean_rr, rr_std)
        current_time += max(0.28, interval)

    # 2. Template P-Q-R-S-T complexes
    lead_ecg = np.zeros((12, total_samples))
    
    # Multi-lead relative lead axis scaling
    lead_scales = {
        'I': 0.8, 'II': 1.0, 'III': 0.4, 'aVR': -0.7, 'aVL': 0.3, 'aVF': 0.7,
        'V1': -0.5, 'V2': 0.6, 'V3': 1.1, 'V4': 1.3, 'V5': 1.1, 'V6': 0.9
    }
    lead_names = list(lead_scales.keys())

    for l_idx, l_name in enumerate(lead_names):
        scale = lead_scales[l_name]
        sig = np.zeros(total_samples)
        
        # Add P, QRS, T for each beat
        for pt in peak_times:
            p_sample = int(pt * fs)
            
            # QRS Complex
            qrs_width = int(0.04 * fs)
            for di in range(-qrs_width, qrs_width + 1):
                idx = p_sample + di
                if 0 <= idx < total_samples:
                    # R wave
                    sig[idx] += scale * 1.2 * np.exp(- (di ** 2) / (2 * (0.015 * fs) ** 2))
                    # Q dip
                    sig[idx] -= scale * 0.25 * np.exp(- ((di + 0.03*fs) ** 2) / (2 * (0.01*fs) ** 2))
                    # S dip
                    sig[idx] -= scale * 0.35 * np.exp(- ((di - 0.03*fs) ** 2) / (2 * (0.01*fs) ** 2))
            
            # P wave (Absent in AFIB, Sawtooth in AF)
            if rhythm_type != "AFIB" and rhythm_type != "AF":
                p_offset = int(-0.16 * fs)
                for di in range(-int(0.05*fs), int(0.05*fs)):
                    idx = p_sample + p_offset + di
                    if 0 <= idx < total_samples:
                        sig[idx] += scale * 0.18 * np.exp(- (di ** 2) / (2 * (0.02 * fs) ** 2))
            
            # T wave
            t_offset = int(0.24 * fs)
            for di in range(-int(0.08*fs), int(0.08*fs)):
                idx = p_sample + t_offset + di
                if 0 <= idx < total_samples:
                    sig[idx] += scale * 0.35 * np.exp(- (di ** 2) / (2 * (0.035 * fs) ** 2))

        # 3. Add Atrial Flutter Sawtooth / AFIB Chaotic baseline
        if rhythm_type == "AF":
            # 5 Hz continuous flutter waves (F-waves)
            sawtooth = 0.22 * scale * np.sin(2 * np.pi * 5.0 * t) + 0.08 * np.sin(2 * np.pi * 10.0 * t)
            sig += sawtooth
        elif rhythm_type == "AFIB":
            # Chaotic baseline fibrillatory wave (6-9 Hz noise)
            f_chaos = 0.12 * scale * (np.sin(2 * np.pi * 6.8 * t) + np.cos(2 * np.pi * 8.2 * t + 0.5))
            sig += f_chaos

        # Subtle physiological baseline wander + thermal noise
        sig += 0.05 * np.sin(2 * np.pi * 0.15 * t)
        sig += np.random.normal(0, 0.025, total_samples)
        
        lead_ecg[l_idx] = sig

    return lead_ecg

def generate_all_sample_files():
    """Generates ready-to-test .mat files for all 7 rhythm classes."""
    classes = ["SR", "SB", "ST", "AFIB", "AF", "SA", "SVT"]
    for c in classes:
        ecg = synthesize_12lead_ecg(c)
        mat_path = os.path.join(SAMPLES_DIR, f"sample_{c}.mat")
        sio.savemat(mat_path, {'val': ecg, 'fs': 500, 'rhythm': c})
        # Also save as csv for universal preview
        csv_path = os.path.join(SAMPLES_DIR, f"sample_{c}.csv")
        np.savetxt(csv_path, ecg.T, delimiter=",", header=",".join(lead_names_list()), comments="")
        print(f"Generated sample for {c}: {mat_path}")

def lead_names_list():
    return ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

if __name__ == "__main__":
    generate_all_sample_files()
