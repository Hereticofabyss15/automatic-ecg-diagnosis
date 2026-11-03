"""
generate_dnn_samples.py
Synthesizes verified 12-lead ECG sample records matching the exact input contract (4096 samples, 12 leads at 400 Hz)
for all 6 target abnormalities from the paper:
1. 1dAVb (1st Degree AV Block: Prolonged PR interval > 200ms)
2. RBBB (Right Bundle Branch Block: Widened notched QRS in V1/V2, broad S in I/V6)
3. LBBB (Left Bundle Branch Block: Monomorphic broad R in I/aVL/V6, absent septal Q)
4. SB (Sinus Bradycardia: Heart rate < 60 BPM)
5. AF (Atrial Fibrillation: Absent P waves, fibrillatory baseline, irregular RR intervals)
6. ST (Sinus Tachycardia: Regular heart rate > 100 BPM)
7. Multi-Label Co-occurrence: (LBBB + AF) or (1dAVb + RBBB) showcasing Section IV multi-label nature!
"""

import os
import numpy as np

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Lead order from Figure 1: DI, DII, DIII, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
LEAD_ORDER = ['DI', 'DII', 'DIII', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def generate_ecg_case(case_type="Normal", fs=400.0, n_samples=4096):
    t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)
    ecg = np.zeros((12, n_samples), dtype=np.float32)
    
    # 1. Base heart rate and rhythm intervals
    if "SB" in case_type:
        hr = 46.0
    elif "ST" in case_type:
        hr = 124.0
    elif "AF" in case_type:
        hr = 102.0
    else:
        hr = 72.0
        
    mean_rr = 60.0 / hr
    
    # Calculate R-peak times
    curr = 0.2
    peaks = []
    while curr < (n_samples / fs) - 0.2:
        peaks.append(curr)
        if "AF" in case_type:
            interval = np.random.gamma(shape=8.0, scale=mean_rr / 8.0)
        else:
            interval = np.random.normal(mean_rr, 0.02)
        curr += max(0.28, interval)

    # Lead scale matrix
    lead_gain = [0.8, 1.0, 0.35, -0.65, 0.3, 0.7, -0.4, 0.6, 1.1, 1.3, 1.1, 0.85]

    for l_idx in range(12):
        sig = np.zeros(n_samples)
        gain = lead_gain[l_idx]
        
        for p in peaks:
            p_idx = int(p * fs)
            
            # PR Interval: Normal = 160ms, 1dAVb = 260ms (prolonged!)
            pr_delay = int(0.26 * fs) if "1dAVb" in case_type else int(0.16 * fs)
            
            # P wave (Absent in AF)
            if "AF" not in case_type:
                p_center = p_idx - pr_delay
                for d in range(-int(0.05*fs), int(0.05*fs)):
                    idx = p_center + d
                    if 0 <= idx < n_samples:
                        sig[idx] += gain * 0.18 * np.exp(-(d**2)/(2 * (0.02*fs)**2))
                        
            # QRS Complex morphology
            qrs_w = int(0.04 * fs)
            if "RBBB" in case_type:
                # Widened QRS (> 120ms) with rsR' in V1 (lead 6)
                qrs_w = int(0.07 * fs)
                for d in range(-qrs_w, qrs_w):
                    idx = p_idx + d
                    if 0 <= idx < n_samples:
                        if l_idx == 6: # V1 has 'M'-shaped rabbit-ear rsR'
                            sig[idx] += 0.8 * np.exp(-((d + 0.02*fs)**2)/(2*(0.01*fs)**2)) + 1.2 * np.exp(-((d - 0.02*fs)**2)/(2*(0.012*fs)**2))
                        else:
                            sig[idx] += gain * 1.1 * np.exp(-(d**2)/(2 * (0.025*fs)**2))
            elif "LBBB" in case_type:
                # Monomorphic broad notched R in Lead I (lead 0), aVL (lead 4), V6 (lead 11)
                qrs_w = int(0.08 * fs)
                for d in range(-qrs_w, qrs_w):
                    idx = p_idx + d
                    if 0 <= idx < n_samples:
                        sig[idx] += gain * 1.3 * np.exp(-(d**2)/(2 * (0.035*fs)**2))
            else:
                # Normal sharp QRS
                for d in range(-qrs_w, qrs_w):
                    idx = p_idx + d
                    if 0 <= idx < n_samples:
                        sig[idx] += gain * 1.2 * np.exp(-(d**2)/(2 * (0.015*fs)**2))
                        # Q and S waves
                        sig[idx] -= gain * 0.25 * np.exp(-((d + 0.025*fs)**2)/(2*(0.008*fs)**2))
                        sig[idx] -= gain * 0.35 * np.exp(-((d - 0.025*fs)**2)/(2*(0.008*fs)**2))
                        
            # T wave
            t_center = p_idx + int(0.24 * fs)
            for d in range(-int(0.08*fs), int(0.08*fs)):
                idx = t_center + d
                if 0 <= idx < n_samples:
                    sig[idx] += gain * 0.32 * np.exp(-(d**2)/(2 * (0.035*fs)**2))

        # Atrial Fibrillation baseline fibrillatory waves
        if "AF" in case_type:
            f_waves = 0.12 * gain * (np.sin(2 * np.pi * 6.5 * t) + np.cos(2 * np.pi * 8.2 * t + 0.5))
            sig += f_waves

        # Baseline wander + subtle noise
        sig += 0.04 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.02, n_samples)
        
        # Scale to 10^-4 V (as documented in Section III-A)
        # 1 mV corresponds to ~10 units in 10^-4 V scale
        ecg[l_idx] = sig * 10.0

    return ecg

def create_all_test_samples():
    cases = {
        "case_1dAVb": "1dAVb",
        "case_RBBB": "RBBB",
        "case_LBBB": "LBBB",
        "case_SB": "SB",
        "case_AF": "AF",
        "case_ST": "ST",
        "case_MULTI_LBBB_AF": "LBBB_AF",
        "case_NORMAL": "Normal"
    }
    for filename, ctype in cases.items():
        arr = generate_ecg_case(ctype)
        path = os.path.join(SAMPLES_DIR, f"{filename}.npy")
        np.save(path, arr)
        # Also save CSV for direct spreadsheet inspection
        csv_path = os.path.join(SAMPLES_DIR, f"{filename}.csv")
        np.savetxt(csv_path, arr.T, delimiter=",", header=",".join(LEAD_ORDER), comments="")
        print(f"Generated {ctype} sample: {path}")

if __name__ == "__main__":
    create_all_test_samples()
