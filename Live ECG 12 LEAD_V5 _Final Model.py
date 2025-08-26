import serial
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import scipy.signal
import math
import neurokit2 as nk
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from time import sleep
import json
import os
import joblib
import re
import sys
import numpy as np

STX, ETX = 0x02, 0x03
NUM_EXPECTED = 12
float_re = re.compile(rb'-?\d+')  # ekstrak integer bertanda
buf = bytearray()

serialPort = serial.Serial(port='COM25', baudrate=921600)  # Change the COM port as needed

save_path = "D:\EKG\Ekstraksi Fitur 12 Lead\Anak WJ\Test rekam 38 (uji lab2)"


# Monkey‐patch internal module name
sys.modules['numpy._core'] = np.core

# ——— Load scaler & models ———
scaler          = joblib.load('D:\EKG\Skripsi Willy\Model paling bagus\Model 2/scaler.joblib')
model_xgb       = joblib.load('D:\EKG\Skripsi Willy\Model paling bagus\Model 2/model_xgb_binary.joblib')
kmeans_abnormal = joblib.load('D:\EKG\Skripsi Willy\Model paling bagus\Model 2/kmeans_abnormal_model.joblib')

def predict_clusters_xgb_integrated(new_data, xgb_binary_model, kmeans_abnormal,
                                    abnormal_label_offset=1):
    """
    new_data: np.ndarray shape (n_samples, n_features), sudah diskalakan.
    Mengembalikan np.ndarray string label:
      0 -> "Normal"
      1 -> "Potential Fast Arrhytmia"
      2 -> "Potential Heart Block"
    """
    # 1. Prediksi Normal(1) vs Abnormal(0)
    is_normal = xgb_binary_model.predict(new_data)

    # 2. Buat array integer awal
    labels_int = np.full(len(new_data), -999, dtype=int)

    # 3. Sampel normal → label 0
    idx_norm = np.where(is_normal == 1)[0]
    labels_int[idx_norm] = 0

    # 4. Sampel abnormal → kluster + offset
    idx_abn = np.where(is_normal == 0)[0]
    if len(idx_abn):
        abn_data     = new_data[idx_abn]
        cls_nums     = kmeans_abnormal.predict(abn_data) + abnormal_label_offset
        labels_int[idx_abn] = cls_nums

    # 5. Mapping ke nama
    name_map = {
        0: "Normal",
        1: "Potential Fast Arrhytmia",
        2: "Potential Heart Block"
    }
    labels_str = np.array([name_map.get(i, f"Cluster {i}") for i in labels_int])

    # 6. Validasi
    if np.any(labels_int == -999):
        print("⚠️ Beberapa sampel belum terlabel!")

    return labels_str
# Ensure the save path exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

print("Program started...")
timerecord = 10
iteration = 1

# Constants from ADS1293 datasheet
VREF = 2.42  # Reference voltage in volts
GAIN = 200  # Programmable gain
RESOLUTION = 24  # 24-bit resolution
LSB_SIZE = VREF / (2**RESOLUTION - 1)  # Least Significant Bit size in volts

def convert_to_millivolts(adc_value):
    return adc_value * LSB_SIZE * 1000 / GAIN  # Convert to millivolts

import numpy as np

def hampel_despike(x, fs, window_ms=120, n_sigma=4.0):
    """Ganti outlier (paku/impuls) pakai median lokal (Hampel filter)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = max(1, int((window_ms/1000.0)*fs))  # jendela dua sisi
    y = x.copy()
    k = 1.4826  # scale MAD ke std
    for i in range(n):
        i0 = max(0, i-w)
        i1 = min(n, i+w+1)
        m = np.median(x[i0:i1])
        s = k * np.median(np.abs(x[i0:i1] - m)) + 1e-12
        if np.abs(x[i] - m) > n_sigma * s:
            y[i] = m
    return y

def saturasi_interpolasi(x, adc_abs_max=8388607, frac=0.995, guard=5):
    """
    Deteksi nilai mendekati batas ADC (saturasi) lalu interpolasi linear.
    guard: sampel di kiri/kanan yang ikut diganti untuk meredam ringing.
    """
    x = np.asarray(x, dtype=float)
    y = x.copy()
    sat = np.abs(x) >= (frac * adc_abs_max)
    if not np.any(sat):
        return y
    idx = np.where(sat)[0]

    # gabungkan indeks yang berdekatan jadi segmen
    seg = []
    start = idx[0]
    for a,b in zip(idx[:-1], idx[1:]):
        if b != a+1:
            seg.append((start, a))
            start = b
    seg.append((start, idx[-1]))

    for s,e in seg:
        i0 = max(0, s-guard)
        i1 = min(len(y)-1, e+guard)
        # titik tepi untuk interpolasi
        left  = i0-1
        right = i1+1
        if left < 0 or right >= len(y):
            # jika di ujung, isi dengan median lokal
            med = np.median(y[max(0,i0-50):min(len(y),i1+50)])
            y[i0:i1+1] = med
        else:
            y[i0:i1+1] = np.linspace(y[left], y[right], i1-i0+1)
    return y

def despike_pipeline(x, fs):
    """Gabungkan deteksi saturasi lalu Hampel untuk impuls pendek."""
    # 1) tangani saturasi dulu
    y = saturasi_interpolasi(x, adc_abs_max=8388607, frac=0.995, guard=6)
    # 2) hapus impuls/sparks pendek
    y = hampel_despike(y, fs, window_ms=100, n_sigma=4.0)
    return y

def _qrs_ms_from_Ronset_Roffset(beat_index, anchor_sample, waves_dwt, rpeaks, Fs):
    """
    Estimasi lebar QRS dari R_onset dan R_offset (ms) untuk beat terdekat.
    - beat_index: indeks iterasi saat ini (fallback jika anchor_sample tidak ada)
    - anchor_sample: sampel acuan di sekitar kompleks (mis. rata-rata Q dan S)
    """
    # Ambil array R-peak, R-onset, R-offset (buang NaN, cast ke int)
    rpeaks_arr  = np.array([x for x in rpeaks['ECG_R_Peaks']   if not np.isnan(x)], dtype=int)
    r_onsets    = np.array([x for x in waves_dwt['ECG_R_Onsets']  if not np.isnan(x)], dtype=int)
    r_offsets   = np.array([x for x in waves_dwt['ECG_R_Offsets'] if not np.isnan(x)], dtype=int)
    if len(rpeaks_arr) == 0 or len(r_onsets) == 0 or len(r_offsets) == 0:
        return None

    # Tentukan beat R yang relevan
    if anchor_sample is not None:
        j = int(np.argmin(np.abs(rpeaks_arr - int(anchor_sample))))
    else:
        j = min(max(0, beat_index), len(rpeaks_arr) - 1)
    rp = rpeaks_arr[j]

    # Cari onset terakhir sebelum R dan offset pertama sesudah R
    onset_candidates  = r_onsets[r_onsets <= rp]
    offset_candidates = r_offsets[r_offsets >= rp]
    if len(onset_candidates) == 0 or len(offset_candidates) == 0:
        return None

    onset  = int(onset_candidates.max())
    offset = int(offset_candidates.min())
    if offset <= onset:
        return None

    return ((offset - onset) / Fs) * 1000.0  # ms


# <<< ADD: LAST-RESORT QT pairing helper >>>
def _qt_last_resort_pairing(R_onsets, T_marks, Fs, rr_ms=None,
                            min_abs_ms=160.0, max_abs_ms=600.0):
    """
    Pasangkan setiap R_onset dengan T_marks pertama yang > R_onset.
    Validasi secara fisiologis:
      - Jika RR diketahui: terima 20%–60% RR (dibatasi 160–600 ms)
      - Jika RR tidak ada: pakai 160–600 ms
    Return: (list_qt_ms, median_qt_ms atau np.nan)
    """
    R = np.asarray(R_onsets, dtype=float)
    T = np.asarray(T_marks, dtype=float)
    R = R[~np.isnan(R)]
    T = T[~np.isnan(T)]
    if len(R) == 0 or len(T) == 0:
        return [], np.nan

    # Gate fisiologis dinamis berbasis RR (opsional)
    if rr_ms is not None and np.isfinite(rr_ms):
        lo = max(min_abs_ms, 0.20 * rr_ms)
        hi = min(max_abs_ms, 0.60 * rr_ms)
    else:
        lo, hi = min_abs_ms, max_abs_ms

    qt_ms = []
    j = 0
    for r in R:
        # cari T pertama sesudah r
        while j < len(T) and T[j] <= r:
            j += 1
        if j >= len(T):
            break
        dt_ms = (T[j] - r) * 1000.0 / Fs
        if lo <= dt_ms <= hi:
            qt_ms.append(dt_ms)
        else:
            # coba T berikutnya jika kandidat pertama tidak masuk gate
            jj = j + 1
            chosen = False
            while jj < len(T):
                dt2 = (T[jj] - r) * 1000.0 / Fs
                if lo <= dt2 <= hi:
                    qt_ms.append(dt2)
                    j = jj
                    chosen = True
                    break
                jj += 1
            if not chosen:
                # biarkan j tetap, lanjut ke R berikutnya
                pass

    return qt_ms, (float(np.nanmedian(qt_ms)) if len(qt_ms) else np.nan)

def apply_iec_diagnostic_filter(signal, fs):
    """
    IEC 60601-2-25:2011 compliant diagnostic ECG filter (stabilized version)
    
    Tahapan:
    1. Baseline correction (detrending)
    2. High-pass filter (0.05 Hz)
    3. Low-pass filter (150 Hz)
    4. Notch filter (50 Hz)
    5. Edge padding untuk mencegah artefak
    """
    # Baseline correction (hilangkan offset DC)
    signal = signal - np.mean(signal)

    # Padding sinyal untuk menghindari artefak filtfilt di tepi
    pad_len = int(fs * 2)  # 2 detik padding
    padded_signal = np.pad(signal, (pad_len, pad_len), mode='edge')

    # Nyquist frequency
    nyquist = fs / 2.0

    # --- Step 1: High-pass filter 0.05 Hz
    hp_cutoff = 0.05 / nyquist
    hp_cutoff = min(hp_cutoff, 0.99)
    b_hp, a_hp = scipy.signal.butter(4, hp_cutoff, btype='highpass')
    signal_hp = scipy.signal.filtfilt(b_hp, a_hp, padded_signal)

    # --- Step 2: Low-pass filter 150 Hz
    lp_cutoff = min(150 / nyquist, 0.99)
    b_lp, a_lp = scipy.signal.butter(4, lp_cutoff, btype='lowpass')
    signal_lp = scipy.signal.filtfilt(b_lp, a_lp, signal_hp)

    # --- Step 3: Notch filter 50 Hz
    f0 = 50.0
    Q = 30.0  # Quality factor
    if f0 < nyquist:
        w0 = f0 / nyquist
        b_notch, a_notch = scipy.signal.iirnotch(w0, Q)
        signal_filtered = scipy.signal.filtfilt(b_notch, a_notch, signal_lp)
    else:
        signal_filtered = signal_lp

    # --- Remove padding
    final_signal = signal_filtered[pad_len:-pad_len]

    return final_signal

def apply_iec_monitoring_filter(signal, fs):
    """
    Apply IEC 60601-2-27 compliant monitoring ECG filter
    
    IEC 60601-2-27 specifies:
    - High-pass filter: 0.67 Hz (for monitoring mode)
    - Low-pass filter: 40 Hz (for monitoring applications)
    
    Args:
        signal: Input ECG signal
        fs: Sampling frequency
    
    Returns:
        filtered_signal: IEC compliant monitoring filtered ECG signal
    """
    
    nyquist = fs / 2
    
    # High-pass filter at 0.67 Hz (IEC 60601-2-27 monitoring requirement)
    high_cutoff = min(0.67 / nyquist, 0.99)
    b_high, a_high = scipy.signal.butter(4, high_cutoff, btype='highpass')
    signal_high_filtered = scipy.signal.filtfilt(b_high, a_high, signal)
    
    # Low-pass filter at 40 Hz (IEC 60601-2-27 monitoring requirement)
    low_cutoff = min(40 / nyquist, 0.99)
    b_low, a_low = scipy.signal.butter(4, low_cutoff, btype='lowpass')
    signal_filtered = scipy.signal.filtfilt(b_low, a_low, signal_high_filtered)
    
    return signal_filtered

while True:
    Raw_data = {"I": [], "II": [], "III": [], "AVR": [], "AVL": [], "AVF": [], "V1": [], "V2": [],"V3": [], "V4": [], "V5": [], "V6": [] }
    converted_data = {"I-mV": [], "II-mV": [], "III-mV": [], "AVR-mV": [], "AVL-mV": [], "AVF-mV": [], "V1-mV": [], "V2-mV": [], "V3-mV": [], "V4-mV": [], "V5-mV": [], "V6-mV": []}
    
    # Initialize all features for all leads
    leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    packedData = {}
    
    for lead in leads:
        packedData[f"rr_{lead.lower()}"] = 0
        packedData[f"rr_std_{lead.lower()}"] = 0
        packedData[f"pr_{lead.lower()}"] = 0
        packedData[f"qs_{lead.lower()}"] = 0
        packedData[f"qtc_{lead.lower()}"] = 0
        packedData[f"st_{lead.lower()}"] = 0
        packedData[f"rs_ratio_{lead.lower()}"] = 0
        packedData[f"heartrate_{lead.lower()}"] = 0
    
    print("Gathering data...")
    t_end = time.time() + timerecord

    while time.time() < t_end:
        chunk = serialPort.read(4096)      # baca dalam blok
        if not chunk:
            continue
        buf.extend(chunk)

        # buang noise sebelum STX
        while buf and buf[0] != STX:
            buf.pop(0)

        # proses semua frame lengkap di buffer
        while True:
            try:
                end = buf.index(ETX, 1)
            except ValueError:
                break  # belum ada ETX → tunggu data lagi

            frame = bytes(buf[1:end])      # isi antara STX..ETX
            del buf[:end+1]

           # langsung ambil semua angka dari frame (tanpa wajib “12,”)
            nums = float_re.findall(frame)
            if len(nums) < NUM_EXPECTED:
                continue  # frame terlalu pendek, skip

            # proses per 12 angka (kalau 24 → 2 sampel, dst)
            for i in range(0, len(nums), NUM_EXPECTED):
                if i + NUM_EXPECTED > len(nums):
                    break
                vals = [float(x) for x in nums[i:i+NUM_EXPECTED]]
                # ✅ print ke terminal biar kelihatan data mentah yang diterima
                print(f"[FRAME] {vals}")


                (value1, value2, value3, value4, value5, value6,
                value7, value8, value9, value10, value11, value12) = vals

                # append ke Raw_data & converted_data (tetap sama seperti punyamu)
                Raw_data["I"].append(value1);  Raw_data["II"].append(value2);  Raw_data["III"].append(value3)
                Raw_data["AVR"].append(value4); Raw_data["AVL"].append(value5); Raw_data["AVF"].append(value6)
                Raw_data["V1"].append(value7);  Raw_data["V2"].append(value8);  Raw_data["V3"].append(value9)
                Raw_data["V4"].append(value10); Raw_data["V5"].append(value11); Raw_data["V6"].append(value12)

                converted_data["I-mV"].append(convert_to_millivolts(value1))
                converted_data["II-mV"].append(convert_to_millivolts(value2))
                converted_data["III-mV"].append(convert_to_millivolts(value3))
                converted_data["AVR-mV"].append(convert_to_millivolts(value4))
                converted_data["AVL-mV"].append(convert_to_millivolts(value5))
                converted_data["AVF-mV"].append(convert_to_millivolts(value6))
                converted_data["V1-mV"].append(convert_to_millivolts(value7))
                converted_data["V2-mV"].append(convert_to_millivolts(value8))
                converted_data["V3-mV"].append(convert_to_millivolts(value9))
                converted_data["V4-mV"].append(convert_to_millivolts(value10))
                converted_data["V5-mV"].append(convert_to_millivolts(value11))
                converted_data["V6-mV"].append(convert_to_millivolts(value12))


    # Save raw gathered data to CSV
    df_raw = pd.DataFrame(Raw_data)
    file_path_raw = os.path.join(save_path, f'Data_{iteration}_raw.csv')
    df_raw.to_csv(file_path_raw, index=False)
    print(f'Raw data saved for iteration {iteration} at {file_path_raw}.')

    # Save converted data to CSV
    df_converted = pd.DataFrame(converted_data)
    file_path_converted = os.path.join(save_path, f'Data_{iteration}_converted.csv')
    df_converted.to_csv(file_path_converted, index=False)
    print(f'Converted data saved for iteration {iteration} at {file_path_converted}.')

    # Process each lead
    for channel in ['I-mV', 'II-mV', 'III-mV', 'AVR-mV', 'AVL-mV', 'AVF-mV', 'V1-mV', 'V2-mV', 'V3-mV', 'V4-mV', 'V5-mV', 'V6-mV']:
        try:
            # Get lead name without -mV suffix
            lead_name = channel.replace('-mV', '')
            lead_lower = lead_name.lower()
            
            ecgmv = np.asarray(converted_data[channel], dtype=float)
            t = np.arange(len(ecgmv), dtype=float)
            
            #Raw data plot
            plt.figure(figsize=(10, 6))
            plt.plot(t, ecgmv, color='blue')
            plt.title(f'Electrocardiogram Signal ({channel}) - Raw Data')
            plt.xlabel('Time [ms]')
            plt.ylabel('Amplitude [mV]')
            plt.grid(True)
            plt.tight_layout()
            # plt.show(block=False)
            # plt.pause(0.1)
            
           # Estimasi Fs seperti sebelumnya
            Fs = max(1, int(len(ecgmv) / 10))

            # 🔧 PRE-CLEAN: hilangkan spike & saturasi sebelum filter IEC
            ecgmv_clean = despike_pipeline(ecgmv, Fs)
            
            diff = ecgmv - ecgmv_clean
            num_spike = int(np.sum(np.abs(diff) > (0.002 * np.max(np.abs(ecgmv)+1e-9))))  # ambang heuristik
            if num_spike:
                print(f"⚠️  {channel}: {num_spike} sampel dikoreksi (spike/saturasi).")
                
            # =================================================================
            # PIPELINE: Raw Signal → IEC Filter → Baseline → Butterworth → FIR
            # =================================================================
            print(f"\nApplying IEC 60601-2-25:2011 filter for {channel}...")
            
            # STEP 1: Apply IEC 60601-2-25 Diagnostic Filter (setelah raw signal)
            iec_filtered_signal = apply_iec_diagnostic_filter(ecgmv_clean, Fs)
            
            # STEP 2: Baseline Correction (menggunakan hasil IEC filter)
            detr_ecg = scipy.signal.detrend(iec_filtered_signal, axis=-1, type='linear', bp=0, overwrite_data=False)

            # Subplot for all filtering stages
            plt.figure(figsize=(15, 18))

            # Raw signal
            plt.subplot(5, 1, 1)
            plt.plot(t, ecgmv, color='blue')
            plt.title(f'Signal ({channel}) - Raw Data')
            plt.xlabel('Time [ms]')
            plt.ylabel('Amplitude [mV]')
            plt.grid(True)

            # IEC 60601-2-25 Diagnostic Filter
            plt.subplot(5, 1, 2)
            plt.plot(t, iec_filtered_signal, color='red', linewidth=2)
            plt.title(f'Signal ({channel}) - IEC 60601-2-25:2011 Diagnostic Filter (0.05-150 Hz + 50Hz notch)')
            plt.xlabel('Time [ms]')
            plt.ylabel('Amplitude [mV]')
            plt.grid(True)

            # STEP 3: Baseline Correction plot
            plt.subplot(5, 1, 3)
            plt.plot(t, detr_ecg, color='yellow')
            plt.title(f'Signal ({channel}) - Baseline Correction (setelah IEC filter)')
            plt.xlabel('Time [ms]')
            plt.ylabel('Amplitude [mV]')
            plt.grid(True)

            # y-axis (menggunakan hasil baseline correction yang sudah melalui IEC filter)
            y = [e for e in detr_ecg]
            N = len(y)
            T = 1.0 / Fs
            # Compute x-axis
            x = np.linspace(0.0, N*T, N)
            # Compute FFT
            yf = scipy.fftpack.fft(y)
            # Compute frequency x-axis
            xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
            
            # STEP 4: Butterworth filter (dilanjutkan dari baseline correction)
            b, a = scipy.signal.butter(4, 0.6, 'low')
            tempf_butter = scipy.signal.filtfilt(b, a, y)
            # b, a = scipy.signal.butter(4, 60, btype='low')
            # tempf_butter = scipy.signal.filtfilt(b, a, y)  # zero-phase
            

            # Butterworth plot
            plt.subplot(5, 1, 4)
            plt.plot(t, tempf_butter, color='green')
            plt.title(f'Signal ({channel}) - Butterworth Filtered (setelah IEC + Baseline)')
            plt.xlabel('Time [ms]')
            plt.ylabel('Amplitude [mV]')
            plt.grid(True)

            # STEP 5: FIR filter (tahap terakhir sebelum feature extraction)
            Fsf = int(len(converted_data[channel]) / 10)
            cutoff_hz = 0.05  # Cutoff frequency for FIR filter
            min_Fsf = 2 * cutoff_hz
            if Fsf < min_Fsf:
                Fsf = min_Fsf

            nyq_rate = Fsf / 2
            width = 5.0 / nyq_rate
            ripple_db = 60.0
            O, beta = scipy.signal.kaiserord(ripple_db, width)
            if O % 2 == 0:
                O += 1
            else:
                O, beta = scipy.signal.kaiserord(ripple_db, width)
            taps = scipy.signal.firwin(O, cutoff_hz/nyq_rate, window=('kaiser', beta), pass_zero=False)
            y_filt = scipy.signal.lfilter(taps, 1.0, tempf_butter)

            #  kompensasi delay FIR (kalau tetap ingin lfilter):
            n_delay = (len(taps) - 1) // 2
            y_filt = np.roll(y_filt, -n_delay)
            if n_delay > 0:
                y_filt[-n_delay:] = y_filt[-n_delay-1]  # hold last value
            # # (alternatif paling bersih: y_filt = scipy.signal.filtfilt(taps, [1.0], tempf_butter))
        
            # FIR plot (hasil akhir untuk feature extraction)
            plt.subplot(5, 1, 5)
            plt.plot(t, y_filt, color='orange')
            plt.title(f'Signal ({channel}) - FIR Filtered (Final signal untuk feature extraction)')
            plt.xlabel('Time [ms]')
            plt.ylabel('Amplitude [mV]')
            plt.grid(True)

            plt.tight_layout()
            #plt.show()

            # =================================================================
            # FEATURE EXTRACTION menggunakan ALGORITMA ORIGINAL
            # Pipeline: Raw → IEC → Baseline → Butterworth → FIR → Original Features
            # =================================================================

            # PQRST Peak Detection
            try:
                # PQRST Peak Detection
                _, rpeaks = nk.ecg_peaks(y_filt, sampling_rate=Fs)
                signal_dwt, waves_dwt = nk.ecg_delineate(y_filt, rpeaks, sampling_rate=Fs, method="dwt")

                # Remove Nan and change to ndarray int (PERSIS SEPERTI ORIGINAL)
                peaksR = np.array([x for x in rpeaks['ECG_R_Peaks'] if math.isnan(x) is False]).astype(int)
                peaksP = np.array([x for x in waves_dwt['ECG_P_Peaks'] if math.isnan(x) is False]).astype(int)
                peaksQ = np.array([x for x in waves_dwt['ECG_Q_Peaks'] if math.isnan(x) is False]).astype(int)
                peaksS = np.array([x for x in waves_dwt['ECG_S_Peaks'] if math.isnan(x) is False]).astype(int)
                peaksT = np.array([x for x in waves_dwt['ECG_T_Peaks'] if math.isnan(x) is False]).astype(int)
                peaksPOnsets = np.array([x for x in waves_dwt['ECG_P_Onsets'] if math.isnan(x) is False]).astype(int)
                peaksPOffsets = np.array([x for x in waves_dwt['ECG_P_Offsets'] if math.isnan(x) is False]).astype(int)
                peaksROnsets = np.array([x for x in waves_dwt['ECG_R_Onsets'] if math.isnan(x) is False]).astype(int)
                peaksROffsets = np.array([x for x in waves_dwt['ECG_R_Offsets'] if math.isnan(x) is False]).astype(int)
                peaksTOnsets = np.array([x for x in waves_dwt['ECG_T_Onsets'] if math.isnan(x) is False]).astype(int)
                peaksTOffsets = np.array([x for x in waves_dwt['ECG_T_Offsets'] if math.isnan(x) is False]).astype(int)

                # Remove Nan and change to ndarray int (PERSIS SEPERTI ORIGINAL)
                rpeaks['ECG_R_Peaks'] = np.array([x for x in rpeaks['ECG_R_Peaks'] if math.isnan(x) is False]).astype(int)
                waves_dwt['ECG_P_Peaks'] = np.array([x for x in waves_dwt['ECG_P_Peaks'] if math.isnan(x) is False]).astype(int)
                waves_dwt['ECG_Q_Peaks'] = np.array([x for x in waves_dwt['ECG_Q_Peaks'] if math.isnan(x) is False]).astype(int)
                waves_dwt['ECG_S_Peaks'] = np.array([x for x in waves_dwt['ECG_S_Peaks'] if math.isnan(x) is False]).astype(int)
                waves_dwt['ECG_T_Peaks'] = np.array([x for x in waves_dwt['ECG_T_Peaks'] if math.isnan(x) is False]).astype(int)
                waves_dwt['ECG_P_Onsets'] = np.array([x for x in waves_dwt['ECG_P_Onsets'] if math.isnan(x) is False]).astype(int)
                waves_dwt['ECG_P_Offsets'] = np.array([x for x in waves_dwt['ECG_P_Offsets'] if math.isnan(x) is False]).astype(int)
                waves_dwt['ECG_R_Onsets'] = np.array([x for x in waves_dwt['ECG_R_Onsets'] if math.isnan(x) is False]).astype(int)
                waves_dwt['ECG_R_Offsets'] = np.array([x for x in waves_dwt['ECG_R_Offsets'] if math.isnan(x) is False]).astype(int)
                waves_dwt['ECG_T_Onsets'] = np.array([x for x in waves_dwt['ECG_T_Onsets'] if math.isnan(x) is False]).astype(int)
                waves_dwt['ECG_T_Offsets'] = np.array([x for x in waves_dwt['ECG_T_Offsets'] if math.isnan(x) is False]).astype(int)

                # ================================================================
                # CORRECTIONS SEPERTI ORIGINAL (tidak terlalu ketat)
                # ================================================================
                
                # Correcting first cycle (PERSIS SEPERTI ORIGINAL)
                if len(rpeaks['ECG_R_Peaks']) > 0 and len(waves_dwt['ECG_P_Onsets']) > 0:
                    if rpeaks['ECG_R_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
                        rpeaks['ECG_R_Peaks'] = np.delete(rpeaks['ECG_R_Peaks'], 0)
                
                if len(waves_dwt['ECG_P_Peaks']) > 0 and len(waves_dwt['ECG_P_Onsets']) > 0:
                    if waves_dwt['ECG_P_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
                        waves_dwt['ECG_P_Peaks'] = np.delete(waves_dwt['ECG_P_Peaks'], 0)
                
                if len(waves_dwt['ECG_Q_Peaks']) > 0 and len(waves_dwt['ECG_P_Onsets']) > 0:
                    if waves_dwt['ECG_Q_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
                        waves_dwt['ECG_Q_Peaks'] = np.delete(waves_dwt['ECG_Q_Peaks'], 0)
                
                if len(waves_dwt['ECG_S_Peaks']) > 0 and len(waves_dwt['ECG_P_Onsets']) > 0:
                    if waves_dwt['ECG_S_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
                        waves_dwt['ECG_S_Peaks'] = np.delete(waves_dwt['ECG_S_Peaks'], 0)
                
                if len(waves_dwt['ECG_T_Peaks']) > 0 and len(waves_dwt['ECG_P_Onsets']) > 0:
                    if waves_dwt['ECG_T_Peaks'][0] < waves_dwt['ECG_P_Onsets'][0]:
                        waves_dwt['ECG_T_Peaks'] = np.delete(waves_dwt['ECG_T_Peaks'], 0)

                # Additional corrections (PERSIS SEPERTI ORIGINAL)
                if len(rpeaks['ECG_R_Peaks']) > 1:
                    if y_filt[rpeaks['ECG_R_Peaks']][0] < y_filt[rpeaks['ECG_R_Peaks']][1]/2:
                        rpeaks['ECG_R_Peaks'] = np.delete(rpeaks['ECG_R_Peaks'], 0)
                        if len(waves_dwt['ECG_P_Peaks']) > 0:
                            waves_dwt['ECG_P_Peaks'] = np.delete(waves_dwt['ECG_P_Peaks'], 0)
                        if len(waves_dwt['ECG_Q_Peaks']) > 0:
                            waves_dwt['ECG_Q_Peaks'] = np.delete(waves_dwt['ECG_Q_Peaks'], 0)
                        if len(waves_dwt['ECG_S_Peaks']) > 0:
                            waves_dwt['ECG_S_Peaks'] = np.delete(waves_dwt['ECG_S_Peaks'], 0)
                        if len(waves_dwt['ECG_T_Peaks']) > 0:
                            waves_dwt['ECG_T_Peaks'] = np.delete(waves_dwt['ECG_T_Peaks'], 0)

                # ================================================================
                # FEATURE EXTRACTION DENGAN ALGORITMA 100% ORIGINAL
                # ================================================================

                # Initialize variables for this lead
                RR_avg = 0
                RR_stdev = 0
                PR_avg = 0
                QS_avg = 0
                QT_avg = 0
                QTc_avg = 0
                ST_avg = 0
                RS_ratio = 0
                bpm = 0
                RR_list = []

                # 1. RR INTERVAL (ALGORITMA 100% ORIGINAL)
                if len(rpeaks['ECG_R_Peaks']) > 1:
                    RR_list = []
                    cnt = 0
                    while (cnt < (len(rpeaks['ECG_R_Peaks']) - 1)):
                        RR_interval = (rpeaks['ECG_R_Peaks'][cnt + 1] - rpeaks['ECG_R_Peaks'][cnt])
                        RRms_dist = ((RR_interval / Fs) * 1000.0)  # Convert sample distances to ms distances
                        RR_list.append(RRms_dist)
                        cnt += 1
                    
                    if len(RR_list) > 0:
                        dfRR = pd.DataFrame(RR_list)
                        dfRR = dfRR.fillna(0)
                        RR_stdev = np.std(RR_list, axis=None)  # Save stdev
                        
                        # Manual averaging seperti original
                        sum_rr = 0.0
                        count_rr = 0.0
                        for index in range(len(RR_list)):
                            if (np.isnan(RR_list[index]) == True):
                                continue
                            else:
                                sum_rr += RR_list[index]
                                count_rr += 1
                        
                        if count_rr > 0:
                            RR_avg = (sum_rr / count_rr)
                            bpm = 60000 / np.mean(RR_list)  # BPM calculation

                # 2. PR INTERVAL (ALGORITMA 100% ORIGINAL)
                if len(waves_dwt['ECG_R_Onsets']) > 1 and len(waves_dwt['ECG_P_Onsets']) > 1:
                    PR_peak_list = []
                    idex = ([x for x in range(0, len(waves_dwt['ECG_R_Onsets']) - 1)])
                    for i in idex:
                        if waves_dwt['ECG_R_Onsets'][i] < waves_dwt['ECG_P_Onsets'][i]:
                            cnt = 0
                            while (cnt < (len(waves_dwt['ECG_R_Onsets']) - 1)):
                                if cnt < len(waves_dwt['ECG_Q_Peaks']):
                                    PR_peak_interval = (waves_dwt['ECG_Q_Peaks'][cnt] - waves_dwt['ECG_P_Onsets'][cnt])
                                    ms_dist = ((PR_peak_interval / Fs) * 1000.0)
                                    PR_peak_list.append(ms_dist)
                                cnt += 1
                        else:
                            cnt = 0
                            while (cnt < (len(waves_dwt['ECG_R_Onsets']) - 1)):
                                PR_peak_interval = (waves_dwt['ECG_R_Onsets'][cnt] - waves_dwt['ECG_P_Onsets'][cnt])
                                ms_dist = ((PR_peak_interval / Fs) * 1000.0)
                                PR_peak_list.append(ms_dist)
                                cnt += 1
                    
                    if len(PR_peak_list) > 0:
                        dfPR = pd.DataFrame(PR_peak_list)
                        dfPR = dfPR.fillna(0)
                        
                        # Manual averaging seperti original
                        sum_pr = 0.0
                        count_pr = 0.0
                        for index in range(len(PR_peak_list)):
                            if (np.isnan(PR_peak_list[index]) == True):
                                continue
                            else:
                                sum_pr += PR_peak_list[index]
                                count_pr += 1
                        
                        if count_pr > 0:
                            PR_avg = (sum_pr / count_pr)
                
                # 3. QS INTERVAL (ALGORITMA 100% ORIGINAL)
                if len(waves_dwt['ECG_S_Peaks']) > 1 and len(waves_dwt['ECG_Q_Peaks']) > 1:
                    QS_peak_list = []
                    try:
                        idex = [x for x in range(0, len(waves_dwt['ECG_S_Peaks']) - 1)]
                        for i in idex:
                            # Hitung QS berbasis Q & S seperti versi asli
                            if waves_dwt['ECG_S_Peaks'][i] < waves_dwt['ECG_Q_Peaks'][i]:
                                qs_samples = (waves_dwt['ECG_S_Peaks'][i + 1] - waves_dwt['ECG_Q_Peaks'][i])
                            else:
                                qs_samples = (waves_dwt['ECG_S_Peaks'][i] - waves_dwt['ECG_Q_Peaks'][i])

                            ms_dist = ((qs_samples / Fs) * 1000.0)

                            # === FALLBACK: jika QS = 0 ms, pakai R_offset - R_onset beat terkait ===
                            if not np.isnan(ms_dist) and (abs(ms_dist) < 1e-9):
                                # Anchor di sekitar kompleks: pakai rata-rata Q & S (abaikan NaN)
                                q_i = waves_dwt['ECG_Q_Peaks'][i]
                                s_i = waves_dwt['ECG_S_Peaks'][i]
                                anchor_vals = [v for v in [q_i, s_i] if not np.isnan(v)]
                                anchor = float(np.nanmean(anchor_vals)) if len(anchor_vals) > 0 else None

                                fallback_ms = _qrs_ms_from_Ronset_Roffset(i, anchor, waves_dwt, rpeaks, Fs)
                                if fallback_ms is not None:
                                    ms_dist = fallback_ms  # gantikan dengan Roffset-Ronset

                            QS_peak_list.append(ms_dist)

                        if len(QS_peak_list) > 0:
                            dfQS = pd.DataFrame(QS_peak_list).fillna(0)

                            # Manual averaging seperti original
                            sum_qs = 0.0
                            count_qs = 0.0
                            for index in range(len(QS_peak_list)):
                                if np.isnan(QS_peak_list[index]):
                                    continue
                                else:
                                    sum_qs += QS_peak_list[index]
                                    count_qs += 1

                            if count_qs > 0:
                                QS_avg = (sum_qs / count_qs)

                    except:
                        print(f"QRS width Error for {lead_name}")

                # 4. QT/QTc INTERVAL (ALGORITMA 100% ORIGINAL)
                if len(waves_dwt['ECG_T_Offsets']) > 1 and len(waves_dwt['ECG_R_Onsets']) > 1:
                    QT_peak_list = []
                    try:
                        idex = ([x for x in range(0, len(waves_dwt['ECG_T_Offsets']) - 1)])
                        for i in idex:
                            if waves_dwt['ECG_T_Offsets'][i] < waves_dwt['ECG_R_Onsets'][i]:
                                QTdeff = (waves_dwt['ECG_T_Offsets'][i + 1] - waves_dwt['ECG_R_Onsets'][i])
                                ms_dist = ((QTdeff / Fs) * 1000.0)
                                QT_peak_list.append(ms_dist)
                            else:
                                QTdeff = (waves_dwt['ECG_T_Offsets'][i] - waves_dwt['ECG_R_Onsets'][i])
                                ms_dist = ((QTdeff / Fs) * 1000.0)
                                QT_peak_list.append(ms_dist)
                        
                        if len(QT_peak_list) > 0:
                            dfQT = pd.DataFrame(QT_peak_list)
                            dfQT = dfQT.fillna(0)
                            
                            # Manual averaging seperti original
                            sum_qt = 0.0
                            count_qt = 0.0
                            for index in range(len(QT_peak_list)):
                                if (np.isnan(QT_peak_list[index]) == True):
                                    continue
                                else:
                                    sum_qt += QT_peak_list[index]
                                    count_qt += 1
                            
                            if count_qt > 0:
                                QT_avg = (sum_qt / count_qt)
                                
                                # QTc calculation seperti original
                                if len(RR_list) > 0:
                                    QTc_avg = QT_avg / (math.sqrt(np.mean(RR_list)/1000))
                    except:
                        print(f"QT Interval Error for {lead_name}")
                    
                        # <<< ADD: LAST-RESORT FIX untuk QT bila masih 0/kosong >>>
                    try:
                        # kondisi perlu perbaikan: QT_avg 0/NaN atau list kosong/semua mendekati 0
                        need_qt_fix = False
                        if ('QT_avg' in locals()) is False:
                            need_qt_fix = True
                        else:
                            if (not np.isfinite(QT_avg)) or (abs(QT_avg) < 1e-9):
                                need_qt_fix = True

                        # kalau ada list hasil sebelumnya, cek apakah semuanya ~0
                        if ('QT_peak_list' in locals()) and (len(QT_peak_list) > 0):
                            if np.allclose(np.nan_to_num(QT_peak_list, nan=0.0), 0.0, atol=1e-9):
                                need_qt_fix = True
                        else:
                            # tidak ada list sama sekali → perlu fix
                            need_qt_fix = True

                        if need_qt_fix:
                            rr_ms_mean = None
                            if 'RR_list' in locals() and len(RR_list) > 0:
                                rr_ms_mean = float(np.mean(RR_list))

                            # Coba pairing dengan T_Offsets (prioritas)
                            qt_list_fb, qt_avg_fb = _qt_last_resort_pairing(
                                waves_dwt['ECG_R_Onsets'],
                                waves_dwt['ECG_T_Offsets'],
                                Fs,
                                rr_ms=rr_ms_mean
                            )

                            # Jika gagal, coba pairing dengan T_Peaks
                            if (not np.isfinite(qt_avg_fb)) or (len(qt_list_fb) == 0):
                                if 'ECG_T_Peaks' in waves_dwt and len(waves_dwt['ECG_T_Peaks']) > 0:
                                    qt_list_fb, qt_avg_fb = _qt_last_resort_pairing(
                                        waves_dwt['ECG_R_Onsets'],
                                        waves_dwt['ECG_T_Peaks'],
                                        Fs,
                                        rr_ms=rr_ms_mean
                                    )

                            # Terapkan jika berhasil
                            if np.isfinite(qt_avg_fb):
                                QT_avg = float(qt_avg_fb)
                                if rr_ms_mean is not None and rr_ms_mean > 0:
                                    QTc_avg = QT_avg / math.sqrt(rr_ms_mean / 1000.0)
                                # opsional: logging singkat
                                print(f"[QT-fix:{lead_name}] last-resort applied "
                                    f"(n={len(qt_list_fb)}), QT≈{QT_avg:.1f} ms, QTc≈{QTc_avg:.1f} ms")
                    except Exception as _e_qtfix:
                        print(f"[QT-fix:{lead_name}] fallback error: {_e_qtfix}")

                # 5. ST Interval (ALGORITMA 100% ORIGINAL)
                if len(waves_dwt['ECG_T_Offsets']) > 1 and len(waves_dwt['ECG_R_Offsets']) > 1:
                    ST_peak_list = []
                    try:
                        for i in ([x for x in range(0, len(waves_dwt['ECG_T_Offsets']) - 1)]):
                            if waves_dwt['ECG_T_Offsets'][i] < waves_dwt['ECG_R_Offsets'][i]:
                                cnt = 0
                                while (cnt < (len(waves_dwt['ECG_T_Offsets']) - 1)):
                                    ST_peak_interval = (waves_dwt['ECG_T_Offsets'][cnt+1] - waves_dwt['ECG_R_Offsets'][cnt])
                                    ms_dist = ((ST_peak_interval / Fs) * 1000.0)
                                    ST_peak_list.append(ms_dist)
                                    cnt += 1
                            else:
                                cnt = 0
                                while (cnt < (len(waves_dwt['ECG_T_Offsets']) - 1)):
                                    ST_peak_interval = (waves_dwt['ECG_T_Offsets'][cnt] - waves_dwt['ECG_R_Offsets'][cnt])
                                    ms_dist = ((ST_peak_interval / Fs) * 1000.0)
                                    ST_peak_list.append(ms_dist)
                                    cnt += 1
                        
                        if len(ST_peak_list) > 0:
                            dfST = pd.DataFrame(ST_peak_list)
                            dfST = dfST.fillna(0)
                            
                            # Manual averaging seperti original
                            sum_st = 0.0
                            count_st = 0.0
                            for index in range(len(ST_peak_list)):
                                if (np.isnan(ST_peak_list[index]) == True):
                                    continue
                                else:
                                    sum_st += ST_peak_list[index]
                                    count_st += 1
                            
                            if count_st > 0:
                                ST_avg = (sum_st / count_st)
                    except:
                        print(f"Error in calculating ST interval for {lead_name}")

                # 6. R/S RATIO (ALGORITMA 100% ORIGINAL)
                if 'ECG_S_Peaks' in waves_dwt.keys() and len(waves_dwt['ECG_S_Peaks']) > 0 and len(rpeaks['ECG_R_Peaks']) > 0:
                    try:
                        R_mean_amp = np.mean([y_filt[int(i)] for i in rpeaks['ECG_R_Peaks']])
                        S_mean_amp = np.mean([y_filt[int(i)] for i in waves_dwt['ECG_S_Peaks']])
                        
                        if S_mean_amp != 0:  # Avoid division by zero
                            RS_ratio = (R_mean_amp) / abs(S_mean_amp)
                    except Exception as e:
                        print(f"Error calculating R/S ratio for {lead_name}: {e}")

                # Store calculated features for this lead
                packedData[f"rr_{lead_lower}"] = RR_avg
                packedData[f"rr_std_{lead_lower}"] = RR_stdev
                packedData[f"pr_{lead_lower}"] = PR_avg
                packedData[f"qs_{lead_lower}"] = QS_avg
                packedData[f"qtc_{lead_lower}"] = QTc_avg
                packedData[f"st_{lead_lower}"] = ST_avg
                packedData[f"rs_ratio_{lead_lower}"] = RS_ratio
                packedData[f"heartrate_{lead_lower}"] = bpm

                # Print results for this lead dengan format original
                print(f'\n{lead_name} - HASIL DENGAN ALGORITMA ORIGINAL + IEC FILTER')
                print(f'{lead_name} - RR Interval (ms) - Mean: {RR_avg:.2f}, Standard Deviation: {RR_stdev:.2f}')
                print(f'{lead_name} - PR Interval (ms) - Mean: {PR_avg:.2f}')
                print(f'{lead_name} - QS Interval (ms) - Mean: {QS_avg:.2f}')
                print(f'{lead_name} - QT Interval (ms) - Mean: {QT_avg:.2f}')
                print(f'{lead_name} - QTc Interval (ms) - Mean: {QTc_avg:.2f}')
                print(f'{lead_name} - ST Interval (ms) - Mean: {ST_avg:.2f}')
                print(f'{lead_name} - R/S Ratio: {RS_ratio:.4f}')
                print(f'{lead_name} - BPM: {bpm:.2f}')

            except Exception as e:
                print(f"Error in peak detection for {channel}: {e}")
            
            

            # # Show PQRST plot for the last processed lead (V6) with complete pipeline
            # if channel == 'V6-mV':
            #     plt.figure(figsize=(15, 8))
            #     plt.title(f'Electrocardiogram Signal ({channel}) - PQRST Peaks Detection\n(Pipeline: Raw → IEC 60601-2-25 → Baseline → Butterworth → FIR + Original Algorithms)')
            #     plt.plot(y_filt, color='orange', label="Final Filtered Signal (IEC + Pipeline)", linewidth=2)
            #     if 'rpeaks' in locals() and len(rpeaks['ECG_R_Peaks']) > 0:
            #         plt.plot(rpeaks['ECG_R_Peaks'], y_filt[rpeaks['ECG_R_Peaks']], "x", color='black', markersize=8, label="R Peak")
            #     if 'waves_dwt' in locals():
            #         if len(waves_dwt['ECG_P_Peaks']) > 0:
            #             plt.plot(waves_dwt['ECG_P_Peaks'], y_filt[waves_dwt['ECG_P_Peaks']], "x", color='blue', markersize=8, label="P Peak")
            #         if len(waves_dwt['ECG_Q_Peaks']) > 0:
            #             plt.plot(waves_dwt['ECG_Q_Peaks'], y_filt[waves_dwt['ECG_Q_Peaks']], "x", color='green', markersize=8, label="Q Peak")
            #         if len(waves_dwt['ECG_S_Peaks']) > 0:
            #             plt.plot(waves_dwt['ECG_S_Peaks'], y_filt[waves_dwt['ECG_S_Peaks']], "x", color='orange', markersize=8, label="S Peak")
            #         if len(waves_dwt['ECG_T_Peaks']) > 0:
            #             plt.plot(waves_dwt['ECG_T_Peaks'], y_filt[waves_dwt['ECG_T_Peaks']], "x", color='purple', markersize=8, label="T Peak")
            #     plt.xlabel('Time [ms]')
            #     plt.ylabel('Amplitude [mV]')
            #     plt.legend(loc="lower left")
            #     plt.grid(True, alpha=0.3)
                
            #     # Add annotations
            #     if 'waves_dwt' in locals():
            #         for i, j in zip(waves_dwt['ECG_P_Peaks'], y_filt[waves_dwt['ECG_P_Peaks']]):
            #             plt.annotate('P', xy=(i, j), fontsize=10, fontweight='bold')
            #         for i, j in zip(waves_dwt['ECG_Q_Peaks'], y_filt[waves_dwt['ECG_Q_Peaks']]):
            #             plt.annotate('Q', xy=(i, j), fontsize=10, fontweight='bold')
            #         if 'rpeaks' in locals():
            #             for i, j in zip(rpeaks['ECG_R_Peaks'], y_filt[rpeaks['ECG_R_Peaks']]):
            #                 plt.annotate('R', xy=(i, j), fontsize=10, fontweight='bold')
            #         for i, j in zip(waves_dwt['ECG_S_Peaks'], y_filt[waves_dwt['ECG_S_Peaks']]):
            #             plt.annotate('S', xy=(i, j), fontsize=10, fontweight='bold')
            #         for i, j in zip(waves_dwt['ECG_T_Peaks'], y_filt[waves_dwt['ECG_T_Peaks']]):
            #             plt.annotate('T', xy=(i, j), fontsize=10, fontweight='bold')

        except Exception as e:
            print(f"Error processing {channel}: {e}")

    
    # ——— Integrasi Deteksi untuk SEMUA fitur (96 fitur = 8×12) ———
    # Urutan kolom konsisten dengan struktur yang kamu pakai saat save Excel:
    # rr_i, rr_std_i, pr_i, qs_i, qtc_i, st_i, rs_ratio_i, heartrate_i, lalu lanjut ke lead berikutnya.
    FEATURE_FAMILIES = ['rr', 'rr_std', 'pr', 'qs', 'qtc', 'st', 'rs_ratio', 'heartrate']
    EXPECTED_FEATURES = [f"{fam}_{lead.lower()}" for lead in leads for fam in FEATURE_FAMILIES]
    # Bangun DataFrame dengan NAMA KOLOM yang konsisten
    X = pd.DataFrame(
        [{k: packedData.get(k, 0.0) for k in EXPECTED_FEATURES}],
        columns=EXPECTED_FEATURES
    ).astype(float)

    # Samakan SET & URUTAN kolom persis seperti saat scaler di-fit
    if hasattr(scaler, "feature_names_in_"):
        missing = [c for c in scaler.feature_names_in_ if c not in X.columns]
        extra   = [c for c in X.columns if c not in scaler.feature_names_in_]
        if missing:
            print("⚠️ Kolom yang diharapkan scaler tapi tidak ada:", missing)
        if extra:
            print("ℹ️ Kolom ekstra (akan diabaikan):", extra)

        # Reindex sesuai urutan scaler; kolom yang hilang diisi 0.0
        X = X.reindex(columns=scaler.feature_names_in_, fill_value=0.0)


    # Scale → Prediksi label: "Normal" / "Potential Fast Arrhytmia" / "Potential Heart Block"
    try:
        X_scaled = scaler.transform(X)
        # feature_for_model = feature_vector
        cluster_label = predict_clusters_xgb_integrated(
            X_scaled,
            model_xgb,
            kmeans_abnormal
        )[0]
        # Simpan ke packedData (ikut tersimpan ke Excel)
        packedData["cluster_label"] = cluster_label
        print(f"Predicted Cluster Label: {cluster_label}")
    except Exception as e:
        print(f"❌ Gagal melakukan prediksi klaster: {e}")
       

    # plt.show()
    
    # #================= END OF PROCESSING 12 LEADS =================
    # Save results to Excel with original algorithm naming
    if iteration == 1:
        df = pd.DataFrame([packedData])
        df.to_excel("D:\EKG\Ekstraksi Fitur 12 Lead\Anak WJ\Test rekam 38 (uji lab2)/HASILEKG12LEAD_ORIGINAL_ALGORITHM_IEC_REALTIME.xlsx", index=False)
        print(f"\nExtracted features saved with ORIGINAL ALGORITHM + IEC FILTER (Real-time). Total features: {len(packedData)}")
        print(f"Features per lead: 8")
        print(f"Total leads processed: 12")
        print(f"Pipeline: Raw → IEC 60601-2-25 → Baseline → Butterworth → FIR → Original Feature Extraction")
    elif iteration > 1:
        df = pd.DataFrame([packedData])
        filepath = r'D:\EKG\Ekstraksi Fitur 12 Lead\Anak WJ\Test rekam 38 (uji lab2)/HASILEKG12LEAD_ORIGINAL_ALGORITHM_IEC_REALTIME.xlsx'
        with pd.ExcelWriter(
                filepath,
                engine='openpyxl',
                mode='a',
                if_sheet_exists='overlay') as writer:
            reader = pd.read_excel('D:\EKG\Ekstraksi Fitur 12 Lead\Anak WJ\Test rekam 38 (uji lab2)/HASILEKG12LEAD_ORIGINAL_ALGORITHM_IEC_REALTIME.xlsx')
            df.to_excel(
                writer,
                startrow=reader.shape[0] + 1,
                index=False,
                header=False)
    
    iteration += 1
    sleep(2)
    
serialPort.close()
