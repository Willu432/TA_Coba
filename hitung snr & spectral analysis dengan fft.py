import pandas as pd
import numpy as np
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt
import os


# <<< ADD: constants konversi ADC → mV (sama seperti real-time)
VREF = 2.42       # volt
GAIN = 200        # sesuai setelan ADS1293
RESOLUTION = 24
LSB_SIZE = VREF / (2**RESOLUTION - 1)

def to_mV(adc):
    return adc * LSB_SIZE * 1000.0 / GAIN

# <<< ADD: deteksi & perbaikan spike
def despike_hampel(x, k=7, nsigma=6.0):
    """
    Hampel filter di domain waktu untuk mendeteksi outlier tajam.
    k: half-window (total window = 2k+1)
    nsigma: threshold berbasis MAD
    """
    x = np.asarray(x, dtype=float)
    med = scipy.signal.medfilt(x, kernel_size=2*k+1)
    diff = np.abs(x - med)
    # MAD yang robust
    mad = scipy.signal.medfilt(diff, kernel_size=2*k+1)
    mad = np.maximum(mad, 1e-12)             # cegah div/zero
    mask = diff > (nsigma * 1.4826 * mad)    # 1.4826 ≈ k MAD→σ
    # interpolasi linear di sampel outlier
    xi = x.copy()
    if np.any(mask):
        idx = np.arange(len(x))
        xi[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return xi, mask

# <<< ADD: deteksi saturasi dekat tepi ADC (±FS)
ADC_EDGE = 0.95 * (2**23 - 1)  # 95% full-scale 24-bit signed
def desaturate_edges(x):
    x = np.asarray(x, dtype=float)
    mask = (np.abs(x) >= ADC_EDGE)
    xi = x.copy()
    if np.any(mask):
        idx = np.arange(len(x))
        xi[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return xi, mask


def estimate_snr(signal, fs):
    """
    Mengestimasi SNR dengan asumsi noise berada di atas frekuensi 40 Hz.
    """
    try:
        nyquist = fs / 2.0
        b, a = scipy.signal.butter(4, 40 / nyquist, btype='low')
        estimated_signal_component = scipy.signal.filtfilt(b, a, signal)
        estimated_noise_component = signal - estimated_signal_component
        power_signal = np.mean(estimated_signal_component ** 2)
        power_noise = np.mean(estimated_noise_component ** 2)
        if power_noise == 0:
            return float('inf')
        snr = 10 * np.log10(power_signal / power_noise)
        return snr
    except Exception as e:
        print(f"Error calculating SNR: {e}")
        return np.nan

def calculate_fft(signal, fs):
    """
    Menghitung Fast Fourier Transform (FFT) dari sebuah sinyal.
    """
    N = len(signal)
    T = 1.0 / fs
    yf = scipy.fft.fft(signal)
    xf = scipy.fft.fftfreq(N, T)[:N//2]
    yf_abs = 2.0/N * np.abs(yf[0:N//2])
    return xf, yf_abs

def apply_iec_diagnostic_filter(signal, fs):
    """
    Menerapkan filter diagnostik EKG sesuai standar IEC 60601-2-25:2011.
    """
    signal = signal - np.mean(signal)
    pad_len = int(fs * 2)
    padded_signal = np.pad(signal, (pad_len, pad_len), mode='edge')
    nyquist = fs / 2.0
    
    hp_cutoff = 0.05 / nyquist
    b_hp, a_hp = scipy.signal.butter(4, min(hp_cutoff, 0.99), btype='highpass')
    signal_hp = scipy.signal.filtfilt(b_hp, a_hp, padded_signal)
    
    lp_cutoff = 150 / nyquist
    b_lp, a_lp = scipy.signal.butter(4, min(lp_cutoff, 0.99), btype='lowpass')
    signal_lp = scipy.signal.filtfilt(b_lp, a_lp, signal_hp)
    
    f0 = 50.0
    if f0 < nyquist:
        w0 = f0 / nyquist
        Q = 30.0
        b_notch, a_notch = scipy.signal.iirnotch(w0, Q)
        signal_filtered = scipy.signal.filtfilt(b_notch, a_notch, signal_lp)
    else:
        signal_filtered = signal_lp
        
    return signal_filtered[pad_len:-pad_len]

# --- SETUP ---
output_dir = r'D:\EKG\Skripsi Willy'
os.makedirs(output_dir, exist_ok=True)

try:
    dataset = pd.read_csv(r"D:\EKG\Ekstraksi Fitur 12 Lead\Anak WJ\Test rekam 10 duduk (Jeffry)/Data_3_raw.csv", names=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"], sep=',', skiprows=1)
except FileNotFoundError as e:
    print(f"Error: File tidak ditemukan di path yang diberikan.\n{e}")
    dataset = pd.DataFrame(np.random.randn(5000, 12), columns=["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"])
    print("Menggunakan data dummy untuk demonstrasi.")

leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
snr_results = []
fft_data_storage = {}

# --- PEMROSESAN UTAMA ---
for lead in leads:
    print(f"Memproses Lead {lead}...")
    # ecgmv = (dataset[lead].astype(float)) * (2.42 / ((2**24))) * 1000
    ecg_adc = dataset[lead].values.astype(float)
    ecg_adc, mask_sat = desaturate_edges(ecg_adc)        # <<< ADD: hapus sampel saturasi
    ecg_adc, mask_spk = despike_hampel(ecg_adc, k=7, nsigma=6.0)  # <<< ADD: hapus spike tajam
    ecgmv = to_mV(ecg_adc)                                # hasil dalam mV
    Fs = int(len(dataset) / 10)
    if Fs < 81: Fs = 500

    # --- Tahap 0: Sinyal Mentah (setelah Detrend) ---
    s0_raw_detrended = scipy.signal.detrend(ecgmv.values, type='linear')
    snr_before = estimate_snr(s0_raw_detrended, Fs)

    # --- Tahap 1: IEC Filter ---
    s1_iec_filtered = apply_iec_diagnostic_filter(ecgmv.values, Fs)

    # --- Tahap 2: Detrend (setelah IEC) ---
    s2_detrended_post_iec = scipy.signal.detrend(s1_iec_filtered, type='linear')

    # --- Tahap 3: Butterworth Filter ---
    b_butter, a_butter = scipy.signal.butter(4, 0.6, 'low')

    s3_butterworth = scipy.signal.filtfilt(b_butter, a_butter, s2_detrended_post_iec)
    # --- Tahap 4: FIRFilter ---
    nyq_rate = Fs / 2.0
    width = 5.0 / nyq_rate
    ripple_db = 60.0
    O, beta = scipy.signal.kaiserord(ripple_db, width)
    if O % 2 == 0: O += 1
    taps = scipy.signal.firwin(O, 0.05 / nyq_rate, window=('kaiser', beta), pass_zero=False)
    s4_final_fir = scipy.signal.lfilter(taps, 1.0, s3_butterworth)
    #  kompensasi delay FIR (kalau tetap ingin lfilter):
    n_delay = (len(taps) - 1) // 2
    s4_final_fir = np.roll(s4_final_fir, -n_delay)
    if n_delay > 0:
        s4_final_fir[-n_delay:] = s4_final_fir[-n_delay-1]  # hold last value
    # (alternatif paling bersih: y_filt = scipy.signal.filtfilt(taps, [1.0], tempf_butter))
    
    snr_after = estimate_snr(s4_final_fir, Fs)
    snr_results.append({
        'Lead': lead,
        'SNR Sebelum Filter (dB)': snr_before,
        'SNR Sesudah Filter (dB)': snr_after
    })

    # --- Hitung FFT untuk semua tahapan ---
    fft_data_storage[lead] = {
        's0': calculate_fft(s0_raw_detrended, Fs),
        's1': calculate_fft(s1_iec_filtered, Fs),
        's2': calculate_fft(s2_detrended_post_iec, Fs),
        's3': calculate_fft(s3_butterworth, Fs),
        's4': calculate_fft(s4_final_fir, Fs)
    }

# --- PENYIMPANAN DATA KE EXCEL ---
output_fft_excel = os.path.join(output_dir, 'hasil_fft_lengkap.xlsx')
with pd.ExcelWriter(output_fft_excel, engine='openpyxl') as writer:
    for lead in leads:
        data = fft_data_storage[lead]
        fft_df = pd.DataFrame({
            'Frekuensi (Hz)': data['s0'][0],
            'Amplitudo_Raw_Detrended': data['s0'][1],
            'Amplitudo_IEC_Filtered': data['s1'][1],
            'Amplitudo_Post_IEC_Detrend': data['s2'][1],
            'Amplitudo_Butterworth': data['s3'][1],
            'Amplitudo_Final_FIR': data['s4'][1]
        })
        fft_df.to_excel(writer, sheet_name=f'Lead_{lead}', index=False, float_format='%.5f')
print(f"\nData spektral lengkap telah disimpan ke: {output_fft_excel}")

snr_df = pd.DataFrame(snr_results)
output_snr_excel = os.path.join(output_dir, 'hasil_snr_konsisten_detrended.xlsx')
snr_df.to_excel(output_snr_excel, index=False, float_format='%.2f', engine='openpyxl')
print(f"Perbandingan SNR telah disimpan ke: {output_snr_excel}")

# --- PLOTTING GRAFIK SECARA DETAIL PER LEAD ---
for lead in leads:
    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True, constrained_layout=True)
    fig.suptitle(f'Analisis Spektral Bertahap - Lead {lead}', fontsize=16)
    data = fft_data_storage[lead]
    
    # Dapatkan Fs lagi untuk perhitungan cutoff di plot
    Fs = int(len(dataset) / 10)
    if Fs < 81: Fs = 500

    # Plot 1: Raw vs IEC
    axes[0].plot(data['s0'][0], data['s0'][1], label='S0: Raw (Detrended)', color='blue', alpha=0.7)
    axes[0].plot(data['s1'][0], data['s1'][1], label='S1: Setelah IEC Filter', color='red', linewidth=1.5)
    axes[0].set_title('Tahap 1: Efek IEC Filter (HP, LP, Notch)')
    axes[0].axvline(x=0.05, color='g', linestyle='--', label='HP Cutoff (0.05 Hz)')
    axes[0].axvline(x=150, color='g', linestyle='--') # Tanpa label agar tidak menumpuk
    axes[0].axvline(x=50, color='m', linestyle=':', label='Notch (50 Hz)')
    
    # Plot 2: IEC vs Detrend
    axes[1].plot(data['s1'][0], data['s1'][1], label='S1: Sebelum Detrend', color='blue', alpha=0.7)
    axes[1].plot(data['s2'][0], data['s2'][1], label='S2: Setelah Detrend', color='red', linewidth=1.5)
    axes[1].set_title('Tahap 2: Efek Detrending (setelah IEC)')

    # Plot 3: Detrend vs Butterworth
    axes[2].plot(data['s2'][0], data['s2'][1], label='S2: Sebelum Butterworth', color='blue', alpha=0.7)
    axes[2].plot(data['s3'][0], data['s3'][1], label='S3: Setelah Butterworth', color='red', linewidth=1.5)
    axes[2].set_title('Tahap 3: Efek Butterworth Low-pass Filter')
    butterworth_cutoff_hz = 0.6 * (Fs / 2) # Cutoff 0.6 dari frekuensi Nyquist
    axes[2].axvline(x=butterworth_cutoff_hz, color='g', linestyle='--', label=f'Cutoff ({butterworth_cutoff_hz:.1f} Hz)')

    # Plot 4: Butterworth vs FIR
    axes[3].plot(data['s3'][0], data['s3'][1], label='S3: Sebelum FIR', color='blue', alpha=0.7)
    axes[3].plot(data['s4'][0], data['s4'][1], label='S4: Setelah FIR (Final)', color='red', linewidth=1.5)
    axes[3].set_title('Tahap 4: Efek FIR Filter')
    fir_cutoff_hz = 0.05
    axes[3].axvline(x=fir_cutoff_hz, color='g', linestyle='--', label=f'Cutoff ({fir_cutoff_hz} Hz)')
    axes[3].set_xlabel('Frekuensi (Hz)')

    for ax in axes:
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--")
        ax.set_ylabel('Amplitudo (Log)')
        ax.legend()
        ax.set_xlim(0, Fs / 2)

plt.show()

print("\n--- Selesai ---")
print("Perbandingan SNR (Metode Konsisten, Sinyal Mentah di-Detrend):")
print(snr_df.to_string(index=False))
