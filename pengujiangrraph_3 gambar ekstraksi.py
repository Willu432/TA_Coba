import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.signal
import neurokit2 as nk
import math

# IEC 60601-2-25:2011 compliant diagnostic ECG filter
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

file_path = r"D:\EKG\Ekstraksi Fitur 12 Lead\Anak WJ\Test rekam 10 duduk (Jeffry)\Data_3_raw.csv"

try:
    dataset = pd.read_csv(file_path, names=["I", "II", "III", "AVR", "AVL", "AVF", 
                                            "V1", "V2", "V3", "V4", "V5", "V6"], 
                           sep=',', skiprows=1, dtype=float)
    t = np.arange(0, len(dataset))

    # Function to disable scientific notation and show full integers
    def disable_sci_not(axs):
        for ax in axs:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='y')  # no scientific notation
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))  # no decimals

    # --- Window 1: Limb Leads ---
    fig1, axs1 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig1.suptitle('Limb Leads (I, II, III)', fontsize=24)
    axs1[0].plot(t, dataset['I'], label='I', color='blue'); axs1[0].set_ylabel('I ADC Value'); axs1[0].legend(); axs1[0].grid(True)
    axs1[1].plot(t, dataset['II'], label='II', color='green'); axs1[1].set_ylabel('II ADC Value'); axs1[1].legend(); axs1[1].grid(True)
    axs1[2].plot(t, dataset['III'], label='III', color='red'); axs1[2].set_ylabel('III ADC Value'); axs1[2].set_xlabel('Sample'); axs1[2].legend(); axs1[2].grid(True)
    disable_sci_not(axs1)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Window 2: Augmented Limb Leads ---
    fig2, axs2 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig2.suptitle('Augmented Limb Leads (AVR, AVL, AVF)', fontsize=24)
    axs2[0].plot(t, dataset['AVR'], label='AVR', color='blue'); axs2[0].set_ylabel('AVR ADC Value'); axs2[0].legend(); axs2[0].grid(True)
    axs2[1].plot(t, dataset['AVL'], label='AVL', color='green'); axs2[1].set_ylabel('AVL ADC Value'); axs2[1].legend(); axs2[1].grid(True)
    axs2[2].plot(t, dataset['AVF'], label='AVF', color='red'); axs2[2].set_ylabel('AVF ADC Value'); axs2[2].set_xlabel('Sample'); axs2[2].legend(); axs2[2].grid(True)
    disable_sci_not(axs2)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Window 3: Precordial Leads (V1-V3) ---
    fig3, axs3 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig3.suptitle('Precordial Leads (V1, V2, V3)', fontsize=24)
    axs3[0].plot(t, dataset['V1'], label='V1', color='blue'); axs3[0].set_ylabel('V1 ADC Value'); axs3[0].legend(); axs3[0].grid(True)
    axs3[1].plot(t, dataset['V2'], label='V2', color='green'); axs3[1].set_ylabel('V2 ADC Value'); axs3[1].legend(); axs3[1].grid(True)
    axs3[2].plot(t, dataset['V3'], label='V3', color='red'); axs3[2].set_ylabel('V3 ADC Value'); axs3[2].set_xlabel('Sample'); axs3[2].legend(); axs3[2].grid(True)
    disable_sci_not(axs3)
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Window 4: Precordial Leads (V4-V6) ---
    fig4, axs4 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig4.suptitle('Precordial Leads (V4, V5, V6)', fontsize=24)
    axs4[0].plot(t, dataset['V4'], label='V4', color='blue'); axs4[0].set_ylabel('V4 ADC Value'); axs4[0].legend(); axs4[0].grid(True)
    axs4[1].plot(t, dataset['V5'], label='V5', color='green'); axs4[1].set_ylabel('V5 ADC Value'); axs4[1].legend(); axs4[1].grid(True)
    axs4[2].plot(t, dataset['V6'], label='V6', color='red'); axs4[2].set_ylabel('V6 ADC Value'); axs4[2].set_xlabel('Sample'); axs4[2].legend(); axs4[2].grid(True)
    disable_sci_not(axs4)
    fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
  
 # ========================================================================================================================================================
  # ========================================================================================================================================================
   # ========================================================================================================================================================   
try:
    dataset = pd.read_csv(file_path, names=["I", "II", "III", "AVR", "AVL", "AVF", 
                                            "V1", "V2", "V3", "V4", "V5", "V6"], 
                           sep=',', skiprows=1, dtype=float)
    t = np.arange(0, len(dataset))
    
    # Calculate sampling frequency
    Fs = int(len(dataset) / 10)  # Define Fs as the sampling frequency
    print(f"Sampling frequency: {Fs} Hz")
    
    # Define leads groups
    limb_leads = ['I', 'II', 'III']
    augmented_leads = ['AVR', 'AVL', 'AVF']
    precordial_v1_v3 = ['V1', 'V2', 'V3']
    precordial_v4_v6 = ['V4', 'V5', 'V6']
    
    all_leads = limb_leads + augmented_leads + precordial_v1_v3 + precordial_v4_v6

    # Function to disable scientific notation and show full integers
    def disable_sci_not(axs):
        for ax in axs:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='y')  # no scientific notation
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))  # no decimals

    # Function to process and detect PQRST for a single lead
    def process_lead_with_pqrst(lead_name, lead_data, fs):
        """Process a single lead with IEC filter and PQRST detection"""
        
        # Convert to mV
        ecgmv = (lead_data) * (2.42 / ((2**24))) * 1000
        
        # Apply IEC 60601-2-25 Diagnostic Filter
        iec_filtered_signal = apply_iec_diagnostic_filter(ecgmv, fs)
        
        # Baseline Correction
        detr_ecg = scipy.signal.detrend(iec_filtered_signal, axis=-1, type='linear', bp=0, overwrite_data=False)
        
        # Butterworth filter
        b, a = scipy.signal.butter(4, 0.6, 'low')
        tempf_butter = scipy.signal.filtfilt(b, a, detr_ecg)
        
        # FIR filter
        Fsf = int(len(dataset) / 10)
        cutoff_hz = 0.05
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
        
        # PQRST Peak Detection
        try:
            _, rpeaks = nk.ecg_peaks(y_filt, sampling_rate=fs)
            signal_dwt, waves_dwt = nk.ecg_delineate(y_filt, rpeaks, sampling_rate=fs, method="dwt")
            
            # Clean peaks (remove NaN values)
            clean_peaks = {}
            peak_types = ['ECG_R_Peaks', 'ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']
            
            # Clean R peaks
            clean_peaks['ECG_R_Peaks'] = np.array([x for x in rpeaks['ECG_R_Peaks'] if not math.isnan(x)]).astype(int)
            
            # Clean other peaks
            for peak_type in peak_types[1:]:  # Skip R peaks as already processed
                if peak_type in waves_dwt and waves_dwt[peak_type] is not None:
                    clean_peaks[peak_type] = np.array([x for x in waves_dwt[peak_type] if not math.isnan(x)]).astype(int)
                else:
                    clean_peaks[peak_type] = np.array([])
            
            # =================================================================
            # CORRECTIONS CYCLE (sama seperti kode original)
            # =================================================================
            
            # Basic corrections dengan pengecekan array length
            if len(clean_peaks['ECG_R_Peaks']) > 0 and 'ECG_P_Onsets' in waves_dwt and waves_dwt['ECG_P_Onsets'] is not None:
                p_onsets = np.array([x for x in waves_dwt['ECG_P_Onsets'] if not math.isnan(x)]).astype(int)
                if len(p_onsets) > 0 and clean_peaks['ECG_R_Peaks'][0] < p_onsets[0]:
                    clean_peaks['ECG_R_Peaks'] = np.delete(clean_peaks['ECG_R_Peaks'], 0)

            if len(clean_peaks['ECG_P_Peaks']) > 0 and 'ECG_P_Onsets' in waves_dwt and waves_dwt['ECG_P_Onsets'] is not None:
                p_onsets = np.array([x for x in waves_dwt['ECG_P_Onsets'] if not math.isnan(x)]).astype(int)
                if len(p_onsets) > 0 and clean_peaks['ECG_P_Peaks'][0] < p_onsets[0]:
                    clean_peaks['ECG_P_Peaks'] = np.delete(clean_peaks['ECG_P_Peaks'], 0)

            if len(clean_peaks['ECG_Q_Peaks']) > 0 and 'ECG_P_Onsets' in waves_dwt and waves_dwt['ECG_P_Onsets'] is not None:
                p_onsets = np.array([x for x in waves_dwt['ECG_P_Onsets'] if not math.isnan(x)]).astype(int)
                if len(p_onsets) > 0 and clean_peaks['ECG_Q_Peaks'][0] < p_onsets[0]:
                    clean_peaks['ECG_Q_Peaks'] = np.delete(clean_peaks['ECG_Q_Peaks'], 0)

            if len(clean_peaks['ECG_S_Peaks']) > 0 and 'ECG_P_Onsets' in waves_dwt and waves_dwt['ECG_P_Onsets'] is not None:
                p_onsets = np.array([x for x in waves_dwt['ECG_P_Onsets'] if not math.isnan(x)]).astype(int)
                if len(p_onsets) > 0 and clean_peaks['ECG_S_Peaks'][0] < p_onsets[0]:
                    clean_peaks['ECG_S_Peaks'] = np.delete(clean_peaks['ECG_S_Peaks'], 0)

            if len(clean_peaks['ECG_T_Peaks']) > 0 and 'ECG_P_Onsets' in waves_dwt and waves_dwt['ECG_P_Onsets'] is not None:
                p_onsets = np.array([x for x in waves_dwt['ECG_P_Onsets'] if not math.isnan(x)]).astype(int)
                if len(p_onsets) > 0 and clean_peaks['ECG_T_Peaks'][0] < p_onsets[0]:
                    clean_peaks['ECG_T_Peaks'] = np.delete(clean_peaks['ECG_T_Peaks'], 0)

            # Additional corrections like original
            if len(clean_peaks['ECG_R_Peaks']) > 1:
                if y_filt[clean_peaks['ECG_R_Peaks']][0] < y_filt[clean_peaks['ECG_R_Peaks']][1]/2:
                    clean_peaks['ECG_R_Peaks'] = np.delete(clean_peaks['ECG_R_Peaks'], 0)
                    if len(clean_peaks['ECG_P_Peaks']) > 0:
                        clean_peaks['ECG_P_Peaks'] = np.delete(clean_peaks['ECG_P_Peaks'], 0)
                    if len(clean_peaks['ECG_Q_Peaks']) > 0:
                        clean_peaks['ECG_Q_Peaks'] = np.delete(clean_peaks['ECG_Q_Peaks'], 0)
                    if len(clean_peaks['ECG_S_Peaks']) > 0:
                        clean_peaks['ECG_S_Peaks'] = np.delete(clean_peaks['ECG_S_Peaks'], 0)
                    if len(clean_peaks['ECG_T_Peaks']) > 0:
                        clean_peaks['ECG_T_Peaks'] = np.delete(clean_peaks['ECG_T_Peaks'], 0)
            
            return y_filt, clean_peaks, True
        
        except Exception as e:
            print(f"PQRST detection failed for {lead_name}: {e}")
            return y_filt, {}, False

    # Function to plot leads with PQRST detection
    def plot_leads_with_pqrst(leads_group, group_name, fig_num):
        """Plot a group of leads with PQRST detection"""
        
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f'{group_name} - Filtered Signal with PQRST Detection', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'green', 'red']
        
        for i, lead in enumerate(leads_group):
            # Process lead with PQRST detection
            filtered_signal, peaks, detection_success = process_lead_with_pqrst(lead, dataset[lead], Fs)
            
            # Plot filtered signal
            axs[i].plot(t, filtered_signal, color=colors[i], linewidth=1, label=f'{lead} (Filtered)')
            axs[i].set_ylabel(f'{lead} [mV]', fontweight='bold')
            axs[i].grid(True, alpha=0.3)
            axs[i].legend(loc='upper right')
            
            # Plot PQRST peaks if detection was successful
            if detection_success and peaks:
                # Plot each type of peak (same as original code)
                if 'ECG_P_Peaks' in peaks and len(peaks['ECG_P_Peaks']) > 0:
                    valid_p = peaks['ECG_P_Peaks'][peaks['ECG_P_Peaks'] < len(filtered_signal)]
                    if len(valid_p) > 0:
                        axs[i].plot(valid_p, filtered_signal[valid_p], "x", color='blue', label="P Peak")

                if 'ECG_Q_Peaks' in peaks and len(peaks['ECG_Q_Peaks']) > 0:
                    valid_q = peaks['ECG_Q_Peaks'][peaks['ECG_Q_Peaks'] < len(filtered_signal)]
                    if len(valid_q) > 0:
                        axs[i].plot(valid_q, filtered_signal[valid_q], "x", color='green', label="Q Peak")

                if 'ECG_R_Peaks' in peaks and len(peaks['ECG_R_Peaks']) > 0:
                    valid_r = peaks['ECG_R_Peaks'][peaks['ECG_R_Peaks'] < len(filtered_signal)]
                    if len(valid_r) > 0:
                        axs[i].plot(valid_r, filtered_signal[valid_r], "x", color='black', label="R Peak")

                if 'ECG_S_Peaks' in peaks and len(peaks['ECG_S_Peaks']) > 0:
                    valid_s = peaks['ECG_S_Peaks'][peaks['ECG_S_Peaks'] < len(filtered_signal)]
                    if len(valid_s) > 0:
                        axs[i].plot(valid_s, filtered_signal[valid_s], "x", color='red', label="S Peak")

                if 'ECG_T_Peaks' in peaks and len(peaks['ECG_T_Peaks']) > 0:
                    valid_t = peaks['ECG_T_Peaks'][peaks['ECG_T_Peaks'] < len(filtered_signal)]
                if len(valid_t) > 0:
                    axs[i].plot(valid_t, filtered_signal[valid_t], "x", color='purple', label="T Peak")
                
                # Add simple annotations like original code (limit to first few peaks)
                max_annotations = 5
                try:
                    # Annotate P peaks
                    if 'ECG_P_Peaks' in peaks and len(peaks['ECG_P_Peaks']) > 0:
                        valid_p = peaks['ECG_P_Peaks'][peaks['ECG_P_Peaks'] < len(filtered_signal)]
                        for idx in valid_p[:max_annotations]:
                            axs[i].annotate('P', xy=(idx, filtered_signal[idx]))

                    # Annotate Q peaks
                    if 'ECG_Q_Peaks' in peaks and len(peaks['ECG_Q_Peaks']) > 0:
                        valid_q = peaks['ECG_Q_Peaks'][peaks['ECG_Q_Peaks'] < len(filtered_signal)]
                        for idx in valid_q[:max_annotations]:
                            axs[i].annotate('Q', xy=(idx, filtered_signal[idx]))

                    # Annotate R peaks
                    if 'ECG_R_Peaks' in peaks and len(peaks['ECG_R_Peaks']) > 0:
                        valid_r = peaks['ECG_R_Peaks'][peaks['ECG_R_Peaks'] < len(filtered_signal)]
                        for idx in valid_r[:max_annotations]:
                            axs[i].annotate('R', xy=(idx, filtered_signal[idx]))

                    # Annotate S peaks
                    if 'ECG_S_Peaks' in peaks and len(peaks['ECG_S_Peaks']) > 0:
                        valid_s = peaks['ECG_S_Peaks'][peaks['ECG_S_Peaks'] < len(filtered_signal)]
                        for idx in valid_s[:max_annotations]:
                            axs[i].annotate('S', xy=(idx, filtered_signal[idx]))

                    # Annotate T peaks
                    if 'ECG_T_Peaks' in peaks and len(peaks['ECG_T_Peaks']) > 0:
                        valid_t = peaks['ECG_T_Peaks'][peaks['ECG_T_Peaks'] < len(filtered_signal)]
                        for idx in valid_t[:max_annotations]:
                            axs[i].annotate('T', xy=(idx, filtered_signal[idx]))
                            
                except Exception as e:
                    print(f"Annotation error for {lead}: {e}")
                
                # Update legend to include peaks
                axs[i].legend(loc="lower left")
            
            else:
                axs[i].text(0.02, 0.98, 'PQRST Detection Failed', 
                           transform=axs[i].transAxes, fontsize=10, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        axs[2].set_xlabel('Sample', fontweight='bold')
        disable_sci_not(axs)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig

    # Plot all eight groups: 4 Raw + 4 Filtered with PQRST detection
    print("Processing and plotting leads...")
    print("Generating 8 figures total: 4 Raw Data + 4 Filtered with PQRST Detection")
    
    # --- RAW DATA FIGURES (4 windows) ---
    print("\nGenerating Raw Data Plots...")
    
    # # --- Figure 1: Raw Limb Leads ---
    # fig1_raw = plot_raw_leads(limb_leads, 'Limb Leads (I, II, III)', 1)
    
    # # --- Figure 2: Raw Augmented Limb Leads ---
    # fig2_raw = plot_raw_leads(augmented_leads, 'Augmented Limb Leads (AVR, AVL, AVF)', 2)
    
    # # --- Figure 3: Raw Precordial Leads (V1-V3) ---
    # fig3_raw = plot_raw_leads(precordial_v1_v3, 'Precordial Leads (V1, V2, V3)', 3)
    
    # # --- Figure 4: Raw Precordial Leads (V4-V6) ---
    # fig4_raw = plot_raw_leads(precordial_v4_v6, 'Precordial Leads (V4, V5, V6)', 4)
    
    # --- FILTERED DATA WITH PQRST FIGURES (4 windows) ---
    print("Generating Filtered Data with PQRST Detection Plots...")
    
    # --- Figure 5: Filtered Limb Leads with PQRST ---
    fig5_filtered = plot_leads_with_pqrst(limb_leads, 'Limb Leads (I, II, III)', 5)
    
    # --- Figure 6: Filtered Augmented Limb Leads with PQRST ---
    fig6_filtered = plot_leads_with_pqrst(augmented_leads, 'Augmented Limb Leads (AVR, AVL, AVF)', 6)
    
    # --- Figure 7: Filtered Precordial Leads (V1-V3) with PQRST ---
    fig7_filtered = plot_leads_with_pqrst(precordial_v1_v3, 'Precordial Leads (V1, V2, V3)', 7)
    
    # --- Figure 8: Filtered Precordial Leads (V4-V6) with PQRST ---
    fig8_filtered = plot_leads_with_pqrst(precordial_v4_v6, 'Precordial Leads (V4, V5, V6)', 8)
    
    print("\nAll 8 figures generated successfully!")
    print("\n" + "="*80)
    print("SUMMARY OF GENERATED FIGURES:")
    print("="*80)
    print("RAW DATA FIGURES (ADC Values):")
    print("  - Figure 1: Raw Limb Leads (I, II, III)")
    print("  - Figure 2: Raw Augmented Limb Leads (AVR, AVL, AVF)")
    print("  - Figure 3: Raw Precordial Leads (V1, V2, V3)")  
    print("  - Figure 4: Raw Precordial Leads (V4, V5, V6)")
    print("\nFILTERED DATA FIGURES (with PQRST Detection):")
    print("  - Figure 5: Filtered Limb Leads (I, II, III)")
    print("  - Figure 6: Filtered Augmented Limb Leads (AVR, AVL, AVF)")
    print("  - Figure 7: Filtered Precordial Leads (V1, V2, V3)")
    print("  - Figure 8: Filtered Precordial Leads (V4, V5, V6)")
    print("\n" + "="*80)
    print("PQRST Peak Detection Legend (Filtered Figures Only):")
    print("  - P Peak: Blue X marks")
    print("  - Q Peak: Green X marks")  
    print("  - R Peak: Black X marks")
    print("  - S Peak: Red X marks")
    print("  - T Peak: Purple X marks")
    print("\nFilter Applied: IEC 60601-2-25:2011 Diagnostic ECG Filter")
    print("  - High-pass filter: 0.05 Hz")
    print("  - Low-pass filter: 150 Hz") 
    print("  - Notch filter: 50 Hz")
    print("  - Additional Butterworth and FIR filtering")
    print("  - Peak correction cycle applied")
    print("="*80)
    
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")