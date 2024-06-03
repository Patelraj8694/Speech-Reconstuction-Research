import numpy as np
import soundfile as sf
from pystoi import stoi
from pesq import pesq
from pymcd.mcd import Calculate_MCD

def match_length(signal1, signal2):
    """Ensure that two signals have the same length by trimming or padding the longer one."""
    len1, len2 = len(signal1), len(signal2)
    if len1 > len2:
        signal1 = signal1[:len2]
    elif len2 > len1:
        signal2 = signal2[:len1]
    return signal1, signal2

def calculate_snr(reference, degraded):
    """Calculate the normal Signal-to-Noise Ratio (SNR)."""
    snr = 10 * np.log10(np.sum(reference ** 2) / np.sum((reference - degraded) ** 2))
    return snr

def calculate_segsnr(reference, degraded, segment_length=160):
    """Calculate the Segmental Signal-to-Noise Ratio (SegSNR)."""
    segsnr = []
    for i in range(0, len(reference) - segment_length + 1, segment_length):
        ref_segment = reference[i:i + segment_length]
        deg_segment = degraded[i:i + segment_length]
        segment_snr = 10 * np.log10(np.sum(ref_segment ** 2) / np.sum((ref_segment - deg_segment) ** 2))
        segsnr.append(segment_snr)
    return np.mean(segsnr)

def objective_evaluation(raw_reference_speech, raw_reconstructed_speech,reference_speech, reconstructed_speech, fs):
    # Ensure same length for evaluation
    ref_speech, recon_speech = match_length(reference_speech, reconstructed_speech)
    
    # MCD
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw_sl")
    mcd_value = mcd_toolbox.calculate_mcd(raw_reference_speech, raw_reconstructed_speech)
    
    # STOI
    stoi_value = stoi(ref_speech, recon_speech, fs, extended=False)

    # PESQ
    pesq_value = pesq(fs, reference_speech, reconstructed_speech, 'wb')

    # SNR and SegSNR
    snr_value = calculate_snr(ref_speech, recon_speech)
    segsnr_value = calculate_segsnr(ref_speech, recon_speech)

    return {
        "MCD": mcd_value,
        "STOI": stoi_value,
        "PESQ": pesq_value,
        "SNR": snr_value,
        "SegSNR": segsnr_value
    }

# Load and process the data
raw_reference_speech = r"C:\laryngectomy\results\converted_wav\mmse-chunk-exp-1\sen_1_normal_total_Normal_04.wav"
raw_reconstructed_speech = r"C:\laryngectomy\dataset\data\Test\Normal\sen_1_normal_total_Normal_04.wav"
reference_speech, sr = sf.read(raw_reference_speech)
reconstructed_speech, sr = sf.read(raw_reconstructed_speech)

results = objective_evaluation(raw_reference_speech, raw_reconstructed_speech, reference_speech, reconstructed_speech, sr)
print(results)
