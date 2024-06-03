import numpy as np
import soundfile as sf
from pystoi import stoi
from pesq import pesq
from pymcd.mcd import Calculate_MCD
import pyworld as pw
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def match_length(signal1, signal2):
    """Ensure that two signals have the same length by trimming or padding the longer one."""
    len1, len2 = len(signal1), len(signal2)
    if len1 > len2:
        signal1 = signal1[:len2]
    elif len2 > len1:
        signal2 = signal2[:len1]
    return signal1, signal2

# def calculate_rmse_voiced_aligned(reference, degraded, fs):
#     """Calculate RMSE for the voiced parts of speech signals using FastDTW for alignment."""
#     # Frame period typically used in pyworld
#     frame_period = 5.0  # in milliseconds
    
#     # Analyze both signals to extract fundamental frequency (F0) and voiced regions
#     f0_ref, _ = pw.dio(reference, fs, frame_period=frame_period)
#     f0_deg, _ = pw.dio(degraded, fs, frame_period=frame_period)
    
#     # Filter out unvoiced parts where f0 is 0
#     voiced_ref = reference[f0_ref > 0]
#     voiced_deg = degraded[f0_deg > 0]

#     # Align signals using FastDTW
#     distance, path = fastdtw(voiced_ref, voiced_deg, dist=euclidean)
#     aligned_ref = np.array([voiced_ref[i] for i, j in path])
#     aligned_deg = np.array([voiced_deg[j] for i, j in path])

#     # Calculate RMSE
#     rmse = np.sqrt(np.mean((aligned_ref - aligned_deg) ** 2))
#     return rmse


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

def objective_evaluation(reference_speech, reconstructed_speech, fs):
    # Ensure same length for evaluation
    ref_speech, recon_speech = match_length(reference_speech, reconstructed_speech)
    
    # # MCD
    # mcd_toolbox = Calculate_MCD(MCD_mode="dtw_sl")
    # mcd_value = mcd_toolbox.calculate_mcd(raw_reference_speech, raw_reconstructed_speech)
    
    # STOI
    stoi_value = stoi(ref_speech, recon_speech, fs, extended=False)

    # PESQ
    pesq_value = pesq(fs, reference_speech, reconstructed_speech, 'wb')

    # SNR and SegSNR
    snr_value = calculate_snr(ref_speech, recon_speech)
    segsnr_value = calculate_segsnr(ref_speech, recon_speech)

    # RMSE for voiced segments
    # rmse_voiced = calculate_rmse_voiced_aligned(reference_speech, reconstructed_speech, fs)

    return {
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

results = objective_evaluation(reference_speech, reconstructed_speech, sr)
print(results)
