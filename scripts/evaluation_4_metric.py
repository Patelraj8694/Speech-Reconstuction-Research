import numpy as np
import soundfile as sf
import librosa
from pystoi import stoi
from pesq import pesq
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def process_audio(file_path1, file_path2, sr):
    """
    Process two audio files to compute various objective metrics after time-aligning them.

    :param file_path1: Path to the first audio file.
    :param file_path2: Path to the second audio file.
    :param sr: Sampling rate to which both audio files should be resampled.
    :return: Dictionary of computed metrics.
    """
    # Read audio files
    data1, sr1 = sf.read(file_path1)
    data2, sr2 = sf.read(file_path2)

    # Ensure the data is mono
    if data1.ndim > 1:
        data1 = np.mean(data1, axis=1)  # Convert to mono by averaging the channels
    if data2.ndim > 1:
        data2 = np.mean(data2, axis=1)

    # Resample if necessary
    if sr1 != sr:
        data1 = librosa.resample(data1, sr1, sr)
    if sr2 != sr:
        data2 = librosa.resample(data2, sr2, sr)

    # Perform DTW using fastdtw
    distance, path = fastdtw(data1.reshape(-1,1), data2.reshape(-1,1), dist=euclidean)
    aligned_data1 = np.array([data1[idx] for idx, _ in path])
    aligned_data2 = np.array([data2[idx] for _, idx in path])

    # # Compute MFCCs
    # mfcc1 = librosa.feature.mfcc(y=data1, sr=sr, n_mfcc=40)
    # mfcc2 = librosa.feature.mfcc(y=data2, sr=sr, n_mfcc=40)

    # distance_2, path_2 = fastdtw(mfcc1.T, mfcc2.T, dist=euclidean)
    # aligned_mfcc1 = np.array([mfcc1.T[idx] for idx, _ in path_2])
    # aligned_mfcc2 = np.array([mfcc2.T[idx] for _, idx in path_2])

    # # Compute MCD
    # mcd_const = (10 / np.log(10)) * np.sqrt(2)  # conversion factor to dB
    # diff_squared = (aligned_mfcc1[1:, :] - aligned_mfcc2[1:, :])**2  # exclude the zeroth MFCC coefficient
    # mcd = mcd_const * np.mean(np.sqrt(np.sum(diff_squared, axis=0)))  # sum over coefficients, mean over frames

    # Compute STOI
    d_stoi = stoi(aligned_data1, aligned_data2, sr, extended=False)

    # Compute PESQ
    d_pesq = pesq(sr, aligned_data1, aligned_data2, 'wb')

    # Compute SNR
    noise = aligned_data1 - aligned_data2
    snr = 10 * np.log10(np.sum(aligned_data1 ** 2) / np.sum(noise ** 2))

    # Compute Segmental SNR
    seg_len = int(sr * 0.200)  # 20 ms segments
    seg_snr = []
    for i in range(0, len(aligned_data1) - seg_len, seg_len):
        seg_signal = aligned_data1[i:i+seg_len]
        seg_noise = noise[i:i+seg_len]
        seg_snr.append(10 * np.log10(np.sum(seg_signal ** 2) / np.sum(seg_noise ** 2)))
    seg_snr = np.mean(seg_snr)

    # Compute RMSE
    # rmse = np.sqrt(np.mean((aligned_data1 - aligned_data2) ** 2))

    return {
        'STOI': d_stoi,
        'PESQ': d_pesq,
        'SNR': snr,
        'Segmental SNR': seg_snr
    }

# Example usage
raw_reference_speech = r"C:\laryngectomy\results\converted_wav\mmse-chunk-exp-1\sen_1_normal_total_Normal_04.wav"
raw_reconstructed_speech = r"C:\laryngectomy\dataset\data\Test\Normal\sen_1_normal_total_Normal_04.wav"
metrics = process_audio(raw_reference_speech, raw_reconstructed_speech, 16000)
print(metrics)
