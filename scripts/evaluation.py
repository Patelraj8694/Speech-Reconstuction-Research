import numpy as np
import soundfile as sf
import librosa
from pystoi import stoi
from pesq import pesq
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from pathlib import Path
from mel_cepstral_distance import get_metrics_wavs
import pysepm

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

    # Compute STOI
    d_stoi = stoi(aligned_data1, aligned_data2, sr, extended=False)

    # Compute PESQ
    d_pesq = pesq(sr, aligned_data1, aligned_data2, 'wb')

    # Compute LLR
    llr = pysepm.llr(aligned_data1, aligned_data2, sr)

    # compute Ceptrum Distance (CD)
    cd = pysepm.cepstrum_distance(aligned_data1, aligned_data2, sr)

    # compute stoi pysepm
    stoi_pysepm = pysepm.stoi(aligned_data1, aligned_data2, sr)

    # compute SSNR pysepm
    ssnr_pysepm = pysepm.SNRseg(aligned_data1, aligned_data2, sr, frameLen=0.2)

    # Compute SNR
    noise = aligned_data1 - aligned_data2
    snr = 10 * np.log10(np.sum(aligned_data1 ** 2) / np.sum(noise ** 2))

    # Compute Segmental SNR
    seg_len = int(sr * 0.200)  # 200 ms segments
    seg_snr = []
    for i in range(0, len(aligned_data1) - seg_len, seg_len):
        seg_signal = aligned_data1[i:i+seg_len]
        seg_noise = noise[i:i+seg_len]
        seg_snr.append(10 * np.log10(np.sum(seg_signal ** 2) / np.sum(seg_noise ** 2)))
    seg_snr = np.mean(seg_snr)

    # Compute MCD
    wav_file_1 = Path(file_path1)
    wav_file_2 = Path(file_path2)
    if not wav_file_1.is_file() or not wav_file_2.is_file():
        raise FileNotFoundError("One of the audio files does not exist.")
    mcd, penalty, frames = get_metrics_wavs(wav_file_1, wav_file_2, n_mfcc=40)

    return {
        'STOI': d_stoi,
        'PESQ': d_pesq,
        'LLR': llr,
        'CD': cd,
        'STOI pysepm': stoi_pysepm,
        'SSNR pysepm': ssnr_pysepm,
        'SNR': snr,
        'Segmental SNR': seg_snr,
        'MCD': mcd,
        'Penalty': penalty,
        'Frames': frames
    }

# Example usage
raw_reference_speech = r"C:\laryngectomy\results\converted_wav\mmse-chunk-exp-1\sen_1_normal_total_Normal_04.wav"
raw_reconstructed_speech = r"C:\laryngectomy\dataset\data\Test\Normal\sen_1_normal_total_Normal_04.wav"
metrics = process_audio(raw_reference_speech, raw_reconstructed_speech, 16000)
print(metrics)
