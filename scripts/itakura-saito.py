import numpy as np
import librosa
from scipy.signal import freqz, lfilter
from scipy.signal.windows import hamming
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def load_audio(filename, sr=None):
    """ Load audio file """
    signal, sr = librosa.load(filename, sr=sr)
    return signal, sr

def estimate_ar_parameters(signal, order=16):
    """ Estimate AR parameters using LPC from librosa and calculate residual """
    window = hamming(len(signal))
    signal_windowed = signal * window
    ar_coeffs = librosa.lpc(signal_windowed, order=order)
    # Manually calculate the LPC residual
    pred_signal = lfilter(ar_coeffs, [1], signal_windowed)  # LPC prediction
    error = signal_windowed - pred_signal  # error as actual minus prediction
    var = np.var(error)
    return ar_coeffs, var

def itakura_saito_distance(ar1, var1, ar2, var2):
    """ Compute Itakura-Saito distance between two AR models """
    ar1 = np.concatenate(([1], -ar1[1:]))  # include 1 for the AR part
    ar2 = np.concatenate(([1], -ar2[1:]))
    _, h1 = freqz(np.sqrt(var1), ar1, worN=512)
    _, h2 = freqz(np.sqrt(var2), ar2, worN=512)
    ratio = (abs(h1) ** 2) / (abs(h2) ** 2)
    dist = np.mean(ratio - np.log(ratio) - 1)
    return dist

def align_signals(signal1, signal2):
    """ Align two signals using FastDTW """
    distance, path = fastdtw(signal1.reshape(-1,1), signal2.reshape(-1,1), dist=euclidean)
    return distance, path

# Example usage: Load your actual speech data from wav files
signal2, sr2 = load_audio(r"C:\laryngectomy\dataset\data\Test\Normal\sen_1_normal_total_Normal_04.wav")
signal1, sr1 = load_audio(r"C:\laryngectomy\results\converted_wav\mmse-chunk-exp-1\sen_1_normal_total_Normal_04.wav")

# Estimate AR parameters
ar1, var1 = estimate_ar_parameters(signal1, order=16)
ar2, var2 = estimate_ar_parameters(signal2, order=16)

# Calculate Itakura-Saito distance
distance_is = itakura_saito_distance(ar1, var1, ar2, var2)
print(f'Itakura-Saito Distance: {distance_is}')

# # Align signals using FastDTW
# distance_dtw, path_dtw = align_signals(signal1, signal2)
# print(f'FastDTW Distance: {distance_dtw}')
