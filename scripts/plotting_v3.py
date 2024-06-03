import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pyworld
import os

def load_and_extract_features(audio_path, sr=16000):
    """
    Load an audio file and extract waveform, spectrogram, and fundamental frequency.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        y_pw = y.astype(np.float64)
        f0, time_stamps = pyworld.dio(y_pw, sr)
        f0 = pyworld.stonemask(y_pw, f0, time_stamps, sr)
        return y, sr, f0, time_stamps
    except Exception as e:
        print(f"Error loading or processing audio file: {e}")
        return None, None, None, None

def plot_waveform(y, sr, output_path):
    """
    Plot and save waveform using librosa's display feature.
    """
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    librosa.display.waveshow(y, sr=sr, ax=ax, color='b')
    ax.set_title('Waveform')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_spectrogram(y, sr, output_path, n_fft=2048, hop_length=512, fmax=4000):
    """
    Plot and save a spectrogram that only displays up to 4000 Hz.
    """
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_pitch_contour(f0, time_stamps, output_path):
    """
    Plot and save the pitch contour as discrete points, excluding unvoiced regions.
    """
    plt.figure(figsize=(10, 4))
    valid_f0 = f0 > 0
    if np.any(valid_f0):
        plt.scatter(time_stamps[valid_f0], f0[valid_f0], color='red', label='F0 (Pitch)', s=10)
        plt.title('Fundamental Frequency (F0)')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
    else:
        print("No voiced segments detected; no pitch contour to plot.")
    plt.close()

def plot_spectrogram_and_pitch(y, sr, f0, time_stamps, output_path, n_fft=2048, hop_length=512, fmax=4000):
    """
    Plot and save a spectrogram with an overlay of pitch contour up to 4000 Hz.
    """
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
    plt.colorbar(img, ax=ax, format='%+2.0f dB', label='Decibels')
    ax.set_title('Spectrogram and Pitch Contour')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    valid_f0 = f0 > 0
    if np.any(valid_f0):
        ax.scatter(time_stamps[valid_f0], f0[valid_f0], color='red', label='Pitch Contour', s=10)
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_and_plot_audio(audio_path, output_dir, sr=16000):
    """
    Load, process, and plot audio data from a file, saving plots to specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    y, sr, f0, time_stamps = load_and_extract_features(audio_path, sr)
    if y is not None:
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        plot_waveform(y, sr, os.path.join(output_dir, f"{base_filename}_waveform.png"))
        plot_spectrogram(y, sr, os.path.join(output_dir, f"{base_filename}_spectrogram.png"))
        plot_pitch_contour(f0, time_stamps, os.path.join(output_dir, f"{base_filename}_pitch.png"))
        plot_spectrogram_and_pitch(y, sr, f0, time_stamps, os.path.join(output_dir, f"{base_filename}_combined.png"))

# Example usage
audio_path = r"s103u403n.wav"
output_directory = r"C:/world/checking_vocoder/New folder"
process_and_plot_audio(audio_path, output_directory)

