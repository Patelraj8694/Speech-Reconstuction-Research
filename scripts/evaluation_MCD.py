from pathlib import Path
from mel_cepstral_distance import get_metrics_wavs

# Convert string paths to Path objects
wav_file_1 = Path(r"C:\laryngectomy\scripts\refrence\fn001.wav")
wav_file_2 = Path(r"C:\laryngectomy\scripts\genrated\generated_fn001.wav")

# Ensure the files exist
if not wav_file_1.is_file() or not wav_file_2.is_file():
    raise FileNotFoundError("One of the audio files does not exist.")

# Since you mentioned wanting to use n_mfcc=40, ensure your function can handle it
# If the default function does not support n_mfcc=40, you may need to modify the function definition accordingly
# Here we call the function assuming it supports n_mfcc=40 or has been modified to do so
mcd, penalty, frames = get_metrics_wavs(wav_file_1, wav_file_2, n_mfcc=40)
print(f"Mel-Cepstral Distance: {mcd}, Penalty: {penalty}, Frames: {frames}")
