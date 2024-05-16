import os
import subprocess
from pathlib import Path

# Define the source and target directories
source_dir = Path("D:/Distorted Speech Segmented Data")
target_dir = Path("C:/laryngectomy/dataset/data")

# Function to process and convert files
def process_files(subfolder, category, folder_suffix):
    files = list(subfolder.glob('*.wav'))
    if not files:
        return  # Skip if no files found

    # Ensure target directory exists
    category_dir = target_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Process each .wav file
    for file_path in sorted(files):
        new_file_name = f"{file_path.stem}_{folder_suffix}.wav"  # Append folder number
        new_file_path = category_dir / new_file_name
        
        # Command to convert sampling rate using sox
        # Ensure paths are quoted to handle spaces
        cmd = f'sox "{file_path}" -r 16000 "{new_file_path}"'
        subprocess.run(cmd, shell=True)
        print(f"Processed {file_path} to {new_file_path}")

# Loop through the folders FOLDER01 to FOLDER12
for i in range(1, 13):
    folder_name = f"FOLDER{i:02d}"
    folder_path = source_dir / folder_name
    print(f"Checking {folder_path}")

    # Process 'Normal' and 'Whisper' folders if they exist
    if (folder_path / 'Normal').exists():
        process_files(folder_path / 'Normal', 'Normal', folder_name)
    if (folder_path / 'Whisper').exists():
        process_files(folder_path / 'Whisper', 'Whisper', folder_name)
