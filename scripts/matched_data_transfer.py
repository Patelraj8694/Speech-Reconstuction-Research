import os
from pathlib import Path
import shutil

# Define the directories for 'Normal' and 'Whisper'
base_dir = Path(r'C:\laryngectomy\dataset\features')
normal_dir_base = base_dir / 'Normal'
whisper_dir_base = base_dir / 'Whisper'

# File types and corresponding folders for matched files
file_types_folders = {'mcc': 'mcc_matched', 'f0': 'f0_matched', 'fv': 'fv_matched'}

# Function to check if the first three letters and the last two characters of filenames match
def is_match(normal_filename, whisper_filename):
    return (normal_filename[:3] == whisper_filename[:3] and 
            normal_filename[-2:] == whisper_filename[-2:])

# Process each file type
for file_type, matched_folder_name in file_types_folders.items():
    normal_dir = normal_dir_base / file_type
    whisper_dir = whisper_dir_base / file_type
    normal_matched_dir = normal_dir_base / matched_folder_name  # Matched folder in the Normal directory
    whisper_matched_dir = whisper_dir_base / matched_folder_name  # Matched folder in the Whisper directory
    normal_matched_dir.mkdir(exist_ok=True)
    whisper_matched_dir.mkdir(exist_ok=True)

    # Get a list of normal files
    normal_files = [f for f in os.listdir(normal_dir) if f.endswith(file_type)]

    # Counter for the number of matched files
    normal_matched_count = 0
    whisper_matched_count = 0

    # Match and copy files from Normal to matched folder in Normal directory, and Whisper to matched folder in Whisper directory
    for whisper_file in os.listdir(whisper_dir):
        if whisper_file.endswith(file_type):
            # Find a matching normal file
            for normal_file in normal_files:
                if is_match(normal_file, whisper_file):
                    # Copy the Normal file to the matched directory in Normal
                    shutil.copy(normal_dir / normal_file, normal_matched_dir / normal_file)
                    normal_matched_count += 1
                    print(f"Copied {normal_file} to Normal matched directory")
                    # Copy the Whisper file to the matched directory in Whisper
                    shutil.copy(whisper_dir / whisper_file, whisper_matched_dir / whisper_file)
                    whisper_matched_count += 1
                    print(f"Copied {whisper_file} to Whisper matched directory")
                    break

    print(f"Total matched {file_type} files in Normal: {normal_matched_count}")
    print(f"Total matched {file_type} files in Whisper: {whisper_matched_count}")

print("Finished organizing matched files.")
