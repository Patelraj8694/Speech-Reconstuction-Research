import os
import shutil

def find_matching_files(base_dir1, base_dir2, extensions):
    # Dictionary to store the matched files paths grouped by extension
    matched_files = {ext: [] for ext in extensions}
    for ext in extensions:
        dir1 = os.path.join(base_dir1, ext)
        dir2 = os.path.join(base_dir2, ext)
        files1 = {f: os.path.join(dir1, f) for f in os.listdir(dir1) if f.endswith(ext)}
        files2 = {f: os.path.join(dir2, f) for f in os.listdir(dir2) if f.endswith(ext)}
        
        # Finding intersection of filenames and store full paths
        common_files = files1.keys() & files2.keys()
        for fname in common_files:
            matched_files[ext].append((files1[fname], files2[fname]))

    return matched_files

def copy_matched_files(matched_files, base_dir1, base_dir2, extensions):
    # Creating and copying to matched directories for each file extension
    for ext in extensions:
        dest_dir1 = os.path.join(base_dir1, ext + '_matched')
        dest_dir2 = os.path.join(base_dir2, ext + '_matched')
        os.makedirs(dest_dir1, exist_ok=True)
        os.makedirs(dest_dir2, exist_ok=True)
        
        for file_path1, file_path2 in matched_files[ext]:
            shutil.copy2(file_path1, dest_dir1)
            shutil.copy2(file_path2, dest_dir2)

# Example usage
base_dir1 = 'C:/laryngectomy/dataset_chunk/features/Normal'  # Base directory for Normal
base_dir2 = 'C:/laryngectomy/dataset_chunk/features/Whisper'  # Base directory for Whisper
extensions = ['mcc', 'f0', 'fv']  # Folder names and file extensions to match and copy

# Finding matching files
matched_files = find_matching_files(base_dir1, base_dir2, extensions)
# Copying matched files
copy_matched_files(matched_files, base_dir1, base_dir2, extensions)
