import os
import shutil

def find_matching_files(dir1, dir2):
    # Maps to hold filenames and their full paths for both directories
    files1 = {}
    files2 = {}
    
    # Traverse the first directory (normal) and add files to the map
    for root, _, filenames in os.walk(dir1):
        for filename in filenames:
            if filename.endswith('.wav'):
                files1[filename] = os.path.join(root, filename)

    # Traverse the second directory (whisper) and add files to the map
    for root, _, filenames in os.walk(dir2):
        for filename in filenames:
            if filename.endswith('.wav'):
                files2[filename] = os.path.join(root, filename)

    # Find intersection of keys (filenames) and return paths
    matched_files = {fname: (files1[fname], files2[fname]) for fname in files1 if fname in files2}
    return matched_files

def copy_matched_files(matched_files, dest_dir1, dest_dir2):
    # Ensure the destination directories exist
    os.makedirs(dest_dir1, exist_ok=True)
    os.makedirs(dest_dir2, exist_ok=True)
    
    # Copy matching files to their respective directories
    for filename, paths in matched_files.items():
        shutil.copy2(paths[0], os.path.join(dest_dir1, filename))
        shutil.copy2(paths[1], os.path.join(dest_dir2, filename))

# Example usage
dir1 = 'C:/laryngectomy/dataset_chunk/data/Normal'  # Replace with the actual path to the normal directory
dir2 = 'C:/laryngectomy/dataset_chunk/data/Whisper'  # Replace with the actual path to the whisper directory
dest_dir1 = 'C:/laryngectomy/dataset_chunk/data/Normal_matched'  # Destination for matched normal files
dest_dir2 = 'C:/laryngectomy/dataset_chunk/data/Whisper_matched'  # Destination for matched whisper files

# Find matching files
matched_files = find_matching_files(dir1, dir2)
# Copy matched files
copy_matched_files(matched_files, dest_dir1, dest_dir2)
