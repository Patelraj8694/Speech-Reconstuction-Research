#!/bin/bash

# Function to verify if the file is a WAV file
verify_wav() {
    local file_path="$1"
    # Verify the file is a WAV file by checking its format
    file "$file_path" | grep -q 'WAVE audio'
    if [ $? -ne 0 ]; then
        echo "Error: $file_path is not a valid WAV file"
        return 1
    fi
    echo "Verified: $file_path"
    return 0
}

# Function to convert the file to the desired format
convert_wav() {
    local input_path="$1"
    local output_path="$2"
    local filename="$3"
    local output_file="${output_path}/${filename}.wav"
    # Convert the file to the desired format and save it
    sox "$input_path" -r 16000 -c 1 -b 16 "$output_file"
    if [ $? -ne 0 ]; then
        echo "Error converting $input_path"
        return 1
    fi
    echo "Converted: $input_path -> $output_file"
    return 0
}

# Function to process the directory
process_directory() {
    local input_dir="$1"
    local output_dir="$2"
    
    mkdir -p "$output_dir"

    find "$input_dir" -type f -name "*.wav" | while read -r file_path; do
        folder_name=$(basename "$(dirname "$file_path")")
        base_name=$(basename "$file_path" .wav)
        filename="${base_name}_${folder_name}"

        echo "Processing file: $file_path"
        echo "Output will be saved to: $output_dir"

        verify_wav "$file_path"
        if [ $? -eq 0 ]; then
            convert_wav "$file_path" "$output_dir" "$filename"
        fi
    done
}

# Main execution
input_directory="/mnt/d/Distorted_Speech_Segmented_Data/Normal07"  # Replace with your input directory
output_directory="/mnt/c/laryngectomy/dataset/data/total_laryngectomy/Normal"  # Replace with your output directory

process_directory "$input_directory" "$output_directory"
