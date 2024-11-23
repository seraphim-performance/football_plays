import os
import re
import shutil

def organize_files(root_dir):
    # Regex pattern to match files like {directory_name}-{digits}.mp4
    pattern = re.compile(r"^(.*?)-(\d+)\.mp4$")
    
    # Walk through each directory and file in the root directory
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the filename matches the pattern
            match = pattern.match(filename)
            if match:
                directory_name = match.group(1)
                digits = int(match.group(2))

                # Determine destination subfolder and new filename
                if digits % 2 == 0:
                    subfolder = "front"
                    new_digits = digits // 2
                else:
                    subfolder = "side"
                    new_digits = (digits // 2) + 1  # Integer division
                
                # Define the new filename and the destination path
                new_filename = f"{directory_name}-{new_digits}.mp4"
                destination_folder = os.path.join(dirpath, subfolder)
                destination_path = os.path.join(destination_folder, new_filename)

                # Ensure the destination subfolder exists
                os.makedirs(destination_folder, exist_ok=True)

                # Move and rename the file
                original_path = os.path.join(dirpath, filename)
                shutil.move(original_path, destination_path)
                print(f"Moved {original_path} to {destination_path}")

# Specify the root directory to start traversal
root_directory = "C:/Users/jchbe/Downloads/footballplays/23"  # Update this path as needed
organize_files(root_directory)