import os
import re

# Define the root directory containing your .mp4 files
root_dir = "C:/Users/jchbe/Downloads/footballplays/23"

# Regex pattern to match filenames with 10 digits, a hyphen, and some digits before .mp4
pattern = re.compile(r"(\d{10})-(\d+)\.mp4$")

# Walk through the directory
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        match = pattern.match(filename)
        if match:
            # Get the parts of the filename
            prefix = match.group(1)
            number = match.group(2)
            
            # Zero-pad the last digits to 3 places
            new_filename = f"{prefix}-{int(number):03}.mp4"
            
            # Rename the file if the new filename is different
            if new_filename != filename:
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")