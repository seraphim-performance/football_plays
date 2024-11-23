import os
import subprocess
import re
import shutil

def traverse_directories(root_dir):
    # Walk through each subdirectory in the root directory
    for dirpath, _, filenames in os.walk(root_dir):
        # Find the first .mkv file in the current directory
        for filename in filenames:
            print(filename)
            if filename.endswith('.mkv'):
                # Get the full path to the file and the directory name
                directory_name = os.path.basename(dirpath)
                
                # Run the scenedetect command with the required parameters
                command = [
                    "scenedetect",
                    "-i", filename,
                    "split-video",
                    "-c",
                    "-f", f"{directory_name}-$SCENE_NUMBER"
                ]
                
                # Execute the command
                subprocess.run(command, cwd=dirpath, check=True)
                
                # Exit loop once the first .mkv file in the directory is processed
                break

# Specify the root directory to start traversal
#root_directory = os.getcwd()  # Update this path as needed
root_directory = "C:/Users/jchbe/Downloads/footballplays/23" 
traverse_directories(root_directory)