import os
import shutil

def delete_mkv_files_in_mp4_rich_directories(directory):
    for root, dirs, files in os.walk(directory):
        # Count .mp4 files in the current directory
        mp4_files = [file for file in files if file.endswith('.mp4')]
        
        # If there are three or more .mp4 files, delete .mkv files
        if len(mp4_files) >= 3:
            for file in files:
                if file.endswith('.mkv'):
                    mkv_path = os.path.join(root, file)
                    try:
                        os.remove(mkv_path)
                        print(f"Deleted: {mkv_path}")
                    except Exception as e:
                        print(f"Error deleting {mkv_path}: {e}")

# Specify the root directory you want to search
root_directory = "C:/Users/jchbe/Downloads/footballplays/23"
delete_mkv_files_in_mp4_rich_directories(root_directory)