import cv2
import os
import random
import re 

def sort_key(filename):
    match = re.search(r"-(\d+)\.mp4$", filename)
    return int(match.group(1)) if match else 0


plays_in_location = "C:/Users/jchbe/Downloads/footballplays/23"
plays_out_location = "C:/Users/jchbe/x/dataset/videos"
captions_file = "C:/Users/jchbe/x/dataset/captions.txt"
videos_file = "C:/Users/jchbe/x/dataset/videos.txt"

def shiftconverted(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames to drop
    frames_to_drop = total_frames % 4

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    dropped_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        # Break if video ends
        if not ret:
            break
        
        # Skip frames to make total frame count divisible by 4
        if dropped_frames < frames_to_drop:
            dropped_frames += 1
            continue

        # Write remaining frames to output video
        out.write(frame)
        frame_count += 1

    # Release everything when done
    cap.release()
    out.release()

def randomAngle():
    if(random.choice([0,1])):
        return "front"
    else:
        return "side"

def get_15_text(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()  # Read all lines from the file
    # Get the first 15 lines
    first_15_lines = lines[:15]
    # Get the last 15 lines
    last_15_lines = lines[-15:]
    return first_15_lines, last_15_lines


for i in range(3,19):
    week_name = os.path.join(plays_in_location, str(i))
    for game_name in os.listdir(week_name):
        dir_game_name = os.path.join(week_name,game_name)
        files = os.listdir(os.path.join(dir_game_name,"front"))
        mp4_files = sorted(files, key=sort_key)
        # Get the first and last 15 .mp4 files
        f15v = mp4_files[:15]
        l15v = mp4_files[-15:]
        f15t, l15t = get_15_text(os.path.join(dir_game_name, "captions.txt"))
        for fv, ft, in zip(f15v, f15t):
            fvsmall = fv[4:]
            with open(captions_file, 'a') as cap, open(videos_file, 'a') as vid:
                cap.write(ft)
                vid.write("videos/" + fvsmall + "\n")
            shiftconverted(os.path.join(dir_game_name,randomAngle(),fv),os.path.join(plays_out_location,fvsmall))
            print(fv)
        for lv, lt in zip(l15v, l15t):
            lvsmall = lv[4:]
            with open(captions_file, 'a') as cap, open(videos_file, 'a') as vid:
                cap.write(lt)
                vid.write("videos/" + lvsmall + "\n")
            shiftconverted(os.path.join(dir_game_name,randomAngle(),lv),os.path.join(plays_out_location,lvsmall))
            print(lv)