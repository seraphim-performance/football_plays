import torch
from transformers import Trainer, TrainingArguments, VivitConfig, VivitModel, AutoModelForVideoClassification, VivitImageProcessor
from transformers import AutoTokenizer, DefaultDataCollator
#from datasets import load_metric
import os
import numpy as np
import av
from torch.optim import AdamW
from torch import nn

device = torch.device("cuda")
frames = 48
skip_rate = 4

def read_video_pyav(container, indices):
  '''
  Decode the video with PyAV decoder.
    Args:
     container (`av.container.input.InputContainer`): PyAV container.
       indices (`List[int]`): List of frame indices to decode.
   Returns:
       result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
  '''
  frames = []
  container.seek(0)
  start_index = indices[0]
  end_index = indices[-1]
  for i, frame in enumerate(container.decode(video=0)):
    if i > end_index:
       break
    if i >= start_index and i in indices:
      reformatted_frame = frame.reformat(width=224,height=224)
      frames.append(reformatted_frame)
  new=np.stack([x.to_ndarray(format="rgb24") for x in frames])

  return new

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
  '''
  Sample a given number of frame indices from the video.
     Args:
       clip_len (`int`): Total number of frames to sample.
       frame_sample_rate (`int`): Sample every n-th frame.
       seg_len (`int`): Maximum allowed index of sample's last frame.
       Returns:
         indices (`List[int]`): List of sampled frame indices
  '''
  converted_len = int(clip_len * frame_sample_rate)
  end_idx = 32 + converted_len
  start_idx = 32
  indices = np.linspace(start_idx, end_idx, num=clip_len)
  indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
  return indices

def convert_label(s):
  if s == "rush":
    return 0
  if s == "pass":
    return 1
  if s == "scramble":
    return 2
  return 3

class FBVivitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, vid_list, note_list, image_processor):
        self.image_processor = image_processor
        self.root_dir = root_dir
        self.vid_list = []
        self.note_list = []
        with open(os.path.join(root_dir, vid_list), 'r') as file:
            self.vid_list = [os.path.join(self.root_dir,line.strip()) for line in file]
        with open(os.path.join(root_dir, note_list), 'r') as file:
            self.note_list = [line.strip() for line in file]

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, idx):
        with av.open(self.vid_list[idx]) as container:
          label = convert_label(self.note_list[idx])
          indices = sample_frame_indices(frames,skip_rate,container.streams.video[0].frames)
          video = read_video_pyav(container=container, indices=indices)
          inputs = self.image_processor(list(video), return_tensors="pt")
 #       print("***inputs")
 #       print(inputs)
 #       print("***pixelvalues")
#        print(inputs["pixel_values"])
        return {"pixel_values": inputs["pixel_values"].squeeze(), "labels": label}

# Set paths
#video_folder = '/home/ubuntu/JBFS2024/dataset'
#video_file = 'videos.txt'
#prompt_file = 'labels.txt'

processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2")


#def collate_fn(batch):
#    return {
#        'pixel_values': torch.stack([(torch.tensor(x['pixel_values']))  for x in batch]),
#        'labels': torch.tensor([x['labels'] for x in batch])}
#collate_fn = DefaultDataCollator()


# Remove trailing newline characters from each line
mylabels = ["rush", "pass", "scramble", "other"]

id2label = {str(i): c for i, c in enumerate(mylabels)}
label2id = {c: str(i) for i, c in enumerate(mylabels)}

#model = AutoModelForVideoClassification.from_pretrained(
#  "google/vivit-b-16x2-kinetics400",
#  ignore_mismatched_sizes=True,
#  config=config
#  ).to(device)

model = AutoModelForVideoClassification.from_pretrained("./results/checkpoint-1414")
model.eval()

optimizer = AdamW(model.parameters(), lr=5e-05, betas=(0.9, 0.999), eps=1e-08)
model.to(device)

# video clip consists of 300 frames (10 seconds at 30 FPS)
#file_path1 = "/home/ubuntu/JBFS2024/infervids/pass/247chie49ers-Scene-069.mp4"
#file_path2 = "/home/ubuntu/JBFS2024/infervids/rush/247texapack-Scene-006.mp4"
#file_path3 = "/home/ubuntu/JBFS2024/infervids/rush/247texapack-Scene-025.mp4"
file_path4 = "/home/ubuntu/JBFS2024/infervids/rush/247titabill-Scene-099.mp4"
#file_path5 = "/home/ubuntu/JBFS2024/infervids/other/247chie49ers-Scene-089.mp4"
#file_path6 = "/home/ubuntu/JBFS2024/infervids/other/247titabill-Scene-299.mp4"
#file_path7 = "/home/ubuntu/JBFS2024/infervids/scramble/247chie49ers-Scene-228.mp4"
#container = av.open(file_path)

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

def infer_vid(vidfile):
  with av.open(vidfile) as container:
    indices = sample_frame_indices(frames,skip_rate,container.streams.video[0].frames)
    video = read_video_pyav(container=container, indices=indices)
    inputs = image_processor(list(video), return_tensors="pt")
    input_tensor = inputs.to(device)
    outputs = model(**input_tensor)
    return outputs

# forward pass
output = infer_vid(file_path4)
print(output)
