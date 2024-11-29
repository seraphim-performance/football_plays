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
video_folder = '/home/ubuntu/JBFS2024/dataset'
video_file = 'videos.txt'
prompt_file = 'labels.txt'

processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2")
dataset = FBVivitDataset(video_folder, video_file, prompt_file, processor)

# Split dataset (e.g., 80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
training_dataset, testing_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


#metric = load_metric("accuracy")
#def compute_metrics(p):
#    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([(torch.tensor(x['pixel_values']))  for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])}
#collate_fn = DefaultDataCollator()

with open(os.path.join(video_folder,prompt_file), 'r') as file:
    labelslist = file.readlines()

# Remove trailing newline characters from each line
labelslist = [line.strip() for line in labelslist]
mylabels = ["rush", "pass", "scramble", "other"]

config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
config.num_classes=4
config.id2label = {str(i): c for i, c in enumerate(mylabels)}
config.label2id = {c: str(i) for i, c in enumerate(mylabels)}
config.num_frames=frames
config.video_size=[frames, 224, 224]

model = AutoModelForVideoClassification.from_pretrained(
  "google/vivit-b-16x2-kinetics400",
  ignore_mismatched_sizes=True,
  config=config
  ).to(device)

training_args = TrainingArguments(
    output_dir="./results",         
    num_train_epochs=1,
    save_strategy="epoch",
    save_steps=10,             
    per_device_train_batch_size=4,   
    per_device_eval_batch_size=4,    
    learning_rate=5e-05,            
    weight_decay=0.01,              
    logging_dir="./logs",           
    logging_steps=10,                
    seed=42,                       
    eval_strategy="steps",    
    eval_steps=10,                   
    warmup_steps=int(0.1 * 20),      
    optim="adamw_torch",          
    lr_scheduler_type="linear",      
    fp16=True,                       
)


optimizer = AdamW(model.parameters(), lr=5e-05, betas=(0.9, 0.999), eps=1e-08)
model.to(device)


# Define the trainer
trainer = Trainer(
    model=model,                      
    args=training_args,
    data_collator=collate_fn,              
    train_dataset=training_dataset,      
    eval_dataset=testing_dataset,       
    optimizers=(optimizer, None),     
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
