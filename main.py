import torch
import torchvision.transforms as transforms
from torch import nn
from torchvision import models
import cv2
import numpy as np
from pathlib import Path

#definim modelul realizat
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=False)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


#pregatirea imaginilor
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
softmax = nn.Softmax(dim=1)


def extract_frames(video_path, sequence_length=20):
    #extragere cadre video
    vid_obj = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // sequence_length)

    for i in range(total_frames):
        success, frame = vid_obj.read()
        if not success:
            break
        if i % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(train_transforms(frame))
            if len(frames) == sequence_length:
                break
    vid_obj.release()
    if len(frames) < sequence_length:
        print(f"avertisment: nu s-au putut extrage {sequence_length} cadre din {video_path}")
    return torch.stack(frames).unsqueeze(0) if frames else None

def predict(model, video_path):
    frames = extract_frames(video_path)
    if frames is None:
        return None, 0.0
    frames = frames.to(device)
    with torch.no_grad():
        _, logits = model(frames)
        logits = softmax(logits)
        prediction = logits.argmax(dim=1).item()
        confidence = logits.max().item() * 100
    return prediction, confidence


#incarca modelul si pregateste predictia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=2).to(device)  #clasele fake si real
model_path = "model_final_data.pt"

#incarca starea modelului
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

#procesam toate videoclipurile din folder
video_folder = Path("video_real") 
video_files = list(video_folder.glob("*.mp4"))


for video_path in video_files:
    prediction, confidence = predict(model, str(video_path))
    if prediction == 1:
        print(f"videoclipul {video_path.name} este real cu o incredere de {confidence:.2f}%")
    else:
        print(f"videoclipul {video_path.name} este fake cu o incredere de {confidence:.2f}%")
