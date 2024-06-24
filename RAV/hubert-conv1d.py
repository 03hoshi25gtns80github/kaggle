import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, HubertModel
import pickle
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

RAV = '/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/'
RAV_list = os.listdir(RAV)
RAV_test = []
for i in range(3):
    RAV_test.append(RAV_list[i])
    
def extract_label(filename):
    part = filename.split('.')[0].split('-')
    label = (int(part[4])-1)
    return label

# 最大長さを格納する変数
max_length = 0

# 各オーディオファイルの長さを計算し、最大長さを見つける
for i in RAV_test:
    RAV_dir = os.path.join(RAV, i)
    filenames = os.listdir(RAV_dir)
    for filename in filenames:
        file_path = os.path.join(RAV_dir, filename)
        raw_speech, sr = sf.read(file_path)
        
        # サンプリングレートを16000にリサンプリング
        if sr != 16000:
            raw_speech = librosa.resample(raw_speech, orig_sr=sr, target_sr=16000)
        
        length = len(raw_speech)
        if length > max_length:
            max_length = length

print(f"最大オーディオ長さ: {max_length} サンプル")

# 最大長さ
max_length = 209809

# バッチサイズ
batch_size = 5

# ゼロパディングされたオーディオデータを格納するリスト
padded_audio_data = []

# バッチ処理用の一時リスト
temp_batch = []

# 各オーディオファイルをゼロパディングしてリストに追加
for i in RAV_test:
    RAV_dir = os.path.join(RAV, i)
    filenames = os.listdir(RAV_dir)
    for filename in filenames:
        file_path = os.path.join(RAV_dir, filename)
        raw_speech, sr = sf.read(file_path)
        
        # サンプリングレートを16000にリサンプリング
        if sr != 16000:
            raw_speech = librosa.resample(raw_speech, orig_sr=sr, target_sr=16000)
        
        length = len(raw_speech)
        
        # ゼロパディング
        if length < max_length:
            padded_speech = np.pad(raw_speech, (0, max_length - length), 'constant')
        else:
            padded_speech = raw_speech[:max_length]  # 長すぎる場合はトリミング
        
        # ラベルを取得（例としてファイル名からラベルを抽出）
        label = extract_label(filename)
        
        temp_batch.append((padded_speech, label))
        
        # バッチが満たされたらメモリに追加
        if len(temp_batch) == batch_size:
            padded_audio_data.extend(temp_batch)
            temp_batch = []

# 残りのデータを追加
if temp_batch:
    padded_audio_data.extend(temp_batch)

print("すべてのオーディオデータがゼロパディングされました。")

train, test = train_test_split(padded_audio_data, test_size=0.2, random_state=42)

# カスタムデータセットクラスの定義
class AudioDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_speech, label = self.dataset[idx]
        inputs = self.feature_extractor(raw_speech, return_tensors="pt", sampling_rate=16000)
        inputs['input_values'] = inputs['input_values'].transpose(0, 1)
        return inputs['input_values'], label
    
train_loader = DataLoader(AudioDataset(train), batch_size=2, shuffle=True)

model_name = "facebook/hubert-base-ls960"
hubert_model = HubertModel.from_pretrained(model_name)

# カスタム分類モデルの定義
class ClassificationModel(nn.Module):
    def __init__(self, hubert_model, num_labels):
        super(ClassificationModel, self).__init__()
        self.hubert = hubert_model
        self.conv1 = nn.Conv1d(in_channels=self.hubert.config.hidden_size, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(512)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_values):
        print(f"Input shape: {input_values.shape}")
        
        outputs = self.hubert(input_values)
        hidden_states = outputs.last_hidden_state
        print(f"Hidden states shape: {hidden_states.shape}")
        
        # 1D-CNNの適用
        x = hidden_states.transpose(1, 2)  # (batch_size, in_channels, sequence_length)に変更
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        print(f"Shape after Conv1, BN1, and Pool1: {x.shape}")
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        print(f"Shape after Conv2, BN2, and Pool2: {x.shape}")
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.global_avg_pool(x)
        print(f"Shape after Conv3, BN3, and GlobalAvgPool: {x.shape}")
        
        x = x.squeeze(-1)  # チャネル次元を削除
        print(f"Shape after squeezing: {x.shape}")
    
        logits = self.fc(x)
        print(f"Shape after FC layer: {logits.shape}")
        
        probs = self.softmax(logits)
        print(f"Output shape: {probs.shape}")
        
        return probs
    
num_labels = 2
model = ClassificationModel(hubert_model, num_labels)
model.to(device)

# 損失関数と最適化手法の設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

num_epochs = 1
model.train()
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # デバッグ用のprint文を追加
        print(f"Batch input shape before squeeze: {inputs.shape}")
        inputs = inputs.squeeze(2)  # 形状を (batch_size, sequence_length) に整形
        print(f"Batch input shape after squeeze: {inputs.shape}")
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
         # labelsをテンソルに変換し、形状を (batch_size,) に整形
        if isinstance(labels, tuple):
            labels = torch.tensor(labels)
        labels = labels.view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    
