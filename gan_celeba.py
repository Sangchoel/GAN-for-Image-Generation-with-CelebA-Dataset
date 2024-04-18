import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd

# CelebA 데이터셋을 위한 사용자 정의 데이터셋 클래스 정의
class CelebADataset(Dataset):
    def __init__(self, file):
        # 데이터셋 파일 오픈
        self.file_object = h5py.File(file, 'r')
        self.dataset = self.file_object['img_align_celeba']

    def __len__(self):
        # 데이터셋의 전체 길이 반환
        return len(self.dataset)

    def __getitem__(self, index):
        # 데이터셋에서 하나의 이미지를 인덱스를 통해 가져옴
        if index >= len(self.dataset):
            raise IndexError("Index out of bound")
        img = np.array(self.dataset[str(index) + '.jpg'])
        return torch.tensor(img, dtype=torch.float32) / 255.0  # 이미지를 텐서로 변환하고 정규화

# 판별자(Discriminator) 모델 아키텍처 정의
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # 이미지를 1차원으로 변환
            nn.Linear(3*218*178, 100),  # 첫 번째 선형 계층
            nn.LeakyReLU(0.2),  # LeakyReLU 활성화 함수
            nn.LayerNorm(100),  # LayerNorm을 통한 정규화
            nn.Linear(100, 1),  # 출력 계층
            nn.Sigmoid()  # Sigmoid 활성화 함수로 확률 출력
        )
        self.loss_function = nn.BCELoss()  # 이진 교차 엔트로피 손실 함수
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)  # Adam 옵티마이저

    def forward(self, inputs):
        return self.model(inputs)  # 입력 데이터를 모델에 통과시켜 결과를 반환

# 생성자(Generator) 모델 아키텍처 정의
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 3 * 10 * 10),  # 입력 노이즈를 받는 선형 계층
            nn.LeakyReLU(0.2),  # LeakyReLU 활성화 함수
            nn.LayerNorm(3 * 10 * 10),  # LayerNorm
            nn.Linear(3 * 10 * 10, 3 * 218 * 178),  # 이미지를 복원하는 선형 계층
            nn.Sigmoid()  # 이미지 픽셀 값을 [0,1] 범위로 압축
        )
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.01)  # Adam 옵티마이저

    def forward(self, inputs):
        output = self.model(inputs)
        return output.view(-1, 3, 218, 178)  # 출력을 이미지 형태로 재구성

# 유틸리티 함수
def generate_random_seed(size):
    # 주어진 크기의 랜덤 텐서 생성
    return torch.rand(size)

# 훈련 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 모델 및 데이터셋 인스턴스 생성
D = Discriminator().to(device)
G = Generator().to(device)
dataset = CelebADataset('/content/drive/MyDrive/Colab Notebooks/img_align_celeba.h5py')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 훈련 루프
epochs = 10
for epoch in range(epochs):
    for i, real_images in enumerate(dataloader):
        real_images = real_images.to(device)
        real_targets = torch.ones(real_images.size(0), 1).to(device)
        fake_targets = torch.zeros(real_images.size(0), 1).to(device)

        # 진짜 이미지로 판별자 훈련
        D.optimiser.zero_grad()
        real_loss = D.loss_function(D(real_images), real_targets)
        real_loss.backward()

        # 가짜 이미지로 판별자 훈련
        noise = generate_random_seed((real_images.size(0), 100)).to(device)
        fake_images = G(noise)
        fake_loss = D.loss_function(D(fake_images.detach()), fake_targets)
        fake_loss.backward()
        D.optimiser.step()

        # 생성자 훈련
        G.optimiser.zero_grad()
        generator_loss = D.loss_function(D(fake_images), real_targets)
        generator_loss.backward()
        G.optimiser.step()

        if i % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {i}, D_Loss: {real_loss + fake_loss}, G_Loss: {generator_loss}')

# 생성된 이미지 시각화
sample_size = 6
fig, axes = plt.subplots(1, sample_size, figsize=(10, 2))
for k in range(sample_size):
    noise = generate_random_seed(100).to(device)
    fake_image = G(noise).detach().cpu().numpy()

    if fake_image.ndim == 4 and fake_image.shape[0] == 1:
        fake_image = fake_image.squeeze(0)  # 첫 번째 배치 차원 제거

    fake_image = np.transpose(fake_image, (1, 2, 0))  # 채널을 마지막으로 이동
    axes[k].imshow(np.clip(fake_image, 0, 1))
    axes[k].axis('off')
plt.show()
