import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AutoencoderImageDataset(Dataset):
	def __init__(self, csv_file, image_root_dir, transform=None, use_only_good=True):
		"""
		Args:
			csv_file (str): metadata.csv의 경로
			image_root_dir (str): 이미지가 저장된 폴더 경로
			transform (callable, optional): 이미지에 적용할 transform
			use_only_good (bool): 'good' 라벨만 사용할지 여부
		"""
		self.df = pd.read_csv(csv_file)
		if use_only_good:
			self.df = self.df[self.df['label'] == 'good'].reset_index(drop=True)

		self.image_root_dir = image_root_dir
		self.transform = transform if transform is not None else transforms.ToTensor()

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		img_name = os.path.join(self.image_root_dir, self.df.iloc[idx]['filename'])
		image = Image.open(img_name).convert("RGB")  # 흑백이면 L로 바꿔도 됨
		if self.transform:
			image = self.transform(image)
		return image, image  # (input, target)
