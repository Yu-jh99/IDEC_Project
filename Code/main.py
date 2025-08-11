import os
from model import *
from preprocess import *
from viz import *
import torch
import torch.nn as nn
import torch.optim as optim
#from datasets.dataset import *
from torch.utils.data import *
import torchvision.transforms as transforms
from model import ConvAutoencoder



def main():
	# Set paths
	base_path = '/Users/juju/Downloads/archive'  # Replace with the base path to the MVTec dataset
	save_path = '/Users/juju/Documents/IDEC/Project' # Replace with the save path to the MVTec output figures
	save_viz = "/Users/juju/Documents/IDEC/Project"  # Replace with the
	cache_dir = "/Users/juju/Documents/IDEC/Project/preprocessed"  #Replace with the cache path to the preprocess 

	# 1.EDA on raw
	# Get dataset stats
	stats = get_dataset_stats(base_path)
	categories = list(stats.keys())

	# Plot dataset distribution
	plot_dataset_distribution(stats, categories, save_viz)
	plot_all_good_images(base_path, sample_count=1, output_path=os.path.join(save_viz, "good.png"))
	plot_good_and_defective_images(base_path, sample_count=1, output_path=os.path.join(save_viz, "good_vs_defect.png"))
	for category in categories:
		if category == "Total":
			continue
		visualize_samples(base_path, category, output_path=os.path.join(save_viz, f"{category}_samples.png"))
		visualize_ground_truth(base_path, category, output_path=os.path.join(save_viz, f"{category}_groundtruth.png"))

    # 2. Preprocess
	df, mean, std = preprocess_mvtec(base_path, cache_dir)

    # 이후 dataset/model/train/evaluate로 이어짐
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor()
	])
	
	train_dataset = AutoencoderImageDataset(
		csv_file = '/Users/juju/Documents/IDEC/Project/preprocessed/metadata.csv',
		image_root_dir = '/Users/juju/Documents/IDEC/Project/preprocessed',
		transform=transform,
		use_only_good = True
	)
	#print(train_dataset.df['cached_path'].dtype)
	#print(train_dataset.df['cached_path'].isnull().sum())
	#print(train_dataset.df['cached_path'].apply(len).describe())
	#print(train_dataset.df['cached_path'].head())
	#for idx in range(len(train_dataset)):
		#img_path = os.path.join(train_dataset.image_root_dir, train_dataset.data.loc[idx, 'cached_path'])
		#if not os.path.isfile(img_path):
			#print(f"Missing file: {img_path}")
	print(f"정상 데이터 개수: {len(train_dataset)}")
	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	print(f"train_dataset length: {len(train_dataset)}")
	for batch in train_loader:
		print(batch.shape)  # 예: torch.Size([32, 3, 256, 256])
		break
	
	# 3. 모델 학습
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = ConvAutoencoder().to(device)
	
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	
	num_epochs = 10
	for epoch in range(num_epochs):
		model.train()
		running_loss = 0.0
		for imgs in train_loader:
			imgs = imgs.to(device)
			outputs = model(imgs)
			loss = criterion(outputs, imgs)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

		print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")

	# save model
	save_model_path = save_path / model
	os.makedirs(save_model_path, exist_ok=True)
	torch.save(model.state_dict(), os.path.join(save_model_path, "autoencoder.pth"))
	print("모델 저장 완료")

if __name__ == "__main__":
	main()
