from utils.data_utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns


def get_dataset_stats(base_path):
	def count_files_in_dir(path):
		if not os.path.isdir(path):
			return 0
		# Only count files, ignore subdirectories if that's intended; adjust if you want recursive
		return len([entry for entry in os.listdir(path) if os.path.isfile(os.path.join(path, entry))])

	dataset_stats = {}
	for category in os.listdir(base_path):
		category_path = os.path.join(base_path, category)
		if not os.path.isdir(category_path):
			continue

        # train/good
		train_good_dir = os.path.join(category_path, "train", "good")
		train_good = count_files_in_dir(train_good_dir)

        # test/good
		test_good_dir = os.path.join(category_path, "test", "good")
		test_good = count_files_in_dir(test_good_dir)

        # test defective types: look under test, skip "good" and non-directories        
		test_dir = os.path.join(category_path, "test")
		test_defective = 0
		defective_types = 0
		if os.path.isdir(test_dir):
			for entry in os.listdir(test_dir):
				entry_path = os.path.join(test_dir, entry)
				if entry == "good":
					continue 
				if os.path.isdir(entry_path):
					defective_types += 1
					test_defective += count_files_in_dir(entry_path)

		dataset_stats[category] = {
			"train (good)": train_good,
			"test (good)": test_good,
			"test (defective)": test_defective,
			"defective type": defective_types,
			"sum": train_good + test_good + test_defective,
		}

    # Totals
	total_train_good = sum(stats["train (good)"] for stats in dataset_stats.values())
	total_test_good = sum(stats["test (good)"] for stats in dataset_stats.values())
	total_test_defective = sum(stats["test (defective)"] for stats in dataset_stats.values())
	total_sum = sum(stats["sum"] for stats in dataset_stats.values())

	dataset_stats["Total"] = {
		"train (good)": total_train_good,
		"test (good)": total_test_good,
		"test (defective)": total_test_defective,
		"defective type": "-",  # aggregate not meaningful
		"sum": total_sum,
	}

	return dataset_stats

# Display dataset stats
def plot_dataset_distribution(stats, categories,save_viz):
    # Create the bar plot
	categories = [cat for cat in  categories if cat != 'Total']
	train_good_counts = [stats[cat]["train (good)"] for cat in categories]
	test_good_counts = [stats[cat]["test (good)"] for cat in categories]
	test_defective_counts = [stats[cat]["test (defective)"] for cat in categories]
	x = np.arange(len(categories))
    
	plt.figure(figsize=(15, 8))  # Increased height to accommodate the table
    
    # Bar plot (first subplot)
	plt.subplot(2, 1, 1)
	bar_width = 0.25
	plt.bar(x - bar_width, train_good_counts, bar_width, label="Train Normal Images")
	plt.bar(x, test_good_counts, bar_width, label="Test Normal Images")
	plt.bar(x + bar_width, test_defective_counts, bar_width, label="Test Defective Images")
	plt.xlabel("Categories")
	plt.ylabel("Number of Images")
	plt.title("Distribution of Normal and Defective Images per Category")
	plt.xticks(x, categories, rotation=45, ha="right")
	plt.legend()

    # Prepare data for the table
	table_data = pd.DataFrame(stats)
    # Table plot (second subplot)
	plt.subplot(2, 1, 2)
	plt.axis('off')
	table = plt.table(cellText=table_data.values,
					  colLabels=table_data.columns,
					  rowLabels=list(stats[categories[0]].keys()),
					  cellLoc='center',
					  loc='center')
    # Adjust table style
	table.auto_set_font_size(False)
	table.set_fontsize(10)
	table.scale(1.2, 1.5)  # Adjust table size

	plt.tight_layout()
	plt.savefig('/Users/juju/Documents/IDEC/Project/MTVec-AD-distribution.png') #change to your directory
	plt.show()


def visualize_samples(base_path, category, output_path=None):
	train_good_path = os.path.join(base_path, category, "train", "good")
	test_path = os.path.join(base_path, category, "test")

	good_samples = []
	if os.path.isdir(train_good_path):
		good_candidates = [
			f for f in os.listdir(train_good_path)
			if is_image_file(f) and os.path.isfile(os.path.join(train_good_path, f))
		]
		if good_candidates:
			good_samples = [("Good", os.path.join(train_good_path, good_candidates[0]))]

	defective_samples = []
	if os.path.isdir(test_path):
		for defect_type in os.listdir(test_path):
			defect_path = os.path.join(test_path, defect_type)
			if defect_type == "good" or not os.path.isdir(defect_path):
				continue  # good은 따로, .DS_Store 같은 파일은 건너뜀

			defect_candidates = [
				f for f in os.listdir(defect_path)
				if is_image_file(f) and os.path.isfile(os.path.join(defect_path, f))
			]
			if defect_candidates:
				defective_samples.append((defect_type, os.path.join(defect_path, defect_candidates[0])))

	all_samples = good_samples + defective_samples
	if not all_samples:
		print(f"No samples to visualize for category '{category}'.")
		return

	num_defective = len(defective_samples)
	if num_defective > 2:
		num_cols = (len(all_samples) + 1) // 2
		rows = 2
	else:
		num_cols = len(all_samples)
		rows = 1

	plt.figure(figsize=(15, rows * 4))
	for i, sample in enumerate(all_samples):
		label, img_path = sample
		plt.subplot(rows, num_cols, i + 1)
		try:
			img = Image.open(img_path)
			plt.imshow(img)
		except Exception:
			plt.text(0.5, 0.5, "Failed to open", ha="center", va="center", fontsize=8)
		plt.axis("off")
		plt.title(label)

	plt.suptitle(f"Samples from Category: {category}")
	plt.tight_layout()
	if output_path is None:
		output_path = os.path.join(save_path, f'{category}_MTVec-AD-samples.png')
	plt.savefig(output_path)
	plt.show()
	print(f"Saved sample figure to: {output_path}")



def visualize_ground_truth(base_path, category, output_path=None):
	ground_truth_path = os.path.join(base_path, category, "ground_truth")
	if not os.path.isdir(ground_truth_path):
		print(f"Ground truth path does not exist or is not a directory: {ground_truth_path}")
		return

    # defect type 목록: good 제외, 실제 디렉터리만
	defect_types = [
		d for d in os.listdir(ground_truth_path)
		if d != "good" and os.path.isdir(os.path.join(ground_truth_path, d))
	]
	if not defect_types:
		print(f"No defect types found in ground truth for category '{category}'.")
		return

	num_defective = len(defect_types)
	if num_defective > 2:
		num_cols = (len(defect_types) + 1) // 2
		num_rows = 2
	else:
		num_cols = len(defect_types)
		num_rows = 1

	plt.figure(figsize=(15, 5 * num_rows))

	for i, defect_type in enumerate(defect_types):
		defect_path = os.path.join(ground_truth_path, defect_type)

		defective_masks = []
		if not os.path.isdir(defect_path):
			continue  # 안전장치

		for img_file in os.listdir(defect_path):
			if not is_image_file(img_file):
				continue  # .DS_Store 등 비이미지 건너뜀
			img_path = os.path.join(defect_path, img_file)
			if not os.path.isfile(img_path):
				continue
			try:
				mask = Image.open(img_path).convert("L").resize((512, 512))
				mask = np.array(mask, dtype=np.float32) / 255.0
				defective_masks.append(mask)
			except Exception as e:
				print(f"Failed to load mask '{img_path}': {e}")
				continue

		if not defective_masks:
			print(f"No valid defective masks for defect type: {defect_type}")
			continue

		aggregated_mask = np.sum(defective_masks, axis=0)
		max_val = np.max(aggregated_mask)
		if max_val == 0:
			heatmap_normalized = aggregated_mask  # 모두 0인 경우
		else:
			heatmap_normalized = aggregated_mask / max_val

		plt.subplot(num_rows, num_cols, i + 1)
		sns.heatmap(
			heatmap_normalized,
			cmap="hot",
			cbar=False,
			xticklabels=False,
			yticklabels=False,
			square=True,
		)
		plt.title(f"Defect: {defect_type}")
		plt.axis("off")

	plt.tight_layout()
	if output_path is None:
		output_path = os.path.join(save_path, f'{category}_MTVec-AD-groundtruth.png')
	plt.savefig(output_path)
	plt.show()
	print(f"Saved ground truth heatmap to: {output_path}")


def plot_all_good_images(base_path, sample_count=5, output_path=None):
	categories = [
		cat for cat in os.listdir(base_path)
		if os.path.isdir(os.path.join(base_path, cat))
	]
	all_good_images = []
	all_labels = []

	for category in categories:
		train_good_path = os.path.join(base_path, category, "train", "good")
		if not os.path.isdir(train_good_path):
			continue

        # 필터링: 이미지 파일만
		candidates = [
			f for f in os.listdir(train_good_path)
			if is_image_file(f) and os.path.isfile(os.path.join(train_good_path, f))
		]
		if not candidates:
			continue

        # 샘플링 (파일 수가 적으면 가능한 만큼)
		sampled = candidates[:sample_count]
		for img_file in sampled:
			img_path = os.path.join(train_good_path, img_file)
			all_good_images.append(img_path)
			all_labels.append(category)

	num_images = len(all_good_images)
	if num_images == 0:
		print("No 'good' images found to plot.")
		return

	num_cols = min(5, num_images)
	num_rows = (num_images + num_cols - 1) // num_cols

	plt.figure(figsize=(15, num_rows * 3))
	for i, (img_path, label) in enumerate(zip(all_good_images, all_labels)):
		plt.subplot(num_rows, num_cols, i + 1)
		try:
			img = Image.open(img_path)
			plt.imshow(img)
		except Exception as e:
			plt.text(0.5, 0.5, f"Failed to open\n{os.path.basename(img_path)}",
					ha="center", va="center", fontsize=8)
		plt.axis("off")
		plt.title(label, fontsize=10)

	plt.tight_layout(rect=[0, 0, 1, 0.96])
	if output_path is None:
		output_path = os.path.join(save_path, f'{category}_MVTec-Good.png')
	plt.savefig(output_path)
	plt.show()
	print(f"Saved figure to: {output_path}")


def plot_good_and_defective_images(base_path, sample_count=5, output_path=None):
	"""
	Plots 'good' images and one random defective image per category side by side.

	Args:
		base_path (str): Path to the MVTec dataset.
		sample_count (int): Number of 'good' images to sample from each category.
	"""
	categories = [
		cat for cat in os.listdir(base_path)
		if os.path.isdir(os.path.join(base_path, cat))
	]
	all_images = []  # list of (path, label)

	for category in categories:
		train_good_path = os.path.join(base_path, category, "train", "good")
		test_path = os.path.join(base_path, category, "test")

        # Collect 'good' images (up to sample_count)
		if os.path.isdir(train_good_path):
			good_candidates = [
				f for f in os.listdir(train_good_path)
				if is_image_file(f) and os.path.isfile(os.path.join(train_good_path, f))
			]
			if good_candidates:
				sampled_good = good_candidates[:sample_count]
				for img_file in sampled_good:
					img_path = os.path.join(train_good_path, img_file)
					all_images.append((img_path, f"{category} (Good)"))

        # Collect one random defective image
		if os.path.isdir(test_path):
			defective_dirs = [
				d for d in os.listdir(test_path)
				if d != "good" and os.path.isdir(os.path.join(test_path, d))
			]
			if defective_dirs:
				random_defect = random.choice(defective_dirs)
				defect_path = os.path.join(test_path, random_defect)
				defect_candidates = [
					f for f in os.listdir(defect_path)
					if is_image_file(f) and os.path.isfile(os.path.join(defect_path, f))
				]
				if defect_candidates:
					random_defective_img = random.choice(defect_candidates)
					random_defective_path = os.path.join(defect_path, random_defective_img)
					all_images.append((random_defective_path, f"{category} (Defective: {random_defect})"))

	if not all_images:
		print("No images found to plot.")
		return

    # Grid sizing
	num_images = len(all_images)
	num_cols = 6
	num_rows = (num_images + num_cols - 1) // num_cols

	plt.figure(figsize=(15, num_rows * 3))
	for i, (img_path, label) in enumerate(all_images):
		plt.subplot(num_rows, num_cols, i + 1)
		try:
			img = Image.open(img_path)
			plt.imshow(img)
		except Exception:
			plt.text(0.5, 0.5, "Failed to open", ha="center", va="center", fontsize=8)
		plt.axis("off")
		plt.title(label, fontsize=10)

	plt.tight_layout(rect=[0, 0, 1, 0.96])
	if output_path is None:
		output_path = os.path.join(save_path, f'{category}_MTVec-AD-Example.png')
	plt.savefig(output_path)
	plt.show()
	print(f"Saved figure to: {output_path}")
