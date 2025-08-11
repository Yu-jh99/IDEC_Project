import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import pandas as pd

from preprocess import *



# Set paths
base_path = '/Users/juju/Downloads/archive'  # Replace with the base path to the MVTec dataset
save_path = '/Users/juju/Documents/IDEC/Project' # Replace with the save path to the MVTec output figures

#def get_dataset_stats(base_path):
#    dataset_stats = {}
#
#    # Collect statistics for each category
#    for category in os.listdir(base_path):
#        category_path = os.path.join(base_path, category)
#        if os.path.isdir(category_path):
#            train_good = len(os.listdir(os.path.join(category_path, "train", "good")))
#            test_good = len(os.listdir(os.path.join(category_path, "test", "good")))
#            test_defective = sum(
#                len(os.listdir(os.path.join(category_path, "test", defect)))
#                for defect in os.listdir(os.path.join(category_path, "test"))
#                if defect != "good"
#            )
#            dataset_stats[category] = {
#                "train (good)": train_good,
#                "test (good)": test_good,
#                "test (defective)": test_defective,
#                "defective type": len(os.listdir(os.path.join(category_path, "test")))-1,
#                "sum": train_good + test_good + test_defective,
#            }
#
#    # Calculate totals for each field
#    total_train_good = sum(stats["train (good)"] for stats in dataset_stats.values())
#    total_test_good = sum(stats["test (good)"] for stats in dataset_stats.values())
#    total_test_defective = sum(stats["test (defective)"] for stats in dataset_stats.values())
#    total_defective_type = "-"  # Not meaningful to sum defective types
#    total_sum = sum(stats["sum"] for stats in dataset_stats.values())
#
#    # Add the total row to the dictionary
#    dataset_stats["Total"] = {
#        "train (good)": total_train_good,
#        "test (good)": total_test_good,
#        "test (defective)": total_test_defective,
#        "defective type": total_defective_type,
#        "sum": total_sum,
#    }
#
#    return dataset_stats





# Visualize a few samples
#def visualize_samples(base_path, category):
#    train_good_path = os.path.join(base_path, category, "train", "good")
#    test_path = os.path.join(base_path, category, "test")
#
#    good_samples = [os.path.join(train_good_path, img) for img in os.listdir(train_good_path)[:1]]
#
#    defective_samples = []
#    for defect_type in os.listdir(test_path):
#        if defect_type != "good":
#            defect_path = os.path.join(test_path, defect_type)
#            images = os.listdir(defect_path)
#            if images:
#                defective_samples.append((defect_type, os.path.join(defect_path, images[0])))
#
#    all_samples = [("Good", img) for img in good_samples] + defective_samples
#    num_defective = len(defective_samples)
#
#    if num_defective > 2:
#        # Split into two rows
#        num_cols = (len(all_samples) + 1) // 2  # Number of columns per row
#        rows = 2
#    else:
#        # Single row
#        num_cols = len(all_samples)
#        rows = 1
#
#    plt.figure(figsize=(15, rows * 4))
#
#    for i, sample in enumerate(all_samples):
#        if len(sample) == 2:  # Defective samples
#            defect_type, img_path = sample
#            title = defect_type
#        else:  # Good sample
#            img_path = sample[0]
#            title = sample[1]
#
#        plt.subplot(rows, num_cols, i + 1)
#        img = Image.open(img_path)
#        plt.imshow(img)
#        plt.axis("off")
#        plt.title(title)
#
#    plt.suptitle(f"Samples from Category: {category}")
#    plt.tight_layout()
#    plt.savefig('your_save_directory/MTVec-AD-samples.png')
#    plt.show()
#






##################################################################################
















    
# Visualize a few samples
#def visualize_ground_truth(base_path, category):
#    ground_truth_path = os.path.join(base_path, category, "ground_truth")
#
#    defect_types = [d for d in os.listdir(ground_truth_path) if d != "good"]
#    num_defective = len(defect_types)
#    if num_defective > 2:
#        # Split into two rows
#        num_cols = (len(defect_types) + 1) // 2  # Number of columns per row
#        num_rows = 2
#    else:
#        # Single row
#        num_cols = len(defect_types)
#        num_rows = 1
#    # Create a figure for the subplots
#    plt.figure(figsize=(15, 5 * num_rows))  # Adjust figure size dynamically
#    # Iterate through defect types (excluding 'good' category)
#    # Iterate through each defect type and create a heatmap subplot
#    for i, defect_type in enumerate(defect_types):
#        defect_path = os.path.join(ground_truth_path, defect_type)
#        
#        # List to store masks for this defect type
#        defective_masks = []
#        
#        # Load and resize masks
#        for img_file in os.listdir(defect_path):
#            img_path = os.path.join(defect_path, img_file)
#            mask = Image.open(img_path).convert('L').resize((512, 512))
#            mask = np.array(mask) / 255.0  # Normalize mask to 0-1 range
#            defective_masks.append(mask)
#        
#        # Skip if no masks are loaded
#        if not defective_masks:
#            print(f"No defective masks for defect type: {defect_type}")
#            continue
#        
#        # Aggregate masks
#        aggregated_mask = np.sum(defective_masks, axis=0)
#        heatmap_normalized = aggregated_mask / np.max(aggregated_mask)
#
#        # Create subplot
#        plt.subplot(num_rows, num_cols, i + 1)
#        sns.heatmap(heatmap_normalized, cmap="hot", cbar=False, xticklabels=False, yticklabels=False, square=True)
#        plt.title(f"Defect: {defect_type}")
#        plt.axis("off")
#    plt.tight_layout()
#    plt.savefig('your_save_directory/MTVec-AD-groundtruth.png')
#    plt.show()


######################################################################################





















#def plot_all_good_images(base_path, sample_count=5):
#    """
#    Plots 'good' images from all categories in a single plot.
#
#    Args:
#        base_path (str): Path to the MVTec dataset.
#        sample_count (int): Number of 'good' images to sample from each category.
#    """
#    categories = [cat for cat in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, cat))]
#    all_good_images = []
#    all_labels = []
#
#    # Collect images and their corresponding category labels
#    for category in categories:
#        train_good_path = os.path.join(base_path, category, "train", "good")
#        if os.path.exists(train_good_path):
#            good_images = os.listdir(train_good_path)[:sample_count]
#            for img_file in good_images:
#                img_path = os.path.join(train_good_path, img_file)
#                all_good_images.append(img_path)
#                all_labels.append(category)
#
#    # Calculate the grid size for the plot
#    num_images = len(all_good_images)
#    num_cols = min(5, num_images)  # Limit to 5 columns
#    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate rows needed
#
#    # Create the plot
#    plt.figure(figsize=(15, num_rows * 3))
#    for i, (img_path, label) in enumerate(zip(all_good_images, all_labels)):
#        plt.subplot(num_rows, num_cols, i + 1)
#        img = Image.open(img_path)
#        plt.imshow(img)
#        plt.axis("off")
#        plt.title(label, fontsize=10)
#    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
#    plt.savefig('your_save_directory/MVTec-Good.png')
#    plt.show()





###############################################################

########################################################################








import random
#def plot_good_and_defective_images(base_path, sample_count=5):
#    """
#    Plots 'good' and one random defective image from each category side by side.
#
#    Args:
#        base_path (str): Path to the MVTec dataset.
#        sample_count (int): Number of 'good' images to sample from each category.
#    """
#    categories = [cat for cat in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, cat))]
#    all_images = []
#    all_labels = []
#
#    for category in categories:
#        train_good_path = os.path.join(base_path, category, "train", "good")
#        test_path = os.path.join(base_path, category, "test")
#        
#        # Get 'good' images
#        if os.path.exists(train_good_path):
#            good_images = os.listdir(train_good_path)[:sample_count]
#            for img_file in good_images:
#                img_path = os.path.join(train_good_path, img_file)
#                all_images.append((img_path, f"{category} (Good)"))
#        
#        # Get one random defective image
#        defective_dirs = [d for d in os.listdir(test_path) if d != "good"]
#        if defective_dirs:
#            random_defect = random.choice(defective_dirs)
#            defect_path = os.path.join(test_path, random_defect)
#            defect_images = os.listdir(defect_path)
#            if defect_images:
#                random_defective_img = defect_images[0]
#                random_defective_path = os.path.join(defect_path, random_defective_img)
#                all_images.append((random_defective_path, f"{category} (Defective: {random_defect})"))
#
#    # Calculate grid size for the plot
#    num_images = len(all_images)
#    num_cols = 6  # Number of columns
#    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate rows needed
#
#    # Create the plot
#    plt.figure(figsize=(15, num_rows * 3))
#    for i, (img_path, label) in enumerate(all_images):
#        plt.subplot(num_rows, num_cols, i + 1)
#        img = Image.open(img_path)
#        plt.imshow(img)
#        plt.axis("off")
#        plt.title(label, fontsize=10)
#        
#    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
#    plt.savefig('your_save_directory/MTVec-AD-Example.png')
#    plt.show()



###############################################################################



















    
# Main script
if __name__ == "__main__":
    # Get dataset stats
    stats = get_dataset_stats(base_path)
    categories = list(stats.keys())
    # Plot dataset distribution
    plot_dataset_distribution(stats, categories)
    
    plot_all_good_images(base_path, sample_count=1)
    plot_good_and_defective_images(base_path, sample_count=1)
    # # Visualize samples from a specific category
    for category in categories:
        if category == 'Total':
            continue
        visualize_samples(base_path, category)
        visualize_ground_truth(base_path, category)
    cache_dir = "/Users/juju/Documents/IDEC/Project/preprocessed"
    df, mean, std = preprocess_mvtec(base_path, cache_dir)
    # # # Analyze image properties for a specific category
    # analyze_image_properties(base_path, category="bottle")
