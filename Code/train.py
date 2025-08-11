import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from model import ConvAutoencoder
from preprocess import AutoencoderImageDataset

def plot_reconstruction(original, reconstructed, error_map, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 원본 이미지
    axes[0].imshow(original.transpose(1, 2, 0))
    axes[0].set_title("Original")
    axes[0].axis("off")

    # 복원 이미지
    axes[1].imshow(reconstructed.transpose(1, 2, 0))
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    # 에러 맵 (heatmap)
    axes[2].imshow(error_map, cmap="jet")
    axes[2].set_title("Error Heatmap")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 경로
    cache_dir = "/Users/juju/Documents/IDEC/Project/preprocessed"
    model_path = "/Users/juju/Documents/IDEC/Project/model/autoencoder.pth"
    save_viz_dir = "/Users/juju/Documents/IDEC/Project/test_results"
    os.makedirs(save_viz_dir, exist_ok=True)

    # 변환
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # 테스트 데이터셋 (good + defect 모두 포함)
    test_dataset = AutoencoderImageDataset(
        csv_file=os.path.join(cache_dir, "metadata.csv"),
        image_root_dir=cache_dir,
        transform=transform,
        use_only_good=True  # 결함 데이터 포함
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 테스트 & 시각화
    criterion = torch.nn.MSELoss(reduction='none')  # 픽셀 단위 loss 계산
    anomaly_scores = []

    with torch.no_grad():
        for idx, img in enumerate(test_loader):
            img = img.to(device)
            output = model(img)

            # Reconstruction error (pixel-wise)
            loss_map = criterion(output, img).mean(dim=1).squeeze().cpu().numpy()
            score = loss_map.mean()  # anomaly score (평균)
            anomaly_scores.append(score)

            # numpy 변환
            print(img.shape) #디버그용: 텐서모양 확인
            img_np = img.squeeze().cpu().numpy()#.transpose(1, 2, 0)
            print(img_np.shape) #디버그용
            img_np = np.transpose(img_np, (0, 2, 1))

            output_np = output.squeeze().cpu().numpy()#.transpose(1, 2, 0)
            output_np = np.transpose(output_np, (0, 2, 1))
            # 시각화 저장
            save_path = os.path.join(save_viz_dir, f"sample_{idx}.png")
            plot_reconstruction(img_np, output_np, loss_map, save_path)

    # Anomaly Score 통계 출력
    anomaly_scores = np.array(anomaly_scores)
    print(f"평균 Anomaly Score: {anomaly_scores.mean():.6f}")
    print(f"최소: {anomaly_scores.min():.6f}, 최대: {anomaly_scores.max():.6f}")

if __name__ == "__main__":
    main()

