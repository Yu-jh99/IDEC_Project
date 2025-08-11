import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from utils.data_utils import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
# config defaults
IMAGE_SIZE = (512, 512)  # 원하는 크기
RANDOM_SEED = 42
NORMAL_LABEL = "normal"
DEFECT_LABEL = "defective"

def remove_ds_store(root: Path):
    for p in root.rglob(".DS_Store"):
        try:
            p.unlink()
        except Exception:
            pass


def preprocess_mvtec(base_path, cache_dir, val_split=0.1, test_split=0.1):
    """
    전처리: 크기 통일, 라벨링 메타데이터 생성, 평균/표준편차 계산, csv 저장.
    """
    random.seed(RANDOM_SEED)
    base_path = Path(base_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. 잡파일 제거
    remove_ds_store(base_path)

    rng = random.Random(RANDOM_SEED)
    records = []  # 메타데이터 수집
    # 각 카테고리 순회
    for category in sorted([d for d in base_path.iterdir() if d.is_dir()]):
        # TRAIN normal (good)
        train_good_dir = category / "train" / "good"
        if train_good_dir.is_dir():
            good_files = [f for f in train_good_dir.iterdir() if f.is_file() and is_image_file(f.name)]
            for img_path in good_files:
                # split into train/val (only for normal images)
                r = rng.random()
                split = "train"
                if r < val_split:
                    split = "val"
                elif r < val_split + test_split:
                    split = "test"

                out_subdir = cache_dir / category.name / split / NORMAL_LABEL
                out_subdir.mkdir(parents=True, exist_ok=True)

                # resize + save
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMAGE_SIZE, Image.BILINEAR)
                out_path = out_subdir / img_path.name
                img.save(out_path)

                records.append({
                    "category": category.name,
                    "original_path": str(img_path),
                    "cached_path": str(out_path.relative_to(cache_dir)),
                    "split": split,
                    "label": NORMAL_LABEL,
                    "defect_type": "good"
                })

        # TEST: normal & defective
        test_dir = category / "test"
        if test_dir.is_dir():
            # good images
            good_files = []
            good_dir = test_dir / "good"
            if good_dir.is_dir():
                good_files = [f for f in good_dir.iterdir() if f.is_file() and is_image_file(f.name)]
                for img_path in good_files:
                    out_subdir = cache_dir / category.name / "test" / NORMAL_LABEL
                    out_subdir.mkdir(parents=True, exist_ok=True)
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
                    out_path = out_subdir / img_path.name
                    img.save(out_path)
                    records.append({
                        "category": category.name,
                        "original_path": str(img_path),
                        "cached_path": str(out_path.relative_to(cache_dir)),
                        "split": "test",
                        "label": NORMAL_LABEL,
                        "defect_type": "good"
                    })

            # defective types
            for defect_type in sorted([d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]):
                defect_files = [f for f in defect_type.iterdir() if f.is_file() and is_image_file(f.name)]
                for img_path in defect_files:
                    out_subdir = cache_dir / category.name / "test" / DEFECT_LABEL / defect_type.name
                    out_subdir.mkdir(parents=True, exist_ok=True)
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
                    out_path = out_subdir / img_path.name
                    img.save(out_path)
                    records.append({
                        "category": category.name,
                        "original_path": str(img_path),
                        "cached_path": str(out_path.relative_to(cache_dir)),
                        "split": "test",
                        "label": DEFECT_LABEL,
                        "defect_type": defect_type.name
                    })

        # ground truth masks (for visualization / pairing)
        gt_root = category / "ground_truth"
        if gt_root.is_dir():
            for defect_type in sorted([d for d in gt_root.iterdir() if d.is_dir() and d.name != "good"]):
                for mask_path in defect_type.iterdir():
                    if not mask_path.is_file() or not is_image_file(mask_path.name):
                        continue
                    out_subdir = cache_dir / category.name / "ground_truth" / defect_type.name
                    out_subdir.mkdir(parents=True, exist_ok=True)
                    mask = Image.open(mask_path).convert("L")  # grayscale
                    mask = mask.resize(IMAGE_SIZE, Image.NEAREST)  # mask는 nearest
                    out_path = out_subdir / mask_path.name
                    mask.save(out_path)
                    # Note: ground truth might not be in the same metadata table; pair later if needed

    # 2. DataFrame 저장
    df = pd.DataFrame.from_records(records)
    meta_csv = cache_dir / "metadata.csv"
    df.to_csv(meta_csv, index=False)

    # 3. 평균/표준편차 계산 (train normal only)
    train_normals = df[(df["split"] == "train") & (df["label"] == NORMAL_LABEL)]
    mean_accum = np.zeros(3, dtype=np.float64)
    sq_accum = np.zeros(3, dtype=np.float64)
    count = 0
    for img_path in train_normals["cached_path"]:
        full_path = cache_dir / img_path
        img = Image.open(full_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC
        # mean over HxW then accumulate
        mean_accum += arr.mean(axis=(0, 1))
        sq_accum += (arr ** 2).mean(axis=(0, 1))
        count += 1
    if count > 0:
        mean = mean_accum / count
        var = sq_accum / count - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-6))
    else:
        mean = np.zeros(3)
        std = np.ones(3)

    norm_stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    # 저장
    np.save(cache_dir / "norm_mean.npy", mean)
    np.save(cache_dir / "norm_std.npy", std)
    with open(cache_dir / "norm_stats.json", "w") as f:
        import json
        json.dump(norm_stats, f, indent=2)

    print(f"Preprocessing done. Metadata: {meta_csv}, normalization stats: {norm_stats}")
    return df, mean, std



class AutoencoderImageDataset(Dataset):
    def __init__(self, csv_file, image_root_dir, transform=None, use_only_good=False):
        self.image_root_dir = image_root_dir
        self.transform = transform

        # CSV 로드
        df = pd.read_csv(csv_file)

        if use_only_good:
            # 정상 데이터만 선택: train + good
            df = df[(df['split'].str.lower() == 'train') &
                    (df['defect_type'].str.lower() == 'good')] 

        # 캐시 경로(cached_path) 기준으로 실제 존재하는 파일만 필터
        df = df[df['cached_path'].apply(lambda p: os.path.isfile(os.path.join(image_root_dir, p)))]

        self.data = df.reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError("정상 데이터가 없습니다. 필터 조건을 확인하세요.")

        for idx in range(len(self.data)):
            img_path = os.path.join(self.image_root_dir, self.data.loc[idx, 'cached_path'])
            if not os.path.isfile(img_path):
                print(f"Missing file: {img_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root_dir, self.data.loc[idx, 'cached_path'])
        image = Image.open(img_path).convert('RGB')# full_path 아니라 img_path였음

        if self.transform:
            image = self.transform(image)

        return image

# 공용 transform 예시
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

