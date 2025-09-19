## uav 프레임 -> 그대로 가져옴 / 위성 프레임 -> 해당 좌표를 직접 크롭해서 사이즈 맞춰서 보여줌. 
## 결과 위성 프레임, 지상 프레임의 비교 이미지 저장 
## output은 따로 output_data로 저장 
## 이때 선택은 랜덤 5개임

import h5py
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# 경로
sat_path = "datasets/maps/satellite/20201117_BingSatellite.png"
query_path = "datasets/satellite_0_thermalmapping_135/train_queries.h5"

# 출력 폴더
os.makedirs("sample_pairs_random", exist_ok=True)

# 위성 이미지 로드
sat_img = cv2.imread(sat_path)
sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
print("원본 위성:", sat_img.shape)

crop_size = 1536
half = crop_size // 2

# h5에서 랜덤 5개 쿼리 불러오기
with h5py.File(query_path, "r") as f:
    total_queries = len(f["image_name"])
    idxs = np.random.choice(total_queries, size=5, replace=False)  # 랜덤 5개 선택
    
    for i, idx in enumerate(idxs):
        qimg = f["image_data"][idx]
        qname = f["image_name"][idx].decode("utf-8")

        # 쿼리 이름에서 좌표 파싱
        _, x, y = qname.split("@")
        x, y = int(x), int(y)

        # 위성 crop
        crop = sat_img[y-half:y+half, x-half:x+half]
        crop_resized = cv2.resize(crop, (768, 768))

        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 원본 위성 + 박스
        sat_copy = sat_img.copy()
        cv2.rectangle(sat_copy, (x-half, y-half), (x+half, y+half), (255,0,0), 10)
        axes[0].imshow(sat_copy)
        axes[0].set_title("full satellite")
        axes[0].axis("off")

        # Crop된 위성 patch
        axes[1].imshow(crop_resized)
        axes[1].set_title("DB, Satellite Patch (768×768)")
        axes[1].axis("off")

        # Query 열화상
        axes[2].imshow(qimg, cmap="gray")
        axes[2].set_title("Query, UAV (768×768)")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = f"check_3_output/check_datasets_compare/pair_{i}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f" Saved {save_path} (idx={idx}, name={qname})")
