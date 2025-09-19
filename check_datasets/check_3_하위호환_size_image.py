## 위성이미지 전체, 1536자른 버전, 786리사이즈 버전, 지상이미지 786을 output으로 내는 코드 

import h5py
import cv2
import matplotlib.pyplot as plt
import os

# 경로
sat_path = "datasets/maps/satellite/20201117_BingSatellite.png"
query_path = "datasets/satellite_0_thermalmapping_135/test_queries.h5"

# 출력 폴더
os.makedirs("sample_vis", exist_ok=True)

# 열화상 쿼리 하나 불러오기
with h5py.File(query_path, "r") as f:
    query_img = f["image_data"][0]          # (768,768,3)
    query_name = f["image_name"][0].decode("utf-8")
    print("쿼리 이름:", query_name)          # 예: @2956@2161

# 좌표 파싱 (@x@y)
_, x, y = query_name.split("@")
x, y = int(x), int(y)

# 원본 위성 이미지 로드
sat_img = cv2.imread(sat_path)
sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
print("원본 위성:", sat_img.shape)

# Crop: 1536x1536 영역 자르기
crop_size = 1536
half = crop_size // 2
crop = sat_img[y-half:y+half, x-half:x+half]

# Resize (768x768, database 이미지와 동일)
crop_resized = cv2.resize(crop, (768, 768))

# 시각화: [원본 일부(박스표시), Crop(1536), Crop Resized(768), Query]
fig, axes = plt.subplots(1, 4, figsize=(22, 6))

# 원본 위성 (crop 영역 표시)
sat_copy = sat_img.copy()
cv2.rectangle(sat_copy, (x-half, y-half), (x+half, y+half), (255,0,0), 10)  # 파란색 박스
axes[0].imshow(sat_copy)
axes[0].set_title("full satellite", fontsize=12)
axes[0].axis("off")

# Crop된 1536 패치
axes[1].imshow(crop)
axes[1].set_title("Crop (1536×1536)", fontsize=12)
axes[1].axis("off")

# Resize 후 Database 저장 형태
axes[2].imshow(crop_resized)
axes[2].set_title("Database (768×768)", fontsize=12)
axes[2].axis("off")

# Query 열화상
axes[3].imshow(query_img, cmap="gray")  # 열화상은 흑백이 적합
axes[3].set_title("Query (UAV Thermal, 768×768)", fontsize=12)
axes[3].axis("off")

# 레이아웃 정리 및 저장
plt.tight_layout()
plt.subplots_adjust(top=0.88)   # 상단 여백 확보 (0~1 비율, 보통 0.85~0.9 정도)
save_path = "check_3_output/check_size_image/compare_satellite_query_full.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"✅ Saved {save_path}")
