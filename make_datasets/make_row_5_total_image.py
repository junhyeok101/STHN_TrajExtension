# query- database 쌍이 있는 데이터 베이스를 만들게 됨.

import h5py
import cv2
import matplotlib.pyplot as plt
import os

# 경로
sat_path = "datasets/maps/satellite/20201117_BingSatellite.png"
row_query_path = "t_datasets/queries/train_queries_row3131.h5"
row_db_path = "t_datasets/database/train_database_row3131.h5"

# 출력 폴더
save_dir = "t_datasets/total_image"
os.makedirs(save_dir, exist_ok=True)

# 원본 위성 이미지 로드
sat_img = cv2.imread(sat_path)
sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

# query 읽기
with h5py.File(row_query_path, "r") as fq:
    query_names = [n.decode("utf-8") for n in fq["image_name"][:]]
    query_imgs = fq["image_data"][:]  # (N, 768, 768, 3)

# database 읽기 (좌표만)
with h5py.File(row_db_path, "r") as fd:
    db_names = [n.decode("utf-8") for n in fd["image_name"][:]]

def parse_coord(name):
    parts = name.split("@")
    return int(parts[-2]), int(parts[-1])

# query와 database 짝 맞춰 저장
for i, (qname, qimg) in enumerate(zip(query_names, query_imgs)):
    r, c = parse_coord(qname)

    # full satellite에서 crop
    crop_size = 1536
    half = crop_size // 2
    crop = sat_img[r-half:r+half, c-half:c+half]
    crop_resized = cv2.resize(crop, (768, 768))

    # 대응하는 database 이미지 (원본 위성에서 crop)
    # query 좌표와 동일한 col 기준으로 database crop
    if qname in db_names:
        db_r, db_c = parse_coord(qname)
        db_crop = sat_img[db_r-half:db_r+half, db_c-half:db_c+half]
        db_resized = cv2.resize(db_crop, (768, 768))
    else:
        db_resized = crop_resized  # fallback

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    sat_copy = sat_img.copy()
    cv2.rectangle(sat_copy, (c-half, r-half), (c+half, r+half), (255,0,0), 10)

    axes[0].imshow(sat_copy)
    axes[0].set_title("Full Satellite", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(crop)
    axes[1].set_title("Crop (1536) -> resize(786)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(qimg, cmap="gray")
    axes[2].set_title("Query (Thermal 768)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{i:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    if i % 20 == 0:
        print(f"Saved {i}/{len(query_names)} images")

print("3131 이미지 쌍 시각화 완료")
