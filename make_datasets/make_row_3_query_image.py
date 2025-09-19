## 해당 쿼리 이미지 저장 -
## input 이미 만든 t_datasets/??_datasets/train_queries_row3131.h5
## output 해당 query 이미지들 모두 저장 -> t_datasets/query_image

import h5py
import os
import cv2  # OpenCV로 이미지 저장

query_h5_path = "t_datasets/queries/train_queries_row3131.h5"
output_dir = "t_datasets/query_image"
os.makedirs(output_dir, exist_ok=True)

with h5py.File(query_h5_path, "r") as f:
    query_names = [n.decode("utf-8") for n in f["image_name"][:]]
    image_data = f["image_data"]  # HDF5 dataset (lazy loading)

    for i, name in enumerate(query_names):
        img = image_data[i]  # (H,W,C) numpy 배열
        # HDF5에서 RGB가 BGR 순서일 수 있으니 OpenCV 저장 전에 변환
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 안전한 파일명 (쿼리 이름 그대로 저장)
        filename = f"{name}.png"
        save_path = os.path.join(output_dir, filename)

        cv2.imwrite(save_path, img_bgr)

        if i % 500 == 0:  # 중간 진행 상황 표시
            print(f"{i}/{len(query_names)} saved")

print(f"모든 이미지 저장 완료 → {output_dir}")
