## 특정 데이터셋에서 특정 기준 직선 경로 query만 뽑아옴. 
## input 실제 데이터셋 속 하나 
## output trajectory 용 database -> t_datasets/datasets

import h5py
import os

query_h5_path = "datasets/satellite_0_thermalmapping_135/train_queries.h5"
output_h5_path = "t_datasets/queries/train_queries_row3131.h5"
os.makedirs("row_query", exist_ok=True)

def parse_coord(name):
    parts = name.split("@")
    return int(parts[-2]), int(parts[-1])  # (row, col)

## 여기 확인 
target_row = 3131

with h5py.File(query_h5_path, "r") as f, h5py.File(output_h5_path, "w") as f_out:
    query_names = [n.decode("utf-8") for n in f["image_name"][:]]
    
    selected_idxs = [i for i, q in enumerate(query_names) if parse_coord(q)[0] == target_row]

    # 이름 저장
    selected_names = [query_names[i] for i in selected_idxs]
    f_out.create_dataset("image_name", data=[n.encode("utf-8") for n in selected_names])

    # 이미지 데이터 → 필요한 인덱스만 개별로 읽어서 저장
    f_out.create_dataset(
        "image_data",
        data=[f["image_data"][i] for i in selected_idxs]
    )

print(f"row {target_row} H5 파일 저장 완료 → {output_h5_path}")
