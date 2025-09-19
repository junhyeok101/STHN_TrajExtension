## 특정 데이터셋에서 query와 맞는 database를 뽑아옴 . 
## input 실제 데이터셋 속 하나 
## output trajectory 용 database -> t_datasets/datasets

import h5py
import os

db_h5_path = "datasets/satellite_0_thermalmapping_135/train_database.h5"
output_dir = "t_datasets/datasets"
os.makedirs(output_dir, exist_ok=True)

def parse_coord(name):
    parts = name.split("@")
    return int(parts[-2]), int(parts[-1])  # (row, col)

target_row = 3131
output_h5_path = os.path.join(output_dir, f"train_database_row{target_row}.h5")

with h5py.File(db_h5_path, "r") as f, h5py.File(output_h5_path, "w") as f_out:
    db_names = [n.decode("utf-8") for n in f["image_name"][:]]
    db_sizes = f["image_size"][:]   # (N,2) 배열 같은 형태일 가능성 높음

    # target row 필터링
    selected_idxs = [i for i, n in enumerate(db_names) if parse_coord(n)[0] == target_row]
    selected_names = [db_names[i] for i in selected_idxs]
    selected_sizes = db_sizes[selected_idxs]

    # 새 h5에 저장
    f_out.create_dataset("image_name", data=[n.encode("utf-8") for n in selected_names])
    f_out.create_dataset("image_size", data=selected_sizes)

    print(f"총 {len(selected_names)}개 database 메타데이터 저장 완료 → {output_h5_path}")
