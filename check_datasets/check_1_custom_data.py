## make한 후 만들어낸 데이터셋에 대하여 해당 모든 좌표 값드릉ㄹ 주르륵 읽어오는 코드 

import h5py

# DB h5 경로
db_h5_path = "t_datasets/3131_datasets/test_database.h5"

def parse_coord(name):
    parts = name.split("@")
    row, col = int(parts[-2]), int(parts[-1])
    return row, col

with h5py.File(db_h5_path, "r") as f:
    db_names = [n.decode("utf-8") for n in f["image_name"][:]]

print(f"총 {len(db_names)}개 database 이미지 발견\n")

for i, name in enumerate(db_names):
    row, col = parse_coord(name)
    print(f"{i:4d}: {name}  →  row={row}, col={col}")
