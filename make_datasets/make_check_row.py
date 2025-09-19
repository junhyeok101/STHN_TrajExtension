# 어떤 경로를 선택할지 확인 용 

import h5py
from collections import Counter

query_h5_path = "datasets/satellite_0_thermalmapping_135/train_queries.h5"

with h5py.File(query_h5_path, "r") as f:
    query_names = [n.decode("utf-8") for n in f["image_name"][:]]

def parse_coord(name):
    parts = name.split("@")
    return int(parts[-2]), int(parts[-1])  # (row, col)

# 모든 row 값 추출
rows = [parse_coord(q)[0] for q in query_names]

# 고유 row와 개수 세기
row_counter = Counter(rows)
unique_rows = sorted(row_counter.keys())

print("총 row 종류 수:", len(unique_rows))
print("row 값 목록:", unique_rows)
print("\n각 row별 query 개수:")
for r in unique_rows:
    print(f"row={r}: {row_counter[r]}개")
