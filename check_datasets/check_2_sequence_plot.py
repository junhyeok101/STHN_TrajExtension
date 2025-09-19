# 실제 데이터셋의 경로를 좌표로 표현해서 이미지 저장 

import h5py
import matplotlib.pyplot as plt

query_h5_path = "datasets/satellite_0_thermalmapping_135/val_queries.h5"

# h5에서 query_name 읽기
with h5py.File(query_h5_path, "r") as f:
    query_names = [n.decode("utf-8") for n in f["image_name"][:500]]

# @row@col → (row, col) 변환
def parse_coord(name):
    parts = name.split("@")
    return int(parts[-2]), int(parts[-1])

coords = [parse_coord(n) for n in query_names]

# ========================
# (1) Trajectory plot
# ========================
plt.figure(figsize=(8,8))
unique_rows = sorted(set(r for (r, _) in coords))
for r in unique_rows:
    row_points = [(rr, c) for (rr, c) in coords if rr == r]
    row_points = sorted(row_points, key=lambda x: x[1])
    cols = [c for (_, c) in row_points]
    rows = [r] * len(cols)
    plt.plot(cols, rows, "r-", markersize=2)

plt.gca().invert_yaxis()
plt.title("Query Trajectory (row blocks connected)")
plt.xlabel("col (x)")
plt.ylabel("row (y)")
plt.savefig("check_2_output/sequence_plot/135_train_val_query_trajectory.png", dpi=300, bbox_inches="tight")
plt.close()

print("플롯 저장 완료 → query_trajectory_by_row.png")

# ========================
# (2) 좌표값 터미널 출력
# ========================
print("\n=== 좌표 예시 (앞 20개) ===")

#범위 제한 가능 
for i, (r, c) in enumerate(coords[:500]):
    print(f"{i}: row={r}, col={c}")

print(f"\n총 query 개수: {len(coords)}")
print(f"row 범위: {min(r for r,c in coords)} ~ {max(r for r,c in coords)}")
print(f"col 범위: {min(c for r,c in coords)} ~ {max(c for r,c in coords)}")
