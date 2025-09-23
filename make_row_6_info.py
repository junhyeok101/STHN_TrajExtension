import os
import h5py

save_dir = "t_outputs"
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "dataset_info.txt"), "w", encoding="utf-8") as f:

    # Query (thermal)
    query_h5_path = "t_datasets/3131_datasets/test_queries.h5"
    with h5py.File(query_h5_path, "r") as qf:
        f.write("=== Query (Thermal) ===\n")
        f.write(f"File: {query_h5_path}\n")
        f.write(f"Keys: {list(qf.keys())}\n")

        if "image_data" in qf:
            f.write(f"image_data shape: {qf['image_data'].shape}\n")
            f.write(f"image_data dtype: {qf['image_data'].dtype}\n")

        if "image_name" in qf:
            f.write(f"image_name 개수: {len(qf['image_name'])}\n")
            f.write(f"첫 5개 이름 예시: {[n.decode('utf-8') for n in qf['image_name'][:5]]}\n")

        f.write("\n- Thermal 이미지는 512×512 patch\n")
        f.write("- 해상도: 1 m/px\n")
        f.write("- 실제 커버 범위: 512 m × 512 m\n\n")

    # Database (satellite)
    db_h5_path = "t_datasets/3131_datasets/test_database.h5"
    with h5py.File(db_h5_path, "r") as df:
        f.write("=== Database (Satellite) ===\n")
        f.write(f"File: {db_h5_path}\n")
        f.write(f"Keys: {list(df.keys())}\n")

        if "image_name" in df:
            f.write(f"image_name 개수: {len(df['image_name'])}\n")
            f.write(f"첫 5개 이름 예시: {[n.decode('utf-8') for n in df['image_name'][:5]]}\n")

        if "image_size" in df:
            f.write(f"image_size shape: {df['image_size'].shape}\n")
            f.write(f"첫 5개 사이즈 예시: {df['image_size'][:5].tolist()}\n")

        f.write("\n- Satellite 이미지는 WS×WS 크기로 crop 후 사용\n")
        f.write("  ((thermal 512×512과 영역 매칭, 단 실제 m/px은 WS에 따라 달라짐))\n")
        f.write("  • WS = 512  → 1 px = 1.0 m → 실제 커버 512 m\n")
        f.write("  • WS = 1024  → 실제 커버 1024 m → Resize(256) 후 1 px = 4 m\n")
        f.write("  • WS = 1536  → 실제 커버 1536 m → Resize(256) 후 1 px = 6 m\n\n")

    f.write("저장 완료!\n")
