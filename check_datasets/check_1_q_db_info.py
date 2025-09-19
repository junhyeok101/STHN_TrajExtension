## 실제 데이터셋의 지상이미지 위상이미지 데이터 구성을 output으로 내는 코드
## 어떤 구조인지 어떤 데이터들을 담고 있는지 한눈에 정리 

import h5py

query_h5_path = "t_datasets/3131_datasets/test_queries.h5"

with h5py.File(query_h5_path, "r") as f:
    print(query_h5_path)
    print("파일 안에 들어있는 key 목록:", list(f.keys()))

    # image_data shape 확인
    if "image_data" in f:
        print("image_data shape:", f["image_data"].shape)
        print("image_data dtype:", f["image_data"].dtype)

    # image_name shape 및 샘플 확인
    if "image_name" in f:
        print("image_name 개수:", len(f["image_name"]))
        print("첫 5개 이름 예시:", [name.decode("utf-8") for name in f["image_name"][:5]])


print("여기까지 queries")



db_h5_path = "t_datasets/3131_datasets/test_database.h5"

with h5py.File(db_h5_path, "r") as f:
    print("파일 안에 들어있는 key 목록:", list(f.keys()))

    # image_data shape 확인 (database에는 없을 가능성이 높음)
    if "image_data" in f:
        print("image_data shape:", f["image_data"].shape)
        print("image_data dtype:", f["image_data"].dtype)

    # image_name 확인
    if "image_name" in f:
        print("image_name 개수:", len(f["image_name"]))
        print("첫 5개 이름 예시:", [name.decode("utf-8") for name in f["image_name"][:5]])

    # image_size 확인
    if "image_size" in f:
        print("image_size shape:", f["image_size"].shape)
        print("첫 5개 사이즈 예시:", f["image_size"][:5].tolist())

print("여기까지 database")
