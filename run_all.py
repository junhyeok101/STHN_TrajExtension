import os
import shutil


# 실행 전에 t_outputs 초기화
if os.path.exists("t_outputs"):
    shutil.rmtree("t_outputs")   # t_outputs 폴더와 그 안의 모든 파일 삭제
os.makedirs("t_outputs", exist_ok=True)  # 빈 폴더 다시 생성

# 공통 옵션 (여기만 수정하면 세 스크립트 모두 동일하게 적용됨)
common_args = (
    "--datasets_folder t_datasets "
    "--dataset_name 3131_datasets "
    "--eval_model pretrained_models/1536_two_stages/STHN.pth "
    "--val_positive_dist_threshold 512 "
    "--lev0 "
    "--database_size 1536 "
    "--corr_level 4 "
    "--test "
    "--num_workers 2 "
    "--batch_size 1 "
    "--augment img "
    "--rotate_max 0 "
    "--resize_max 0 "
    "--perspective_max 30 "
)

# 실행할 스크립트 리스트
scripts = [
    "make_row_6_info.py",
    "local_pipeline/t_evaluate.py",
    "local_pipeline/t_evaluate_debug_plot.py",
    "local_pipeline/t_evaluate_log.py",
]

# 순차 실행
for script in scripts:
    print("=" * 40)
    print(f"Running: {script}")
    print("=" * 40)
    ret = os.system(f"python3 {script} {common_args}")
    if ret != 0:
        print(f"Error occurred while running {script}")
        break



"""    "--datasets_folder t_datasets "
    "--dataset_name 3131_datasets "
    "--eval_model pretrained_models/1536_two_stages/STHN.pth "
    "--val_positive_dist_threshold 512 "
    "--lev0 "
    "--database_size 1536 "
    "--corr_level 4 "
    "--test "
    "--num_workers 2 "
    "--batch_size 1 "
    "--augment img "
    "--rotate_max 0.2 "
    "--resize_max 0 "
    "--perspective_max 0"""