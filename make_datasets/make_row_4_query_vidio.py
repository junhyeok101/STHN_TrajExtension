# query 이미지들 -> vidio 제작해서 uav 가 지상 찍는 영상처럼 만들기 

import cv2
import os
import natsort

# 이미지 폴더 경로
image_dir = "t_datasets/query_image"
output_video = "t_datasets/vidio/uav_flight_row3131.mp4"

# 이미지 파일 정렬 (숫자 순서대로)
images = [f for f in os.listdir(image_dir) if f.endswith(".png")]
images = natsort.natsorted(images)  # 파일명 숫자 순 정렬

# 첫 번째 이미지로 크기 확인
first_img = cv2.imread(os.path.join(image_dir, images[0]))
h, w, c = first_img.shape

# 비디오 저장 설정 (fps=5면 1초에 5장, 느리게 보고 싶으면 fps 더 낮게)
fps = 5
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

for i, img_name in enumerate(images):
    img = cv2.imread(os.path.join(image_dir, img_name))
    out.write(img)

    if i % 10 == 0:
        print(f"{i}/{len(images)} frames written")

out.release()
print(f"UAV 열화상 비행 영상 저장 완료 → {output_video}")
