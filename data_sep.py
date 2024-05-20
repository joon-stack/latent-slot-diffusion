import os
import random
import shutil

# 원본 폴더 경로
source_dir = '/shared/s2/lab01/dataset/lsd/language_table/language_table-train-with-label/labels'  # 00000001 ~ 00008297 폴더들이 있는 경로

# 목적지 폴더 경로
train_dir = '/shared/s2/lab01/dataset/lsd/language_table/labels/train'
val_dir = '/shared/s2/lab01/dataset/lsd/language_table/labels/val'
test_dir = '/shared/s2/lab01/dataset/lsd/language_table/labels/test'

# 목적지 폴더가 없다면 생성
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 원본 폴더 목록 가져오기
files = [f for f in os.listdir(source_dir) if f.endswith('.npy')]
files.sort()

# files = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
# files.sort()

# 셔플링
random.seed(42)  # 재현성을 위해 고정된 시드 사용
random.shuffle(files)

# 비율 계산
total_files = len(files)
train_size = int(total_files * 0.8)
val_size = int(total_files * 0.1)
test_size = total_files - train_size - val_size

# 파일 분배
train_files = files[:train_size]
val_files = files[train_size:train_size + val_size]
test_files = files[train_size + val_size:]

# 파일 이동 함수
def move_files(file_list, destination):
    for file in file_list:
        shutil.move(os.path.join(source_dir, file), os.path.join(destination, file))

# 파일 이동
move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print("파일 분배 완료!")