import csv
import os
import shutil
import typing

train_elbow_folder = 'dataset/train/elbow'
train_finger_folder = 'dataset/train/finger'
train_forearm_folder = 'dataset/train/forearm'
train_hand_folder = 'dataset/train/hand'
train_humerus_folder = 'dataset/train/humerus'
train_shoulder_folder = 'dataset/train/shoulder'
train_wrist_folder = 'dataset/train/wrist'

valid_elbow_folder = 'dataset/valid/elbow'
valid_finger_folder = 'dataset/valid/finger'
valid_forearm_folder = 'dataset/valid/forearm'
valid_hand_folder = 'dataset/valid/hand'
valid_humerus_folder = 'dataset/valid/humerus'
valid_shoulder_folder = 'dataset/valid/shoulder'
valid_wrist_folder = 'dataset/valid/wrist'

image_count = 0

def copy_files(src: str , dest: str):
    global image_count
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest + '/' + str(image_count) + '.png')
            image_count += 1
        else:
            print("Copy failed :\(")

training_file_name = 'MURA-v1.1/train_labeled_studies.csv'
valid_file_name = 'MURA-v1.1/valid_labeled_studies.csv'

with open(training_file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    row_count = 1

    for row in csv_reader:
        if 'ELBOW' in row[0]:
            copy_files(row[0], train_elbow_folder)
        elif 'FINGER' in row[0]:
            copy_files(row[0], train_finger_folder)
        elif 'FOREARM' in row[0]:
            copy_files(row[0], train_forearm_folder)
        elif 'HAND' in row[0]:
            copy_files(row[0], train_hand_folder)
        elif 'HUMERUS' in row[0]:
            copy_files(row[0], train_humerus_folder)
        elif 'SHOULDER' in row[0]:
            copy_files(row[0], train_shoulder_folder)
        elif 'WRIST' in row[0]:
            copy_files(row[0], train_wrist_folder)
        else:
            print("Oops")

        print('Training: ', row_count)
        row_count += 1

with open(valid_file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    row_count = 1

    for row in csv_reader:
        if 'ELBOW' in row[0]:
            copy_files(row[0], valid_elbow_folder)
        elif 'FINGER' in row[0]:
            copy_files(row[0], valid_finger_folder)
        elif 'FOREARM' in row[0]:
            copy_files(row[0], valid_forearm_folder)
        elif 'HAND' in row[0]:
            copy_files(row[0], valid_hand_folder)
        elif 'HUMERUS' in row[0]:
            copy_files(row[0], valid_humerus_folder)
        elif 'SHOULDER' in row[0]:
            copy_files(row[0], valid_shoulder_folder)
        elif 'WRIST' in row[0]:
            copy_files(row[0], valid_wrist_folder)

        print('Valid: ', row_count)
        row_count += 1