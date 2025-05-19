import os
import cv2
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import face_recognition_api


def _get_training_dirs(training_dir_path):
    return [x[0] for x in os.walk(training_dir_path)][1:]


def _get_training_labels(training_dir_path):
    return [x[1] for x in os.walk(training_dir_path)][0]


def _get_each_labels_files(training_dir_path):
    return [x[2] for x in os.walk(training_dir_path)][1:]


def _filter_image_files(training_dir_path):
    exts = {".jpg", ".jpeg", ".png"}
    training_folder_files_list = []
    for file_list in _get_each_labels_files(training_dir_path):
        images = [f for f in file_list if os.path.splitext(f)[1].lower() in exts]
        training_folder_files_list.append(images)
    return training_folder_files_list


def _zipped_folders_labels_images(training_dir_path, labels):
    return list(zip(_get_training_dirs(training_dir_path), labels, _filter_image_files(training_dir_path)))


def create_dataset(training_dir_path, labels):
    X = []
    y = []
    for i in _zipped_folders_labels_images(training_dir_path, labels):
        for fileName in i[2]:
            file_path = os.path.join(i[0], fileName)
            if not os.path.isfile(file_path):
                continue
            print(f"Đang xử lý: {file_path}")
            img = face_recognition_api.load_image_file(file_path)
            imgEncoding = face_recognition_api.face_encodings(img)
            if len(imgEncoding) == 0:
                continue
            X.append(imgEncoding[0])
            y.append(i[1])
    return np.array(X), np.array(y)


if __name__ == '__main__':
    encoding_file_path = './encoded-images-data.csv'
    training_dir_path = './training-images'
    labels_fName = 'labels.pkl'

    # Prepare labels
    labels = _get_training_labels(training_dir_path)
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)

    # Create dataset
    X, y = create_dataset(training_dir_path, labelsNum)
    df = pd.DataFrame(X)
    df['label'] = y

    # Backup existing CSV
    if os.path.isfile(encoding_file_path):
        print(f"{encoding_file_path} already exists. Backing up.")
        os.rename(encoding_file_path, f"{encoding_file_path}.bak")

    df.to_csv(encoding_file_path, index=False)

    print(f"{nClasses} classes created.")
    print(f"Saving labels pickle to '{labels_fName}'")
    with open(labels_fName, 'wb') as f:
        pickle.dump(le, f)
    print(f"Training Image's encodings saved in {encoding_file_path}")
