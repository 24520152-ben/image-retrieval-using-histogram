import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_rgb_hist(img, channels=[0, 1, 2], bins=[8, 8, 8], ranges=[0, 256, 0, 256, 0, 256], normalize=True):
    """3D RGB histogram -> flatten 1D vector"""

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([img_rgb], channels, None, bins, ranges).flatten()

    if normalize:
        total_pixels = hist.sum()
        if total_pixels > 0:
            hist /= total_pixels

    return hist

def compute_hsv_hist(img, channels=[0, 1], bins=[50, 60], ranges=[0, 180, 0, 256], normalize=True):
    """2D HSV histogram (Hue & Saturation) -> flatten 1D vector"""

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img_hsv], channels, None, bins, ranges).flatten()

    if normalize:
        total_pixels = hist.sum()
        if total_pixels > 0:
            hist /= total_pixels

    return hist

def save_csv(df, path, filename):
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, filename), index=False, header=False)
    print(f'Save to {os.path.join(path, filename)}.')

def build_hist_csv(folder_path, preprocessed_folder_path, output_rgb='seg_rgb.csv', output_hsv='seg_hsv.csv'):
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f'Found {len(image_paths)} images in {folder_path}.')

    rows_rgb = []
    rows_hsv = []

    for path in tqdm(image_paths, colour='cyan'):
        img = cv2.imread(path)
        if img is None:
            print(f'Skipping {path}, cannot read.')
            continue

        rgb_vec = compute_rgb_hist(img)
        hsv_vec = compute_hsv_hist(img)

        rows_rgb.append([path] + rgb_vec.tolist())
        rows_hsv.append([path] + hsv_vec.tolist())

    df_rgb = pd.DataFrame(rows_rgb)
    df_hsv = pd.DataFrame(rows_hsv)
    save_csv(df_rgb, preprocessed_folder_path, output_rgb)
    save_csv(df_hsv, preprocessed_folder_path, output_hsv)

def preprocess():
    dataset_name = r'dataset\seg_img'
    dataset_path = os.path.join(os.path.dirname(__file__), dataset_name)

    preprocessed_dataset_name = r'dataset\seg_vec'
    preprocessed_dataset_path = os.path.join(os.path.dirname(__file__), preprocessed_dataset_name)

    folders = os.listdir(dataset_path)
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        preprocessed_folder_path = os.path.join(preprocessed_dataset_path, folder)
        build_hist_csv(folder_path, preprocessed_folder_path)

if __name__ == '__main__':
    preprocess()