import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st

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

def compute_distance(vec1, vec2, method):
    if method == 'Euclidean Distance':
        return np.linalg.norm(vec1 - vec2)
    elif method == 'Manhattan Distance':
        return np.sum(np.abs(vec1 - vec2))
    elif method == 'Cosine Similarity':
        numerator = np.dot(vec1, vec2)
        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10
        return 1 - (numerator / denominator)
    else:
        raise ValueError('Unknown method')

def find_similar_images(query_vec, csv_path, method, top_k=10):
    results = []
    df = pd.read_csv(csv_path, header=None)

    for _, row in df.iterrows():
        img_name = row.iloc[0]
        vec = row.iloc[1:].to_numpy(dtype=np.float32)
        dist = compute_distance(query_vec, vec, method)
        results.append((img_name, dist))

    results.sort(key=lambda x: x[1])
    return results[:top_k]

def build_UI():
    st.set_page_config(page_title='Image Retrieval via Histogram', layout='wide')
    st.title(body='Image Retrieval using Histogram Similarity', width='stretch')

    BASE_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
    TEST_DIR = os.path.join(BASE_DIR, 'seg_test')
    VEC_DIR = os.path.join(BASE_DIR, 'seg_vec')
    IMG_DIR = os.path.join(BASE_DIR, 'seg_img')

    # Image folder selection
    categories = os.listdir(TEST_DIR)
    category = st.sidebar.selectbox('Select category', categories)

    test_folder = os.path.join(TEST_DIR, category)
    vec_folder = os.path.join(VEC_DIR, category)
    img_folder = os.path.join(IMG_DIR, category)

    # Image selection
    test_images = [img for img in os.listdir(test_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected_img_name = st.sidebar.selectbox('Select image', test_images)
    selected_img_path = os.path.join(test_folder, selected_img_name)

    # Show query image
    selected_img = cv2.imread(selected_img_path)
    selected_img_rgb = cv2.cvtColor(selected_img, cv2.COLOR_BGR2RGB)
    st.sidebar.image(selected_img_rgb, caption='Query image', width='stretch')

    # Histogram and similarity metrics settings
    color_space = st.sidebar.radio('Color space', ['RGB', 'HSV'])
    method = st.sidebar.selectbox('Similarity method', ['Euclidean Distance', 'Manhattan Distance', 'Cosine Similarity'])
    top_k = st.sidebar.slider('Number of similar images', 1, 20, 10)

    # Compute histogram for query image
    query_img = cv2.imread(selected_img_path)
    if color_space == 'RGB':
        query_vec = compute_rgb_hist(query_img)
        csv_path = os.path.join(vec_folder, 'seg_rgb.csv')
    else:
        query_vec = compute_hsv_hist(query_img)
        csv_path = os.path.join(vec_folder, 'seg_hsv.csv')

    # Search similar images
    st.subheader(f'Top {top_k} similar images in {color_space} color space using {method} metric')
    results = find_similar_images(query_vec, csv_path, method, top_k)
    cols = st.columns(5)
    for idx, (img_name, dist) in enumerate(results):
        img = cv2.imread(os.path.join(img_folder, img_name))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        col = cols[idx % 5]
        with col:
            st.image(img_rgb, caption=f'{img_name} {method} = {dist:.4f}', width='content')

if __name__ == '__main__':
    build_UI()
