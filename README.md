ỨNG DỤNG TÌM KIẾM ẢNH TƯƠNG TỰ BẰNG PHƯƠNG PHÁP SO KHỚP HISTOGRAM

CS406 - HỒ PHẠM QUỐC BẢO - 24520152

DÙNG TRỰC TIẾP TRÊN STREAMLIT CLOUD: https://image-retrieval-using-histogram-ben.streamlit.app/

TÍNH NĂNG CHÍNH
- TÍNH HISTOGRAM ẢNH THEO
  + RGB (3 kênh màu)
  + HSV (kênh Hue và Saturation)
- SO KHỚP ẢNH BẰNG
  + Euclidean Distance
  + Manhattan Distance
  + Cosine Similarity
- HIỂN THỊ CÁC ẢNH GIỐNG VỚI ẢNH TRUY VẤN

CẤU TRÚC THƯ MỤC
- dataset
  + seg_img 
  + seg_test
  + seg_vec
- app.py
- preprocess.py
- requirements.txt

CÁCH CHẠY LOCAL
- Cài thư viện cần thiết: pip install -r requirements.txt
- Chạy ứng dụng: streamlit run app.py