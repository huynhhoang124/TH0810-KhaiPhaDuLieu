# TH0810 – Khai phá dữ liệu: KNN, K-Means, Naive Bayes, Decision Tree

**Sinh viên:** Hoàng Văn Huynh – B23DCCC086

Áp dụng KNN, K-Means, Naive Bayes và Decision Tree, kèm phân tích thống kê, trên 2 dataset UCI:

- **#697** [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) – phân lớp 3 lớp Dropout / Enrolled / Graduate.
- **#235** [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) – phân lớp mức tải theo giờ (nhãn tự tạo) và phân cụm kiểu ngày tiêu thụ điện.

📄 **Báo cáo:** [BaoCao_TH0810.pdf](BaoCao_TH0810.pdf)

## Kết quả (tập test)

| Dataset | KNN | Naive Bayes | Decision Tree | K-Means |
|---|---|---|---|---|
| #697 (Accuracy / macro-F1) | 0.720 / 0.621 | 0.689 / 0.614 | **0.745 / 0.655** | k=3, có cụm 82% Dropout |
| #235 (Accuracy / macro-F1) | 0.635 / 0.642 | 0.572 / 0.575 | **0.668 / 0.675** | k=4 kiểu ngày tiêu thụ |

## Cấu trúc

| File | Nội dung |
|---|---|
| `BaoCao_TH0810.pdf` | Báo cáo |
| `TH0810_KhaiPhaDuLieu.ipynb` | Notebook chính: code + kết quả + nhận xét |
| `TH0810_KhaiPhaDuLieu.py` | Toàn bộ code dạng script (xuất từ notebook) |
| `xuat_bao_cao.py` | Chụp màn hình output notebook và dựng báo cáo PDF |

## Chạy lại

```bash
pip install pandas numpy scikit-learn matplotlib jupyter playwright
# tải dữ liệu
mkdir -p data/697 figures
curl -L -o data/697/data.csv "https://archive.ics.uci.edu/static/public/697/data.csv"
curl -L -o power.zip "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
mkdir -p data/235 && unzip power.zip -d data/235 && rm power.zip
# chạy notebook, rồi xuất PDF (cần Google Chrome)
jupyter nbconvert --to notebook --execute --inplace TH0810_KhaiPhaDuLieu.ipynb
python xuat_bao_cao.py TH0810_KhaiPhaDuLieu.ipynb
```
