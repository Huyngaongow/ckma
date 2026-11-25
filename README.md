# ckma

- `cau1/` chứa mã nguồn, dữ liệu và tài liệu bổ trợ cho Câu 1.
- [`cau2/`](cau2/README.md) chứa Câu 2a, hướng dẫn chạy ngoại tuyến và thí nghiệm đối chứng FNN–CNN.

## Chạy Câu 1

Dùng Python 3.9 trở lên. Cài thư viện bằng `python -m pip install numpy pandas matplotlib scikit-learn`.

Từ thư mục gốc của repo, chạy:

```sh
cd cau1
python fnn_concrete.py --data data/concrete.csv --five-seeds
```

Bảng kết quả theo từng seed và bảng trung bình ± độ lệch chuẩn mẫu được in ra terminal. Ba biểu đồ của seed 42 được lưu trong `cau1/fnn_results/`. Dùng `--output` để đổi thư mục lưu.

### Dữ liệu

`cau1/data/` chứa tệp `Concrete_Data.xls` và bản CSV chuyển từ tệp này. CSV có hàng tiêu đề, giữ thứ tự tám cột đầu vào và để cường độ nén ở cột cuối. Chương trình đọc tệp cục bộ bằng pandas. [Bộ Concrete Compressive Strength của UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)

Để tự lấy dữ liệu từ kho UCI, tải `Concrete_Data.xls` từ trang trên và lưu vào `cau1/data/`. Mở tệp bằng Excel hoặc LibreOffice, xuất trang tính thành CSV UTF-8 có tên `concrete.csv`. Dùng dấu phẩy phân cách cột, dấu chấm thập phân và giữ đủ độ chính xác của các cột số.

Xem nguồn gốc bản XLS trong repo, số dòng trùng, công thức và kết quả tại [báo cáo Câu 1](cau1/report/cau1.md).
