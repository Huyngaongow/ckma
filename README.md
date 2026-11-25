# ckma

- `cau1/` chứa mã nguồn, dữ liệu và tài liệu bổ trợ cho Câu 1.
- `cau2/` chứa mã nguồn, dữ liệu và kết quả thực nghiệm của Câu 2a.

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

## Chạy Câu 2a

Câu 2a dùng CNN viết bằng PyTorch để phân loại ảnh Fashion-MNIST. Thực nghiệm so sánh bốn cấu hình gồm CNN cơ sở, Dropout, tăng cường dữ liệu và kết hợp cả hai. Phần mở rộng dùng Random Search để tinh chỉnh cấu hình kết hợp. [Kết quả tham chiếu](cau2/outputs/summary.csv)

Dùng Python 3.10–3.13. Chạy từ thư mục gốc của repo.

```sh
cd cau2
python -m pip install -r requirements.txt
python main.py --data-dir du-lieu --offline --device cpu --threads 2 --seed 42 --deterministic --output-dir outputs_run
```

Sau khi cài thư viện, lệnh huấn luyện dùng dữ liệu Fashion-MNIST có sẵn trong repo. Kết quả lần chạy mới được ghi vào `cau2/outputs_run/`. Hướng dẫn chi tiết có cách tạo môi trường ảo, ví dụ suy luận Top 3 và lệnh chạy đối chứng FNN–CNN. [README Câu 2a](cau2/README.md)
