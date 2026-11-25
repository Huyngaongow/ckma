# ckma

## Chuẩn bị

Dùng Python 3.10–3.13. Chạy các lệnh từ thư mục gốc của repo.

```sh
python -m venv .venv
```

Kích hoạt môi trường bằng `source .venv/bin/activate` trên Linux hoặc macOS, hoặc `.\.venv\Scripts\Activate.ps1` trên Windows PowerShell.

## Câu 1 — FNN dự đoán cường độ nén bê tông

```sh
python -m pip install numpy pandas matplotlib scikit-learn
python cau1/fnn_concrete.py --data cau1/data/concrete.csv --five-seeds --output cau1/fnn_results
```

Lệnh kiểm tra gradient và so sánh bốn cấu hình FNN trên năm seed. Bảng MSE, RMSE và MAE được in ra terminal. Ba biểu đồ của seed 42 được lưu trong `cau1/fnn_results/`.

Thư mục `cau1/data/` kèm `concrete.csv` và `Concrete_Data.xls`. Để tạo lại CSV từ XLS, xuất UTF-8 có hàng tiêu đề, dùng dấu phẩy phân cách cột, dấu chấm thập phân và giữ cường độ nén ở cột cuối. [Dữ liệu UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) · [Nguồn gốc dữ liệu và kết quả Câu 1](cau1/report/cau1.md)

## Câu 2a — CNN phân loại ảnh Fashion-MNIST

```sh
python -m pip install -r cau2/requirements.txt
python cau2/main.py --data-dir cau2/du-lieu --offline --device cpu --threads 2 --seed 42 --epochs 10 --trials 8 --deterministic --output-dir cau2/outputs_run
```

Lệnh so sánh CNN cơ sở, Dropout, tăng cường dữ liệu và cấu hình kết hợp, sau đó tinh chỉnh cấu hình kết hợp bằng Random Search. Bốn tệp `.gz` trong `cau2/du-lieu/FashionMNIST/raw/` được tự giải nén khi chạy. Sau khi cài thư viện, chương trình huấn luyện ngoại tuyến và ghi kết quả mới vào `cau2/outputs_run/`. [Dữ liệu gốc](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion) · [Giấy phép dữ liệu](cau2/du-lieu/LICENSE)

[Kết quả tham chiếu](cau2/outputs/summary.csv) · [Minh họa Top 3 và lệnh chạy lại](cau2/outputs/demo_top3.md)

### Đối chứng FNN–CNN

```sh
python cau2/fnn_baseline.py --data-dir cau2/du-lieu --offline --seed 42 --epochs 10 --threads 2 --output-dir cau2/outputs_run/fnn_comparison
```

Lệnh đo độ chính xác, số tham số và thời gian huấn luyện trên cùng phép chia Fashion-MNIST. [Kết quả đối chứng tham chiếu](cau2/outputs/fnn_comparison.csv)
