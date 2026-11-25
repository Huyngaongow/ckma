# Minh họa suy luận Top 3 trên Fashion-MNIST

Hai lần chạy dùng CPU, seed 42, chế độ deterministic, 1 epoch và 8 lượt Random Search. Chương trình huấn luyện lại cấu hình thắng, rồi dự đoán ảnh Validation đầu tiên. Chỉ mục của ảnh trong tập Train chính thức là 47876 (đếm từ 0) ở cả hai lần.

## Trích log của lần chạy dùng trong báo cáo

Nguồn là log kiểm chứng đã lưu tại `/workspace/ckma-dev/rubric.9GKF0y/cau2.log` trong môi trường đám mây.

```text
Nhãn thật của ảnh Validation minh họa: T-shirt/top
Top 3 · Random Search
  T-shirt/top    91.83%
  Shirt          7.93%
  Pullover       0.17%
```

## Kiểm chứng với dữ liệu ngoại tuyến đi kèm

Từ thư mục `cau2/`, chạy lệnh sau bằng môi trường ảo đã cài các thư viện trong `requirements.txt`. Lệnh ghi kết quả vào một thư mục tạm mới.

```bash
demo_output=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 python main.py --data-dir du-lieu --offline --device cpu \
  --threads 2 --seed 42 --epochs 1 --trials 8 --deterministic --output-dir "$demo_output"
```

Kết quả quan sát được với Python 3.12.14, PyTorch 2.7.1+cpu và torchvision 0.22.1+cpu.

```text
Nhãn thật của ảnh Validation minh họa: T-shirt/top
Top 3 · Random Search
  T-shirt/top    92.07%
  Shirt          7.68%
  Pullover       0.17%
```

Hai lần chạy chọn cùng cấu hình Random Search và cho cùng ba nhãn theo thứ tự, với xác suất hiển thị khác nhau ở hai nhãn đầu. Các kết quả này minh họa dự đoán một ảnh sau 1 epoch. Bảng thực nghiệm 10 epoch được lưu riêng. [Kết quả 10 epoch](summary.csv)
