# Câu 2a · CNN phân loại ảnh Fashion-MNIST

## Cài đặt và chạy

Dùng Python 3.10–3.13. Từ thư mục chứa `main.py`, tạo môi trường ảo và chạy:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --deterministic
```

Trên Windows PowerShell, dùng `.venv\Scripts\Activate.ps1` để kích hoạt môi trường. Trên Linux hoặc Windows chỉ có CPU, có thể cài cặp PyTorch CPU trước lệnh cài requirements để tải ít hơn:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

Mặc định chương trình ưu tiên CUDA, sau đó đến MPS trên macOS và cuối cùng là CPU. Có thể buộc dùng CPU với `--device cpu` và chỉnh số luồng với `--threads`. Với `--deterministic`, hãy giữ nguyên seed, thiết bị, phiên bản thư viện và số luồng khi tái lập. Phần kiểm chứng được thực hiện trên CPU.

Chương trình ưu tiên dữ liệu cục bộ và tải từ mirror HTTPS của Fashion-MNIST nếu thiếu. Khi máy chạy không có mạng, chép bốn tệp [Fashion-MNIST chính thức](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion) vào cấu trúc sau. Có thể dùng tệp `.gz` hoặc tệp IDX đã giải nén có cùng tên bỏ đuôi `.gz`.

```text
du-lieu/
└── FashionMNIST/
    └── raw/
        ├── train-images-idx3-ubyte.gz
        ├── train-labels-idx1-ubyte.gz
        ├── t10k-images-idx3-ubyte.gz
        └── t10k-labels-idx1-ubyte.gz
```

```bash
python main.py --data-dir ./du-lieu --offline --deterministic
```

Chạy riêng một cấu hình để kiểm tra công tắc. Tham số `--image` nhận một ảnh JPEG/PNG bất kỳ và dùng mô hình vừa huấn luyện để in top 3. Ảnh nên có bố cục giống Fashion-MNIST, với một vật thể ở giữa trên nền tối.

```bash
python main.py --single --dropout --no-augmentation --epochs 1 --deterministic --output-dir outputs/quick
python main.py --deterministic --image anh_minh_hoa.png
```

## Thiết kế thí nghiệm

- **Dữ liệu.** Chia phân tầng 5.000 ảnh Train và 5.000 ảnh Validation từ tập Train chính thức. Bỏ 50.000 ảnh còn lại. Giữ nguyên 10.000 ảnh Test chính thức. Lưu chỉ mục chia tập bằng seed mặc định 42. Tính trung bình và độ lệch chuẩn của pixel Train ở thang `[0, 1]`, rồi áp dụng cùng công thức chuẩn hóa cho mọi tập.
- **Bốn cấu hình chính.** So sánh cơ sở, Dropout, Data Augmentation và kết hợp cả hai. Dùng cùng kiến trúc, seed, phép chia, batch size, learning rate và 10 epoch. Dropout dùng xác suất 0,4 sau lớp ẩn FC. Data Augmentation kết hợp lật ngang, xoay ngẫu nhiên tối đa 10° và dịch tối đa 8% mỗi chiều, chỉ áp dụng khi cập nhật trọng số. Mỗi epoch đo Train và Validation trong chế độ đánh giá trên ảnh sạch để hai đường cong có thể so sánh trực tiếp.
- **Random Search.** Lấy mẫu không lặp 8 cấu hình từ tích Descartes của 3 learning rate, 3 dropout rate, 3 số filter và 3 batch size. Giữ cả hai kỹ thuật bật trong các lượt thử. Xếp theo accuracy Validation ở epoch cuối, rồi theo loss Validation thấp hơn nếu bằng điểm. Huấn luyện lại cấu hình thắng từ khởi tạo mới, với cùng Train, seed và ngân sách epoch. Tham số `--trials` nhận từ 8 đến 12.
- **Test.** Sau khi chốt tất cả mô hình, mỗi mô hình cuối duyệt Test một lượt. Tính loss trung bình theo ảnh và precision, recall, F1 trung bình đều trên 10 lớp. Ma trận nhầm lẫn có hàng là nhãn thật và cột là nhãn dự đoán.
- **Đọc kết quả.** Trong `learning_curves.png`, xem lúc Train loss giảm nhưng Validation loss tăng hoặc khoảng cách accuracy giữa Train và Validation rộng ra. Đọc `accuracy_gap` cùng Validation accuracy và các chỉ số Test để phân biệt cải thiện tổng quát hóa với trường hợp mô hình học yếu đi. So sánh dòng `both` với `random_search` trong bảng tổng hợp để xem phần thay đổi do tinh chỉnh khi giữ nguyên hai kỹ thuật.

## Kết quả lần chạy kèm theo

Chạy CPU với 2 luồng, seed 42, 10 epoch và 8 lượt Random Search. Tổng thời gian 507.8 giây với Fashion-MNIST đã có cục bộ.

| Cấu hình | Train acc. | Val acc. | Chênh lệch (điểm %) | Test acc. | Test loss | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Cơ sở | 93.58% | 87.54% | 6.04 | 87.13% | 0.3722 | 0.8689 |
| Dropout | 90.54% | 86.90% | 3.64 | 86.16% | 0.3847 | 0.8587 |
| Augmentation | 85.00% | 83.78% | 1.22 | 83.26% | 0.4404 | 0.8248 |
| Dropout + Augmentation | 82.44% | 81.24% | 1.20 | 80.94% | 0.4893 | 0.7958 |
| Random Search | 84.76% | 83.42% | 1.34 | 83.08% | 0.4603 | 0.8229 |

Cấu hình thắng trên Validation dùng learning rate 0.002, dropout rate 0.4, 32 filter ở khối đầu và batch size 64. Xem macro precision và macro recall trong `outputs/summary.csv`.

## Tệp chương trình và đầu ra

| Tệp nguồn | Vai trò |
|---|---|
| `data.py` | Tải hoặc đọc dữ liệu, chia tập, chuẩn hóa và augmentation |
| `model.py` | Kiến trúc CNN, kích thước tensor và siêu tham số cơ sở |
| `train.py` | Seed, deterministic, vòng lặp huấn luyện và đo theo epoch |
| `evaluate.py` | Đánh giá Test, biểu đồ và suy luận top 3 |
| `extension.py` | Không gian siêu tham số và Random Search |
| `main.py` | CLI, điều phối thí nghiệm và xuất kết quả |

| Tệp sinh trong `outputs/` | Nội dung |
|---|---|
| `summary.csv` | Năm mô hình, siêu tham số, kết quả cuối và khoảng cách accuracy |
| `history.csv`, `learning_curves.png` | Lịch sử theo epoch và hình so sánh bốn cấu hình chính |
| `confusion_*.png`, `errors_*.png` | Ma trận nhầm lẫn và tối đa 12 lỗi phân loại cho mỗi mô hình |
| `random_search_trials.csv` | Siêu tham số, kết quả Validation và cờ chọn của 8–12 lượt thử |
| `extension_comparison.csv`, `extension_comparison.png` | So sánh trực tiếp mô hình cơ sở và mô hình mở rộng |
| `split_indices.npz` | Chỉ mục Train và Validation trong tập Train chính thức |

## Lý thuyết ngắn

**Tích chập.** Với đầu vào đã đệm biên, PyTorch tính tương quan chéo theo công thức
$y_{o,i,j}=b_o+\sum_c\sum_{u,v}W_{o,c,u,v}x_{c,is+u,js+v}$.
Bộ lọc $o$ cộng các tích trên toàn bộ kênh đầu vào và cửa sổ $k\times k$. Với dilation bằng 1, mỗi chiều không gian đầu ra là $\lfloor(n+2p-k)/s\rfloor+1$. Trong bài này $k=3,p=1,s=1$, nên tích chập giữ nguyên chiều cao và chiều rộng. ReLU áp dụng $\max(0,x)$ theo từng phần tử.

**Max pooling.** $y_{c,i,j}=\max_{0\le u,v<2}x_{c,2i+u,2j+v}$. Cửa sổ $2\times2$ với stride 2 lấy giá trị lớn nhất trong mỗi vùng và giảm một nửa mỗi chiều ảnh ở hai kích thước đang dùng.

**Softmax và cross-entropy.** Từ 10 logit $z_j$, xác suất lớp $j$ là $p_j=e^{z_j}/\sum_k e^{z_k}$. Với nhãn đúng $t$, cross-entropy của một ảnh là $-\log p_t=\operatorname{logsumexp}(z)-z_t$. PyTorch `CrossEntropyLoss` nhận trực tiếp logit và chỉ mục lớp. Hàm suy luận mới tính softmax để hiển thị xác suất top 3.

| Tầng | Kích thước với batch B và số filter f = 24 |
|---|---|
| Ảnh xám đầu vào | B × 1 × 28 × 28 |
| Conv 3×3, pad 1 → ReLU | B × 24 × 28 × 28 sau mỗi phép |
| MaxPool 2×2 | B × 24 × 14 × 14 |
| Conv 3×3, pad 1 → ReLU | B × 48 × 14 × 14 sau mỗi phép |
| MaxPool 2×2 | B × 48 × 7 × 7 |
| Flatten | B × 2.352 |
| FC → ReLU → Dropout hoặc Identity | B × 128 sau mỗi phép |
| FC đầu ra | B × 10 logit |

## Danh sách tự kiểm tra

- [x] 1. Tập Train nhỏ, chia phân tầng theo seed, Test cố định, thống kê chuẩn hóa chỉ từ Train.
- [x] 2. CNN hai khối Conv–ReLU–Pool, một FC ẩn, mười logit, kích thước ghi bên cạnh các lớp.
- [x] 3. Hai kỹ thuật Dropout và Data Augmentation có cờ độc lập.
- [x] 4. Bốn cấu hình chính chạy cùng seed, số epoch và phép chia dữ liệu.
- [x] 5. Mỗi mô hình cuối duyệt Test một lượt để tính loss, accuracy, macro precision/recall/F1 và confusion matrix.
- [x] 6. PNG đường cong Train/Validation, confusion matrix và ảnh dự đoán sai.
- [x] 7. Random Search 8–12 lượt dùng Validation, huấn luyện lại cấu hình thắng, so sánh với cơ sở bằng bảng và PNG.
- [x] 8. Hàm dự đoán một ảnh và in top 3 xác suất.
- [x] Cấu trúc sáu mô-đun, siêu tham số đầu tệp, requirements, hướng dẫn chạy, seed và deterministic.
- [x] Kết quả tạo từ dữ liệu thật bằng chương trình, in bảng pandas và lưu CSV.
