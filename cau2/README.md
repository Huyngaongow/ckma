# Câu 2a · CNN phân loại ảnh Fashion-MNIST

## Cài đặt và chạy

Dùng Python 3.10–3.13. Từ thư mục chứa `main.py`, tạo môi trường ảo và chạy:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --data-dir du-lieu --offline --device cpu --threads 2 --seed 42 --deterministic --output-dir outputs_run
```

Trên Windows PowerShell, dùng `.venv\Scripts\Activate.ps1` để kích hoạt môi trường. Trên Linux hoặc Windows chỉ có CPU, có thể cài cặp PyTorch CPU trước lệnh cài requirements để tải ít hơn:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

Mặc định chương trình ưu tiên CUDA, sau đó đến MPS trên macOS và cuối cùng là CPU. Có thể buộc dùng CPU với `--device cpu` và chỉnh số luồng với `--threads`. Với `--deterministic`, hãy giữ nguyên seed, thiết bị, phiên bản thư viện và số luồng khi tái lập. Phần kiểm chứng được thực hiện trên CPU.

Repo kèm bốn tệp `.gz` chính thức của Fashion-MNIST để chạy ngoại tuyến sau khi đã cài thư viện. Chương trình tự giải nén vào thư mục `raw/` khi cần. Dung lượng bốn tệp nén khoảng 31 MB. [Dữ liệu gốc](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion) · [Giấy phép MIT đi kèm](du-lieu/LICENSE)

```text
du-lieu/
├── LICENSE
└── FashionMNIST/
    └── raw/
        ├── train-images-idx3-ubyte.gz
        ├── train-labels-idx1-ubyte.gz
        ├── t10k-images-idx3-ubyte.gz
        └── t10k-labels-idx1-ubyte.gz
```

Chạy riêng một cấu hình để kiểm tra công tắc. Tham số `--image` nhận một ảnh JPEG/PNG bất kỳ và dùng mô hình vừa huấn luyện để in top 3. Ảnh nên có bố cục giống Fashion-MNIST, với một vật thể ở giữa trên nền tối.

```bash
python main.py --data-dir du-lieu --offline --single --dropout --no-augmentation --epochs 1 --deterministic --output-dir outputs/quick
python main.py --data-dir du-lieu --offline --deterministic --image anh_minh_hoa.png --output-dir outputs_run/image
```

Để khảo sát ảnh hưởng của phép chia dữ liệu và ngân sách huấn luyện, chạy lại toàn bộ bốn cấu hình cùng Random Search với mỗi seed trong một thư mục kết quả riêng.

```bash
python main.py --data-dir ./du-lieu --offline --device cpu --threads 2 --deterministic \
  --seeds 42 7 13 --epochs 30 --output-dir outputs/multi_seed_30
```

Lệnh này lưu kết quả từng seed trong `seed-42/`, `seed-7/` và `seed-13/` dưới thư mục đầu ra. Tệp `aggregate.csv` ghi số seed, trung bình và độ lệch chuẩn mẫu của từng chỉ số theo cấu hình. Mọi lượt dùng chung tập Test chính thức. Độ lệch chuẩn mô tả dao động giữa các lần chạy khi đổi seed và phép chia Train/Validation.

## Thiết kế thí nghiệm

- **Dữ liệu.** Chia phân tầng 5.000 ảnh Train và 5.000 ảnh Validation từ tập Train chính thức. Bỏ 50.000 ảnh còn lại. Giữ nguyên 10.000 ảnh Test chính thức. Lưu chỉ mục chia tập bằng seed mặc định 42. Tính trung bình và độ lệch chuẩn của pixel Train ở thang `[0, 1]`, rồi áp dụng cùng công thức chuẩn hóa cho mọi tập.
- **Bốn cấu hình chính.** So sánh cơ sở, Dropout, Data Augmentation và kết hợp cả hai. Trong mỗi seed, giữ chung kiến trúc các lớp học tham số, phép chia, batch size, learning rate và số epoch (mặc định 10). Mỗi epoch đo Train và Validation trong chế độ đánh giá trên ảnh sạch để hai đường cong có thể so sánh trực tiếp.
- **Dropout.** Khi huấn luyện, đặt từng kích hoạt sau lớp ẩn FC về 0 với xác suất 0,4 và nhân các kích hoạt được giữ với hệ số $1/(1-0.4)$. Cơ chế này giữ kỳ vọng của kích hoạt và giảm sự phụ thuộc vào một số nơ-ron. Khi đánh giá, Dropout giữ nguyên đầu vào.
- **Data Augmentation.** Khi cập nhật trọng số, ảnh có thể được lật ngang, xoay ngẫu nhiên tối đa 10° và dịch tối đa 8% mỗi chiều. Các biến đổi giữ nhãn gốc và giúp mô hình gặp nhiều bố cục của cùng một lớp. Tập đo Train, Validation và Test đều dùng ảnh sạch.
- **Random Search.** Lấy mẫu không lặp 8 cấu hình từ tích Descartes của 3 learning rate, 3 dropout rate, 3 số filter và 3 batch size. Giữ cả hai kỹ thuật bật trong các lượt thử. Xếp theo accuracy Validation ở epoch cuối, rồi theo loss Validation thấp hơn nếu bằng điểm. Huấn luyện lại cấu hình thắng từ khởi tạo mới, với cùng Train, seed và ngân sách epoch. Tham số `--trials` nhận từ 8 đến 12.
- **Test.** Sau khi chốt tất cả mô hình, mỗi mô hình cuối duyệt Test một lượt. Tính loss trung bình theo ảnh và precision, recall, F1 trung bình đều trên 10 lớp. Ma trận nhầm lẫn có hàng là nhãn thật và cột là nhãn dự đoán.

## Kết quả tham chiếu với 10 epoch

Bảng lấy từ lần chạy kèm theo trong `outputs/` với CPU 2 luồng, seed 42, 10 epoch và 8 lượt Random Search. Tổng thời gian 507,8 giây với Fashion-MNIST đã có cục bộ.

| Cấu hình | Train acc. | Val acc. | Chênh lệch (điểm %) | Test acc. | Test loss | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Cơ sở | 93.58% | 87.54% | 6.04 | 87.13% | 0.3722 | 0.8689 |
| Dropout | 90.54% | 86.90% | 3.64 | 86.16% | 0.3847 | 0.8587 |
| Augmentation | 85.00% | 83.78% | 1.22 | 83.26% | 0.4404 | 0.8248 |
| Dropout + Augmentation | 82.44% | 81.24% | 1.20 | 80.94% | 0.4893 | 0.7958 |
| Random Search | 84.76% | 83.42% | 1.34 | 83.08% | 0.4603 | 0.8229 |

### Ảnh hưởng của Dropout và Data Augmentation

Khoảng cách accuracy Train–Validation của CNN cơ sở tăng từ −0,20 điểm phần trăm ở epoch 1 lên 6,04 ở epoch 10. Trong cùng giai đoạn, Validation Loss giảm từ 0,6335 xuống giá trị thấp nhất trong lần chạy là 0,3619. Điều này cho thấy mô hình cải thiện trên cả hai tập, với accuracy tăng nhanh hơn trên Train.

Dropout, Augmentation và cấu hình kết hợp thu hẹp chênh lệch ở epoch cuối xuống lần lượt 3,64, 1,22 và 1,20 điểm phần trăm. Accuracy Test giảm tương ứng 0,97, 3,87 và 6,19 điểm phần trăm so với cơ sở. Cả ba cấu hình cũng có Test Loss cao hơn và Macro F1 thấp hơn. Trong lần chạy này, chênh lệch nhỏ hơn đi cùng mức accuracy thấp hơn trên cả Train và Test. [Đường cong học](outputs/learning_curves.png) · [Kết quả Test](outputs/summary.csv)

Mức thay đổi khác nhau theo lớp. Trong 1.000 ảnh Test có nhãn thật `Shirt`, CNN cơ sở nhận đúng 564 ảnh và đoán nhầm 106 ảnh thành `Pullover`. Khi kết hợp hai kỹ thuật, số ảnh nhận đúng giảm còn 198 và số ảnh nhầm sang `Pullover` tăng lên 336. Với lớp `Ankle boot`, số ảnh nhận đúng tăng từ 911 lên 940. [Ma trận cơ sở](outputs/confusion_baseline.png) · [Ma trận cấu hình kết hợp](outputs/confusion_both.png)

### Hiệu quả của Random Search

Cấu hình thắng trên Validation dùng learning rate 0,002, dropout rate 0,4, 32 bộ lọc ở khối đầu và batch size 64. Giữ cả hai kỹ thuật, tinh chỉnh nâng accuracy Test từ 80,94% lên 83,08% (2,14 điểm phần trăm) và giảm Test Loss từ 0,4893 xuống 0,4603. So với CNN cơ sở, mô hình tinh chỉnh có khoảng cách Train–Validation nhỏ hơn (1,34 so với 6,04 điểm phần trăm), accuracy Test thấp hơn 4,05 điểm phần trăm và Test Loss cao hơn. Kết quả mô tả một seed với cùng ngân sách 10 epoch cho mỗi lần huấn luyện. [Thứ hạng trên Validation](outputs/random_search_trials.csv) · [Kết quả Test](outputs/summary.csv)

## Tệp chương trình và đầu ra

| Tệp nguồn | Vai trò |
|---|---|
| `data.py` | Tải hoặc đọc dữ liệu, chia tập, chuẩn hóa và augmentation |
| `model.py` | Kiến trúc CNN, kích thước tensor và siêu tham số cơ sở |
| `train.py` | Seed, deterministic, vòng lặp huấn luyện và đo theo epoch |
| `evaluate.py` | Đánh giá Test, biểu đồ và suy luận top 3 |
| `extension.py` | Không gian siêu tham số và Random Search |
| `main.py` | CLI, điều phối thí nghiệm và xuất kết quả |
| `fnn_baseline.py` | Tự cài FNN bằng NumPy và đo đối chứng với CNN cơ sở |

| Tệp trong `outputs/` | Nội dung |
|---|---|
| `summary.csv` | Năm mô hình, siêu tham số, kết quả cuối và khoảng cách accuracy |
| `history.csv`, `learning_curves.png` | Lịch sử theo epoch và hình so sánh bốn cấu hình chính |
| `confusion_*.png`, `errors_*.png` | Ma trận nhầm lẫn và tối đa 12 lỗi phân loại cho mỗi mô hình |
| `random_search_trials.csv` | Siêu tham số, kết quả Validation và cờ chọn của 8–12 lượt thử |
| `split_indices.npz` | Chỉ mục Train và Validation trong tập Train chính thức |
| `fnn_comparison.csv` | Kết quả tham chiếu của phép đo đối chứng FNN–CNN |
| `demo_top3.md` | Lệnh tái lập và trích log suy luận một ảnh sau 1 epoch |

Ba mốc đối chiếu trong `summary.csv` gồm CNN cơ sở, cấu hình kết hợp trước tinh chỉnh và cấu hình sau Random Search. Khi chạy toàn bộ thí nghiệm, chương trình tạo thêm `extension_comparison.csv` và `extension_comparison.png` từ ba mốc này trong thư mục đầu ra.

## Kiến trúc và nguyên lý CNN

**Tích chập.** Với đầu vào đã đệm biên, PyTorch tính tương quan chéo theo công thức
$y_{o,i,j}=b_o+\sum_c\sum_{u,v}W_{o,c,u,v}x_{c,is+u,js+v}$.
Bộ lọc $o$ cộng các tích trên toàn bộ kênh đầu vào và cửa sổ $k\times k$. Với dilation bằng 1, mỗi chiều không gian đầu ra là $\lfloor(n+2p-k)/s\rfloor+1$. Trong bài này $k=3,p=1,s=1$, nên tích chập giữ nguyên chiều cao và chiều rộng. ReLU áp dụng $\max(0,x)$ theo từng phần tử.

**Max pooling.** $y_{c,i,j}=\max_{0\le u,v<2}x_{c,2i+u,2j+v}$. Cửa sổ $2\times2$ với stride 2 lấy giá trị lớn nhất trong mỗi vùng và giảm một nửa mỗi chiều ảnh ở hai kích thước đang dùng.

**Flatten và lớp kết nối đầy đủ.** Flatten xếp các giá trị trên bản đồ đặc trưng thành véc-tơ $h$. Mỗi nơ-ron của lớp kết nối đầy đủ nhận tất cả phần tử của véc-tơ, với đầu ra $z_j=b_j+\sum_i W_{ji}h_i$. Lớp FC ẩn tổng hợp đặc trưng của toàn bộ ảnh rồi áp dụng ReLU. Lớp FC cuối tạo một logit cho mỗi lớp.

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

### Ví dụ huấn luyện và suy luận với Fashion-MNIST

Một mini-batch huấn luyện gồm 64 ảnh xám $28\times28$ và 64 nhãn nguyên. Nhãn `T-shirt/top` có chỉ mục 0. Chuẩn hóa ảnh bằng thống kê Train rồi đưa qua hai khối Conv–ReLU–MaxPool và các lớp FC. Đầu ra là ma trận $64\times10$. Loss mini-batch là trung bình cross-entropy theo nhãn thật. Chẳng hạn, một ảnh `T-shirt/top` có xác suất softmax lớp 0 bằng 0,8 sẽ đóng góp $-\ln(0.8)\approx0.2231$ vào tổng loss. Đạo hàm theo logit lớp $j$ của ảnh, trước khi lấy trung bình batch, là $p_j-\mathbf{1}[j=t]$. PyTorch lan truyền gradient qua lớp FC, qua vị trí cực đại của pooling, qua ReLU và cộng đóng góp từ các vị trí dùng chung bộ lọc. Adam dùng gradient để cập nhật tham số.

Khi suy luận, chuẩn hóa một ảnh theo cùng thống kê, chạy mạng ở chế độ đánh giá và xếp 10 xác suất softmax để lấy top 3. Trong một lần chạy kiểm tra riêng với một epoch, ảnh Validation có nhãn `T-shirt/top` được mô hình Random Search gán xác suất 91,83% cho `T-shirt/top`, 7,93% cho `Shirt` và 0,17% cho `Pullover`. Dự đoán top 1 là `T-shirt/top`. [Log minh họa và kết quả chạy lại](outputs/demo_top3.md)

Chạy lại quy trình Random Search một epoch từ thư mục `cau2` bằng lệnh sau. Chương trình dùng mô hình thắng để in top 3 trên ảnh Validation minh họa. Xác suất có thể chênh lệch giữa các môi trường.

```bash
python main.py --data-dir du-lieu --offline --device cpu --threads 2 --seed 42 --epochs 1 --trials 8 --deterministic --output-dir outputs_demo
```

### So sánh CNN với FNN trên bài toán ảnh

| Khía cạnh | CNN | FNN kết nối đầy đủ |
|---|---|---|
| Đầu vào và liên kết | Giữ trục không gian. Bộ lọc nhìn vùng cục bộ và dùng chung tham số ở các vị trí. | Trải phẳng ảnh. Mỗi nơ-ron ở lớp đầu nhận toàn bộ pixel với một trọng số riêng cho mỗi vị trí. |
| Ưu điểm | Chia sẻ tham số giúp học họa tiết và cạnh xuất hiện ở nhiều vị trí. Pooling giảm kích thước và độ nhạy với dịch chuyển nhỏ. | Kết nối trực tiếp mọi phần tử đầu vào. Kiến trúc đơn giản và phù hợp khi đầu vào là véc-tơ đặc trưng có kích thước nhỏ. |
| Hạn chế và chi phí | Pooling có thể làm mất chi tiết. Chi phí tính toán phụ thuộc kích thước bản đồ đặc trưng, số bộ lọc và cấu hình triển khai. | Lớp dày đặc lớn tăng số tham số theo số pixel và kích thước lớp ẩn. Mạng phải học quan hệ không gian và đặc trưng ở các vị trí từ dữ liệu. |

Ví dụ, lớp tích chập đầu của CNN dùng $24(1\cdot3\cdot3+1)=240$ tham số để sinh bản đồ $24\times28\times28$. Một lớp dày đặc nối 784 pixel với cùng số phần tử đầu ra cần $(784+1)\cdot24\cdot28\cdot28=14.770.560$ tham số. Ví dụ này so sánh hai lớp có cùng số phần tử đầu ra. Bảng dưới đo trực tiếp hai mô hình hoàn chỉnh có số tham số gần nhau.

#### Đối chứng thực nghiệm trên cùng phép chia

`fnn_baseline.py` chạy FNN `784 → 384 → 10` với ReLU ở tầng ẩn và CNN cơ sở trong cùng chương trình. FNN tự tính lan truyền tiến, cross-entropy, gradient và cập nhật Adam bằng NumPy. Hai mô hình dùng cùng 5.000 ảnh Train, 5.000 ảnh Validation, 10.000 ảnh Test, seed 42, batch size 64, 10 epoch và Adam với learning rate 0,001. Cả hai huấn luyện trên ảnh đã chuẩn hóa. Chạy từ thư mục `cau2`.

```bash
python fnn_baseline.py --data-dir du-lieu --offline --seed 42 --epochs 10 --threads 2 --output-dir outputs/fnn_comparison
```

Kết quả tham chiếu đo trên Intel Xeon Platinum 8573C với 2 luồng, NumPy 2.5.4 và PyTorch 2.7.1+cpu. Thời gian gồm khởi tạo mô hình, cập nhật tham số và đo Train/Validation sau mỗi epoch. Việc tải và chuẩn hóa dữ liệu, đánh giá Test và ghi CSV diễn ra ngoài khoảng đo. [Kết quả và cấu hình lần chạy](outputs/fnn_comparison.csv)

| Mô hình | Tham số | Validation accuracy | Test accuracy | Thời gian huấn luyện (giây) |
|---|---:|---:|---:|---:|
| FNN NumPy | 305.290 | 84,30% | 83,07% | 1,85 |
| CNN PyTorch | 313.130 | 87,70% | 87,03% | 17,48 |

Trong lần đo đối chứng này, CNN đạt Test accuracy cao hơn **3,96 điểm phần trăm** với số tham số gần FNN. FNN hoàn tất huấn luyện nhanh hơn với hai bản cài đặt và phần cứng đang dùng. Thời gian ở bảng này mô tả một lần chạy. Bộ số liệu dùng để phân tích Dropout, tăng cường dữ liệu và Random Search được trình bày riêng ở mục kết quả tham chiếu 10 epoch. [Bộ số liệu của bốn cấu hình CNN và Random Search](outputs/summary.csv)

## Danh sách tự kiểm tra

- [x] 1. Tập Train nhỏ, chia phân tầng theo seed, Test cố định, thống kê chuẩn hóa chỉ từ Train.
- [x] 2. CNN hai khối Conv–ReLU–Pool, một FC ẩn, mười logit, kích thước ghi bên cạnh các lớp.
- [x] 3. Hai kỹ thuật Dropout và Data Augmentation có cờ độc lập.
- [x] 4. Bốn cấu hình chính chạy cùng seed, số epoch và phép chia dữ liệu.
- [x] 5. Mỗi mô hình cuối duyệt Test một lượt để tính loss, accuracy, macro precision/recall/F1 và confusion matrix.
- [x] 6. PNG đường cong Train/Validation, confusion matrix và ảnh dự đoán sai.
- [x] 7. Random Search 8–12 lượt dùng Validation, huấn luyện lại cấu hình thắng, so sánh với cơ sở bằng bảng và PNG.
- [x] 8. Hàm dự đoán một ảnh và in top 3 xác suất.
- [x] Các mô-đun có vai trò riêng. Có siêu tham số, requirements, hướng dẫn chạy, seed và deterministic.
- [x] Kết quả tạo từ dữ liệu thật bằng chương trình, in bảng pandas và lưu CSV.
