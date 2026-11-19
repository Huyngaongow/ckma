# Câu 1 — Hồi quy cường độ nén bê tông bằng FNN

## Dữ liệu và tiền xử lý

Thí nghiệm sử dụng bộ **Concrete Compressive Strength** do I-Cheng Yeh cung cấp cho UCI Machine Learning Repository. Bộ dữ liệu có 1.030 mẫu, tám đặc trưng đầu vào và một biến mục tiêu là cường độ nén, tính bằng MPa. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)

Máy đám mây nhận lỗi HTTP 403 khi tải trực tiếp từ máy chủ UCI. Tệp `cau1/data/Concrete_Data.xls` được lấy từ bản lưu công khai có kèm tài liệu của bộ dữ liệu. Toàn bộ giá trị trong tệp được đối chiếu với một bản XLS lưu độc lập và khớp hoàn toàn. Tệp `concrete.csv` được xuất từ XLS, đổi tên cột và giữ nguyên giá trị số. [Bản XLS trong repo được lấy từ đây](https://raw.githubusercontent.com/zackwwhite/UCI-Concrete-Compressive-Strength/9f3cdc046703c2846a163f5b86aba323bf473378/data/Concrete_Data.xls) · [Tài liệu đi kèm](https://raw.githubusercontent.com/zackwwhite/UCI-Concrete-Compressive-Strength/9f3cdc046703c2846a163f5b86aba323bf473378/data/Concrete_Readme.txt) · [Bản XLS dùng để đối chiếu](https://raw.githubusercontent.com/alexandrenm2/Concrete-compressive-strength-prediction/bf18488cd1cae35c66df1c17c554c2ecaea5f1ae/Concrete_Data.xls)

Chương trình kiểm tra dữ liệu thiếu và dòng trùng trước khi chia tập. Dữ liệu có 0 ô thiếu và **25 dòng trùng bị loại, còn 1.005 mẫu**. Tỷ lệ chia xấp xỉ 70/15/15, với 703 mẫu Train, 151 mẫu Validation và 151 mẫu Test. Các scaler của $X$ và $y$ được fit trên Train rồi dùng để biến đổi cả ba tập.

## Hàm mất mát và gradient

Mạng có hai tầng ẩn lần lượt 64 và 32 nơ-ron, dùng tanh, khởi tạo Xavier và bắt đầu với bias bằng 0. Tầng cuối cho đầu ra tuyến tính. Gọi $N$ là số mẫu Train, $B$ là kích thước thực tế của mini-batch và $L$ là chỉ số tầng ra. Ở tầng $\ell$, phép truyền xuôi là $Z^{(\ell)}=A^{(\ell-1)}W^{(\ell)}+b^{(\ell)}$. Với tầng ẩn, $A^{(\ell)}=\tanh Z^{(\ell)}$. Tầng ra trả về $\hat y=Z^{(L)}$.

Hàm mục tiêu mà mỗi bước Gradient Descent dùng là **nửa MSE mini-batch cộng L2**.

$$
J_B = \underbrace{\frac{1}{2B}\sum_{i=1}^B (\hat y_i-y_i)^2}_{\frac12\operatorname{MSE}_B}
      + \frac{\lambda}{2N}\sum_{\ell=1}^{L}\lVert W^{(\ell)}\rVert_F^2.
$$

Đạo hàm của bình phương sai số đưa ra hệ số 2, triệt tiêu hệ số $1/2$. Tầng ra tuyến tính có đạo hàm bằng 1 nên $dZ^{(L)}=(\hat y-y)/B$. Với tầng ẩn, quy tắc dây chuyền nhân gradient truyền ngược với đạo hàm của tanh, phần tử theo phần tử.

$$
dZ^{(\ell)}
= \left(dZ^{(\ell+1)}(W^{(\ell+1)})^\top\right)
  \odot \left(1-(A^{(\ell)})^2\right).
$$

Lấy đạo hàm theo $W$ của tổng bình phương Frobenius cho $2W$. L2 chỉ tác động lên ma trận trọng số.

$$
dW^{(\ell)} = (A^{(\ell-1)})^\top dZ^{(\ell)} + \frac{\lambda}{N}W^{(\ell)},
\qquad
db^{(\ell)} = \sum_{i=1}^{B} dZ^{(\ell)}_{i,:}.
$$

Hệ số $1/B$ đã có trong gradient tầng ra. Mỗi bước cập nhật trừ learning rate nhân với gradient tương ứng khỏi $W$ và $b$.

## Thiết kế thí nghiệm

Trong mỗi seed, bốn cấu hình dùng cùng tập chia, cùng tham số khởi tạo và cùng thứ tự xáo trộn mini-batch. FNN cơ bản và FNN + L2 chạy đủ 2.500 epoch. Hai cấu hình Early Stopping dùng `patience=150`, `min_delta=1e-5` và khôi phục bộ tham số của epoch có Validation Loss thấp nhất.

Chương trình chọn một $\lambda$ từ $\{0.1, 1, 5, 20\}$ theo Validation Loss tại epoch cuối của các lượt chạy L2 cố định, rồi dùng chung giá trị đó cho hai cấu hình có L2. Đường loss Train và Validation dùng nửa MSE thuần trên $y$ chuẩn hóa. Bảng metric dùng dự đoán đã đổi về thang gốc bằng `inverse_transform`. MSE được tính theo MPa², còn RMSE và MAE theo MPa. Mỗi mô hình cuối dự đoán Test một lần trong mỗi seed.

## Kết quả

Gradient check dùng sai phân trung tâm trên mạng nhỏ và năm mẫu Train, với L2 bật. Sai số tương đối đo được là `2.643e-11`, dưới ngưỡng `1e-7`.

Bảng ghi trung bình ± độ lệch chuẩn mẫu qua các seed 42, 7, 13, 21 và 73.

| Cấu hình | Tập | MSE (MPa²) | RMSE (MPa) | MAE (MPa) |
| --- | --- | ---: | ---: | ---: |
| FNN | Train | 7.4979 ± 0.7105 | 2.7357 ± 0.1317 | 1.9570 ± 0.0961 |
| FNN | Validation | 28.8066 ± 9.2305 | 5.3204 ± 0.7903 | 3.5852 ± 0.2871 |
| FNN | Test | 27.4767 ± 4.0049 | 5.2304 ± 0.3860 | 3.6376 ± 0.2337 |
| FNN + ES | Train | 15.1358 ± 3.1776 | 3.8727 ± 0.4150 | 2.9233 ± 0.3331 |
| FNN + ES | Validation | 28.3714 ± 5.9036 | 5.3037 ± 0.5508 | 3.7854 ± 0.4170 |
| FNN + ES | Test | 29.6540 ± 4.7233 | 5.4318 ± 0.4326 | 3.9441 ± 0.3816 |
| FNN + L2 | Train | 10.8455 ± 3.3804 | 3.2620 ± 0.5059 | 2.3957 ± 0.4219 |
| FNN + L2 | Validation | 27.6296 ± 7.6443 | 5.2216 ± 0.6754 | 3.6524 ± 0.3451 |
| FNN + L2 | Test | 30.5944 ± 4.3169 | 5.5199 ± 0.3958 | 3.8124 ± 0.3126 |
| FNN + ES + L2 | Train | 16.8300 ± 1.7468 | 4.0981 ± 0.2104 | 3.0966 ± 0.1721 |
| FNN + ES + L2 | Validation | 29.1340 ± 6.5091 | 5.3719 ± 0.5878 | 3.8791 ± 0.3757 |
| FNN + ES + L2 | Test | 31.6401 ± 5.0066 | 5.6101 ± 0.4571 | 4.0554 ± 0.3569 |

Ở seed 42, Validation Loss chọn $\lambda=1$. FNN cơ bản đạt Validation Loss thấp nhất tại epoch 888 và tiếp tục huấn luyện đến epoch 2.500. Khi áp dụng Early Stopping, mô hình dừng ở epoch 1.038 và khôi phục epoch 888. Khoảng cách Validation–Train giảm từ 0.07344 xuống 0.04358 và Test RMSE giảm từ 5.3194 xuống 5.2817 MPa. Test MAE tăng từ 3.5506 lên 3.7419 MPa.

Qua năm seed, FNN + L2 có Validation RMSE trung bình thấp nhất, còn FNN cơ bản có Test RMSE và MAE trung bình thấp nhất. Mức cải thiện quan sát được phụ thuộc vào cách chia dữ liệu và metric. Bảng Test dùng để so sánh sau huấn luyện. Việc chọn $\lambda$ và epoch dùng Validation.
