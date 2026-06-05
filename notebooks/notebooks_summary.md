# Tổng Quan Chi Tiết Hệ Thống Notebooks (StableVITON Pipeline)

Thư mục này chứa 2 file Jupyter Notebook được tách ra từ luồng chạy chính của web demo (tương đương với luồng xử lý trong `python3.11 -m modal serve modal_app.py` gọi tới `pipeline.py`).

Hệ thống hiện tại đã được nâng cấp đáng kể với hàng loạt kỹ thuật chuyên sâu so với bản gốc của StableVITON:
- **Kiến trúc Độc lập 100% (Standalone Architecture)**: Cả 2 notebook đều được sinh tự động bằng script để nhúng (inline) toàn bộ mã nguồn thô của các mô hình AI trực tiếp vào từng cell. Hoàn toàn KHÔNG dùng lệnh `import` từ các file `.py` bên ngoài, giúp người dùng copy lên Kaggle hoặc Google Colab là chạy ngay lập tức.
- **Quản lý Bộ Nhớ Tối Ưu (VRAM Management)**: Tích hợp kỹ thuật context manager `with gpu_stage("..."):` để load tuần tự 5 mô hình nặng (DINO, SAM, SegFormer, DensePose, StableVITON) lên GPU và xả bộ nhớ (`torch.cuda.empty_cache()`) ngay sau khi dùng xong, chống lỗi tràn VRAM (Out-Of-Memory) trên các GPU yếu.
- **Hỗ trợ Đa Danh Mục Trang Phục (Multi-Category Try-On)**: Linh hoạt thử áo (`Upper-body`), quần (`Lower-body`) hoặc đầm (`Dress`) dựa trên thuật toán lựa chọn động (Dynamic Label Selection). Thay vì fix cứng nhãn 4 (áo), hệ thống sẽ tìm kiếm các nhãn 5 (váy) hoặc 6 (quần) tùy theo lựa chọn của người dùng.
- **Bảo Tồn Vật Chắn V10 (V10 Occlusion Patching)**: Cải tiến từ V10 Patch gốc, thuật toán gộp chung túi xách, mèo vào `agnostic_mask` (bắt buộc AI phải vẽ áo/quần ngầm bên dưới vật thể), sau đó dùng Alpha Blending đắp ngược vật thể lên trên với độ mịn cao nhất (nhưng loại bỏ bước dilation 25px gây lỗi lem màu).

---

## 1. `offline_preprocessing.ipynb` (Giai đoạn Tiền Xử Lý)

**Mục tiêu:** Chạy tất cả các mô hình phân tích ảnh người (Person Image) và lưu lại các file trung gian ra thư mục `outputs/preprocessed/`. Đây là bước tính toán cực nặng nhưng chỉ cần chạy 1 lần.

**Cấu hình đầu vào (Config):**
- **Chuẩn hóa**: Ảnh được hàm `resize_rgb` tự động đưa về tỷ lệ vàng `(768, 1024)`.
- **Category**: Tham số định tuyến xóa đồ (`Upper-body`, `Lower-body`, `Dress`).
- **Preservation**: Khai báo danh sách các vật cần giữ nguyên (VD: `["bag", "cat"]`) và tính năng giữ tay (`preserve_arms = True`).

**Các bước xử lý chi tiết (Stages):**
- **Stage 1: Object Detection (Grounding DINO)**  
  Chuyển đổi danh sách vật thể thành text prompt. Grounding DINO tìm kiếm và trả về hộp giới hạn (Bounding Boxes). Nhờ `gpu_stage`, mô hình chỉ nằm trên VRAM khoảng 2 giây rồi bị xóa sạch.
  
- **Stage 2: Object Segmentation (SAM - Segment Anything)**  
  Dựa vào Bounding Boxes từ Stage 1, SAM vẽ mask nhị phân chi tiết cho từng vật. Các mask được gộp (`np.bitwise_or`) tạo thành `object_mask`. (Không dùng bộ lọc kích thước để tránh làm đứt đuôi mèo hay dây đeo túi).

- **Stage 3: Person Parsing (SegFormer & Lớp Khiên Bảo Vệ)**  
  Tùy vào biến `category`:
  - Thử áo (`Upper-body`): Xóa vùng có nhãn `4` (Upper-clothes).
  - Thử quần (`Lower-body`): Xóa nhãn `5, 6` (Skirt, Pants).
  - Thử đầm (`Dress`): Xóa nhãn `4, 5, 6`.
  
  **Bước bảo vệ cốt lõi**:
  - Gộp chung vùng cần xóa với da thịt lộ ra (nhãn `14, 15` cho áo, hoặc `12, 13` cho quần).
  - Phóng to (`cv2.dilate` 15x15) vùng này để xóa trọn vẹn cả viền nếp gấp quần/áo.
  - **Giới hạn lây lan**: Đối với quần, sử dụng thêm nhãn áo (4) để chặn không cho mask quần phình ngược lên trên áo. Tương tự, chặn mask lây lan lên mặt, tóc.
  Output là `agnostic_img` (ảnh người bị tẩy đồ cũ thành màu xám `127`) và `agnostic_mask` (mask vùng đã tẩy).
  
- **Stage 4: DensePose Estimation (Detectron2)**  
  Trích xuất bề mặt 3D (pose map) với các kênh màu RGB mã hóa tọa độ không gian U, V.
  
- **Tạo Tấm Khiên Bảo Tồn (Preservation Mask) Khôi Phục Bàn Tay:**  
  Thuật toán tự động kích hoạt `preserve_arms = True` cho chế độ thử quần (`Lower-body`) để bảo vệ bàn tay đặt trên đùi. Mask tay (nhãn `14, 15`) được trích xuất, lọc nhiễu bằng Morphological Open/Close (`open_k=3, close_k=5`), rồi gộp chung với `object_mask` (của SAM) bằng hàm `np.maximum`. Tấm khiên này được lưu thành `final_restore_mask.png` để chuẩn bị cho quá trình dán đè ở Stage 7.

---

## 2. `online_inference.ipynb` (Giai đoạn Thử Đồ)

**Mục tiêu:** Nhận đầu vào là ảnh trang phục mới, kết hợp dữ liệu từ Phase 1 để chạy StableVITON. Bước này diễn ra cực nhanh (dưới 5s).

**Các bước xử lý chi tiết (Stages):**
- **Stage 5: Cloth Parsing (SegFormer Mở Rộng)**  
  Cắt nền ảnh chụp trang phục phẳng. Điểm đặc biệt là thay vì fix cứng `[4]`, hệ thống dùng hàm `pick_best_top_labels` quét qua tổ hợp 11 nhãn (ví dụ: `[4]`, `[5]`, `[6]`, `[5, 6]`,...) để đối chiếu với diện tích thật. Sau đó, nó dùng thuật toán xử lý hình thái học (`refine_top_mask` với `morphologyEx`) để xóa nhiễu và lấp lỗ hổng, tạo ra `cloth_mask` nguyên khối, sạch sẽ tuyệt đối hỗ trợ cả quần, áo và đầm.

- **Stage 6: Try-On Inference (StableVITON) & V10 Patching Cải Tiến**  
  Đưa toàn bộ dữ liệu vào mạng StableVITON. Bước này áp dụng kỹ thuật **V10 Patching**: gộp `object_mask` (túi xách/mèo) vào `agnostic_mask`. Điều này ép AI phải xóa túi đi và cố gắng tự "tưởng tượng" (hallucinate) ra phần áo/quần bị che lấp ngầm bên dưới túi. (Lưu ý: Không dùng dilation 25x25 như bản cũ để tránh làm lem họa tiết quần lên áo).

- **Stage 7: Post-processing & Alpha Blending (Chồng lớp 3D khôi phục vật thể)**  
  Dù Stage 6 đã né vẽ lên mèo, nhưng viền ranh giới giữa áo mới và lông mèo sẽ bị hiện tượng "răng cưa" hoặc viền đen. Thuật toán **Alpha Blending** tinh vi (`refined_restore_occluder`) được dùng để khôi phục lớp 3D hoàn hảo:
  
  1. **Nạp dữ liệu**: Lấy ảnh Try-on (`result_img`), ảnh người gốc (`original_img`), và khiên `final_restore_mask`.
  2. **Tiền xử lý Mask (Erode & Blur)**:
     - `cv2.erode(iterations=1)`: Bào mòn đường viền mask để loại bỏ viền đen rác.
     - `cv2.GaussianBlur(ksize=5)`: Làm mờ nhòe (feathering) ranh giới mask để tạo dải ma trận gradient từ `0.0` đến `1.0`. Dải này chính là kênh **`alpha`**.
  3. **Trộn điểm ảnh (Matrix Blending)**: 
     ```python
     blended_np = orig_np * alpha + res_np * (1.0 - alpha)
     ```
     - **Trong lõi vật thể (Tay/Mèo)**: `alpha = 1.0` $\rightarrow$ Lấy `100%` pixel từ ảnh gốc, giữ nguyên vẹn 100% chi tiết lông thú/da người thực tế.
     - **Bên ngoài (Áo/Quần mới)**: `alpha = 0.0` $\rightarrow$ Lấy `100%` pixel từ ảnh Try-on sinh ra.
     - **Vùng viền chuyển giao**: `alpha = 0.5` $\rightarrow$ Trộn `50%` lông mèo thực tế với `50%` sợi vải áo mới, tạo sự hòa quyện (smooth blending) cực kì êm ái, xóa tan mọi ranh giới cắt ghép giả tạo.

  Kết quả cuối cùng là bộ ảnh phân lớp 3D tự nhiên không tì vết, được lưu tại `outputs/final_tryon_result.png`.
