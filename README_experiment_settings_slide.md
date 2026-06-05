# Đề cương Slide: Thiết lập Thực nghiệm (Experimental Settings)

Dựa vào mã nguồn `academic_eval.py` và luồng chạy của `pipeline.py`, dưới đây là gợi ý chi tiết nội dung để bạn đưa vào slide báo cáo phần "Thiết lập thực nghiệm" (Experimental Settings). 

Bạn có thể chia thành 1-2 slide tùy theo thời gian thuyết trình.

---

## 🖥️ Slide 1: Thiết lập Thực nghiệm (Tổng quan)

**1. Mục tiêu (Objective):**
* So sánh chất lượng ảnh sinh ra (Image Quality) và khả năng xử lý vật cản / giữ gìn chi tiết gốc (Occlusion & Detail Preservation) giữa hệ thống đề xuất và mô hình gốc.

**2. Các phương pháp so sánh (Methods):**
* **StableVITON (Baseline):** Mô hình Try-On gốc không có các module tiền xử lý/hậu xử lý vật thể phức tạp.
* **Ours (Pipeline):** Hệ thống đề xuất tích hợp luồng xử lý cải tiến (Grounding DINO + SAM, Dynamic SegFormer, Alpha Blending, v.v.).

**3. Tập dữ liệu đánh giá (Test Set):**
* Chạy kiểm thử trên đa dạng danh mục trang phục: **Upper-body** (Áo) và **Lower-body** (Quần).
* Cặp ảnh đối chứng (Ground Truth) được tự động scale về chuẩn độ phân giải đầu ra `(384, 512)` để đảm bảo công bằng.

**4. Môi trường triển khai:**
* Đánh giá và inference tự động trên GPU với cơ chế giải phóng VRAM (VRAM Management) tối ưu cho từng module (GPU Stage Context).

---

## 📊 Slide 2: Các độ đo đánh giá (Evaluation Metrics)

Để đánh giá toàn diện, thực nghiệm sử dụng 3 nhóm độ đo tiêu chuẩn trong bài toán sinh ảnh:

**1. Đánh giá cấp độ Pixel (Pixel Level):**
* **PSNR (Peak Signal-to-Noise Ratio - ↑):** Đánh giá tỷ lệ nhiễu hoặc méo ảnh ở mức độ pixel. *Chỉ số cao hơn là tốt hơn.*

**2. Đánh giá cảm nhận thị giác (Perceptual Level):**
* **LPIPS (Learned Perceptual Image Patch Similarity - ↓):** Dùng mạng nơ-ron (VGG) trích xuất đặc trưng để đánh giá độ sai lệch theo "cảm nhận" của mắt người thay vì chỉ so sánh pixel toán học. *Chỉ số thấp hơn là tốt hơn.*

**3. Đánh giá phân phối tổng thể (Generative & Distribution Level):**
* **FID (Fréchet Inception Distance - ↓):** Đo lường khoảng cách giữa tập phân phối đặc trưng (feature distribution) của ảnh sinh ra và tập ảnh thật. *Chỉ số thấp hơn là tốt hơn.*
* **KID (Kernel Inception Distance - ↓):** Tương tự FID nhưng dùng ước lượng không chệch (unbiased), cho kết quả đánh giá khách quan và chính xác hơn trên các tập dữ liệu/batch size nhỏ. *Chỉ số thấp hơn là tốt hơn.*

---

## 💡 Mẹo khi thuyết trình (Speaker Notes):
* **Nhấn mạnh:** "Vì Pipeline của chúng em (Ours) có khả năng giữ lại tay (khi thử quần) và giữ lại đồ vật (túi, mèo), ảnh đầu ra của chúng em sẽ khớp với Ground Truth (ảnh thực tế) hơn rất nhiều so với Baseline gốc, do đó các chỉ số như PSNR sẽ cao hơn và LPIPS, FID sẽ thấp hơn rõ rệt."
* Bảng kết quả (Table) trong file `academic_eval.py` sẽ in ra đủ 4 chỉ số này. Bạn nên chạy ra kết quả rồi copy bảng đó dán vào slide tiếp theo luôn (Slide Kết quả - Results).
