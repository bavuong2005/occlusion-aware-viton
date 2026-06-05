import os
import cv2
import numpy as np
import shutil
import torch
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def resize_ground_truth_images(results_dir):
    """
    Tìm các ảnh ground_truth_**.png và resize chúng về đúng kích thước
    của ảnh gen ra (ví dụ: upper_test_**.png).
    Ghi đè lại ảnh ground_truth_**.png bằng ảnh đã resize.
    """
    print("[1] Đang kiểm tra và resize ảnh Ground Truth...")
    files = os.listdir(results_dir)
    gt_files = [f for f in files if f.startswith("ground_truth_") and f.endswith(".png")]
    
    if not gt_files:
        print("    -> Chưa tìm thấy ảnh ground_truth_*.png nào trong results/")
        return False

    resized_count = 0
    for gt_file in gt_files:
        gt_path = os.path.join(results_dir, gt_file)
        # ID từ ground_truth_00.png -> 00
        file_id = gt_file.replace("ground_truth_", "").replace(".png", "")
        
        # Tìm một ảnh output bất kỳ có cùng ID để lấy kích thước chuẩn (ưu tiên upper rồi lower)
        ref_file = f"upper_test_{file_id}.png"
        ref_path = os.path.join(results_dir, ref_file)
        if not os.path.exists(ref_path):
            ref_file = f"lower_test_{file_id}.png"
            ref_path = os.path.join(results_dir, ref_file)
            
        if os.path.exists(ref_path):
            img_ref = cv2.imread(ref_path)
            if img_ref is None:
                continue
            target_h, target_w = img_ref.shape[:2]
            
            img_gt = cv2.imread(gt_path)
            if img_gt is not None:
                h, w = img_gt.shape[:2]
                if h != target_h or w != target_w:
                    img_gt_resized = cv2.resize(img_gt, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(gt_path, img_gt_resized)
                    resized_count += 1
    
    print(f"    -> Đã resize {resized_count} ảnh Ground Truth về chuẩn kích thước đầu ra.")
    return True

def evaluate_metrics(results_dir):
    """
    Đánh giá PSNR, LPIPS cho từng ảnh, sau đó tính trung bình.
    Sử dụng FID và KID từ clean-fid cho phân phối tổng.
    In ra định dạng bảng học thuật.
    """
    print("\n[2] Bắt đầu tính toán metrics...")
    
    # Cố gắng load LPIPS
    try:
        import lpips
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        has_lpips = True
    except ImportError:
        print("Lưu ý: Thư viện 'lpips' chưa được cài đặt. Bỏ qua LPIPS. (pip install lpips)")
        has_lpips = False

    files = os.listdir(results_dir)
    
    # Tìm tất cả các file pipeline kết quả (upper_test_XX.png, lower_test_XX.png)
    pipeline_files = [f for f in files if (f.startswith("upper_test_") or f.startswith("lower_test_")) and not f.endswith("_baseline.png")]
    
    metrics = {
        "Baseline": {"psnr": [], "lpips": []},
        "Ours": {"psnr": [], "lpips": []}
    }
    
    # Dùng để chứa ảnh copy ra tính FID
    os.makedirs("tmp_gt", exist_ok=True)
    os.makedirs("tmp_baseline", exist_ok=True)
    os.makedirs("tmp_ours", exist_ok=True)
    
    valid_pairs = 0
    for pipe_file in pipeline_files:
        # Tách ID, ví dụ upper_test_00.png -> 00
        prefix, ext = os.path.splitext(pipe_file)
        if prefix.startswith("upper_test_"):
            file_id = prefix.replace("upper_test_", "")
        elif prefix.startswith("lower_test_"):
            file_id = prefix.replace("lower_test_", "")
        else:
            continue
            
        base_file = f"{prefix}_baseline.png"
        gt_file = f"ground_truth_{file_id}.png"
        
        path_pipe = os.path.join(results_dir, pipe_file)
        path_base = os.path.join(results_dir, base_file)
        path_gt = os.path.join(results_dir, gt_file)
        
        if os.path.exists(path_base) and os.path.exists(path_gt):
            valid_pairs += 1
            
            # Copy file qua tmp_ folder cho FID/KID tính chung (phải dùng tên file unique để không đè)
            shutil.copy(path_gt, os.path.join("tmp_gt", pipe_file))
            shutil.copy(path_base, os.path.join("tmp_baseline", pipe_file))
            shutil.copy(path_pipe, os.path.join("tmp_ours", pipe_file))
            
            # Đọc ảnh để tính PSNR
            img_gt = cv2.imread(path_gt)
            img_pipe = cv2.imread(path_pipe)
            img_base = cv2.imread(path_base)
            
            gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            gray_pipe = cv2.cvtColor(img_pipe, cv2.COLOR_BGR2GRAY)
            gray_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
            
            # PSNR
            metrics["Ours"]["psnr"].append(calculate_psnr(img_gt, img_pipe))
            
            metrics["Baseline"]["psnr"].append(calculate_psnr(img_gt, img_base))
            
            # LPIPS
            if has_lpips:
                # Chuyển BGR -> RGB -> Tensor [-1, 1]
                tensor_gt = lpips.im2tensor(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)).to(device)
                tensor_pipe = lpips.im2tensor(cv2.cvtColor(img_pipe, cv2.COLOR_BGR2RGB)).to(device)
                tensor_base = lpips.im2tensor(cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB)).to(device)
                
                with torch.no_grad():
                    lpips_pipe = loss_fn_vgg(tensor_gt, tensor_pipe).item()
                    lpips_base = loss_fn_vgg(tensor_gt, tensor_base).item()
                
                metrics["Ours"]["lpips"].append(lpips_pipe)
                metrics["Baseline"]["lpips"].append(lpips_base)
                
    if valid_pairs == 0:
        print("-> Không tìm thấy cặp ảnh hợp lệ (cần đủ cả pipeline, baseline và ground_truth).")
        shutil.rmtree("tmp_gt", ignore_errors=True)
        shutil.rmtree("tmp_baseline", ignore_errors=True)
        shutil.rmtree("tmp_ours", ignore_errors=True)
        return

    # Tính FID / KID bằng clean-fid
    fid_base = kid_base = fid_ours = kid_ours = "N/A"
    try:
        from cleanfid import fid
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Đang tính FID và KID trên tập {valid_pairs} ảnh...")
        
        fid_base = fid.compute_fid("tmp_gt", "tmp_baseline", device=device, num_workers=0)
        kid_base = fid.compute_kid("tmp_gt", "tmp_baseline", device=device, num_workers=0)
        
        fid_ours = fid.compute_fid("tmp_gt", "tmp_ours", device=device, num_workers=0)
        kid_ours = fid.compute_kid("tmp_gt", "tmp_ours", device=device, num_workers=0)
    except ImportError:
        print("Lưu ý: clean-fid chưa cài đặt, bỏ qua FID/KID.")
    except Exception as e:
        print(f"Lỗi tính FID/KID: {e}")
        
    # Xóa folder tạm
    shutil.rmtree("tmp_gt", ignore_errors=True)
    shutil.rmtree("tmp_baseline", ignore_errors=True)
    shutil.rmtree("tmp_ours", ignore_errors=True)
    
    # In bảng kết quả chuẩn học thuật
    avg = lambda lst: sum(lst)/len(lst) if lst else 0
    
    psnr_b = avg(metrics["Baseline"]["psnr"])
    lpips_b = avg(metrics["Baseline"]["lpips"])
    
    psnr_o = avg(metrics["Ours"]["psnr"])
    lpips_o = avg(metrics["Ours"]["lpips"])
    
    def fmt(val):
        return f"{val:<10.3f}" if isinstance(val, (int, float)) else f"{val:<10}"

    print("\n" * 3)
    print("="*75)
    print(f"{'Method':<25} | {'PSNR (↑)':<10} | {'LPIPS (↓)':<10} | {'FID (↓)':<10} | {'KID (↓)':<10}")
    print("-" * 75)
    print(f"{'StableVITON Baseline':<25} | {fmt(psnr_b)} | {fmt(lpips_b)} | {fmt(fid_base)} | {fmt(kid_base)}")
    print(f"{'Ours (Pipeline)':<25} | {fmt(psnr_o)} | {fmt(lpips_o)} | {fmt(fid_ours)} | {fmt(kid_ours)}")
    print("="*75 + "\n")

if __name__ == '__main__':
    res_dir = "results"
    if os.path.exists(res_dir):
        # 1. Tự động resize ảnh ground_truth để đồng bộ
        resize_ground_truth_images(res_dir)
        # 2. Chạy đánh giá
        evaluate_metrics(res_dir)
    else:
        print(f"Không tìm thấy thư mục {res_dir}")
