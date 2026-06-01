import os
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# ================= 配置区域 =================
CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
MASK_THRESHOLD = 0.0

# 解除 Pillow 大图限制
Image.MAX_IMAGE_PIXELS = None 
# ===========================================

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("❌ 错误: 无法导入 SAM 2。")
    sys.exit(1)

input_points = []
input_labels = []

def mouse_callback(event, x, y, flags, param):
    global input_points, input_labels
    if event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])
        input_labels.append(1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        input_points.append([x, y])
        input_labels.append(0)

def select_object_interactive(img_path):
    global input_points, input_labels
    input_points = []
    input_labels = []
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取: {img_path}")
        return None, None

    h, w = img.shape[:2]
    scale = 1.0
    if h > 1000:
        scale = 1000 / h
        display_w = int(w * scale)
        display_h = int(h * scale)
        display_img = cv2.resize(img, (display_w, display_h))
    else:
        display_img = img.copy()

    window_name = "Select Object (Space: Confirm)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print(f"\n[交互] 左键红点，右键绿点，空格确认...")

    while True:
        temp_img = display_img.copy()
        for pt, label in zip(input_points, input_labels):
            cx, cy = int(pt[0]), int(pt[1])
            color = (0, 0, 255) if label == 1 else (0, 255, 0)
            cv2.circle(temp_img, (cx, cy), 5, color, -1)
        cv2.imshow(window_name, temp_img)
        key = cv2.waitKey(1)
        if key == 32: 
            if len(input_points) > 0: break
        elif key == 27: sys.exit()
        elif key == ord('c'): 
            input_points = []
            input_labels = []

    cv2.destroyAllWindows()
    
    real_points = np.array(input_points, dtype=np.float32) / scale
    real_labels = np.array(input_labels, dtype=np.int32)
    return real_points, real_labels

def process_directory(predictor, input_dir):
    input_dir = input_dir.strip().strip('"').strip("'")
    if not os.path.exists(input_dir): return

    clean_path = input_dir.rstrip(os.sep).rstrip('/')
    output_dir = clean_path + "_mask"
    os.makedirs(output_dir, exist_ok=True)
    
    frame_names = [p for p in os.listdir(input_dir) if p.lower().endswith(('.jpg', '.jpeg'))]
    frame_names.sort()
    
    if len(frame_names) == 0:
        print("❌ 文件夹里没找到 .jpg！")
        return

    print(f"\n[任务] {len(frame_names)} 张 | {os.path.basename(input_dir)}")
    
    inference_state = predictor.init_state(video_path=input_dir)
    points, labels = select_object_interactive(os.path.join(input_dir, frame_names[0]))
    if points is None: return

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )

    print("--> 正在生成并保存 (Pillow JPG Mode)...")
    
    for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(
        predictor.propagate_in_video(inference_state), 
        total=len(frame_names)
    ):
        # 1. 提取 Mask (布尔值)
        mask_pred = (out_mask_logits[0] > MASK_THRESHOLD).cpu().numpy()
        
        # 【核心修复】：挤压维度，确保它是 (H, W) 的二维数组，去掉多余的 1
        mask_pred = mask_pred.squeeze() 
        
        # 2. 转为 uint8 (0 或 255)
        mask_u8 = (mask_pred * 255).astype(np.uint8)
        
        # 3. 保存
        original_name = frame_names[out_frame_idx]
        save_name = os.path.splitext(original_name)[0] + ".jpg"
        save_path = os.path.join(output_dir, save_name)
        
        try:
            # 指定 mode='L' 明确告诉它是灰度图
            pil_img = Image.fromarray(mask_u8, mode='L')
            pil_img.save(save_path, quality=100, subsampling=0)
        except Exception as e:
            # 打印更详细的错误形状，方便调试
            print(f"❌ 保存失败: {save_path} | Shape: {mask_u8.shape} | Err: {e}")

    predictor.reset_state(inference_state)
    print(f"✅ 完成！Mask 路径: {output_dir}")

def main():
    if not torch.cuda.is_available(): return
    print(f"Loading Model...")
    try:
        predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device="cuda")
    except Exception as e:
        print(f"❌ {e}")
        return

    while True:
        p = input("\n输入 JPG 文件夹路径 (q退出): ").strip()
        if p.lower() == 'q': break
        if p:
            process_directory(predictor, p)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()