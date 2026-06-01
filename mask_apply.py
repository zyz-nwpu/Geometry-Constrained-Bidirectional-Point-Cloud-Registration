import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm

def process_folders(img_dir, mask_dir):
    # 1. 清理路径字符串 (去除引号和空格)
    img_dir = img_dir.strip().strip('"').strip("'")
    mask_dir = mask_dir.strip().strip('"').strip("'")

    if not os.path.exists(img_dir):
        print(f"❌ 原图路径不存在: {img_dir}")
        return
    if not os.path.exists(mask_dir):
        print(f"❌ Mask 路径不存在: {mask_dir}")
        return

    # 2. 自动生成输出路径 (在原图文件夹旁边)
    # 处理路径尾部的斜杠，确保 basename 获取正确
    img_dir_clean = img_dir.rstrip(os.sep).rstrip('/')
    parent_dir = os.path.dirname(img_dir_clean)
    base_name = os.path.basename(img_dir_clean)
    
    # 输出文件夹名字：原名字_apply_mask
    output_dir = os.path.join(parent_dir, base_name + "_apply_mask")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📂 原图目录: {img_dir_clean}")
    print(f"🎭 Mask目录: {mask_dir}")
    print(f"🚀 输出目录: {output_dir}")

    # 3. 获取所有原图
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    img_files = [f for f in os.listdir(img_dir_clean) if f.endswith(valid_exts)]
    img_files.sort()

    if len(img_files) == 0:
        print("❌ 原图文件夹是空的！")
        return

    success_count = 0
    
    # 4. 开始处理
    for img_name in tqdm(img_files, desc="Processing"):
        img_path = os.path.join(img_dir_clean, img_name)
        name_no_ext = os.path.splitext(img_name)[0]
        
        # 读取原图
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 寻找对应的 Mask (尝试多种后缀)
        mask_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
            possible_path = os.path.join(mask_dir, name_no_ext + ext)
            if os.path.exists(possible_path):
                mask_path = possible_path
                break
        
        if mask_path is None:
            # print(f"⚠️ 警告: 找不到对应的 Mask -> {img_name}")
            continue

        # 读取 Mask (灰度模式)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue

        # 尺寸对齐 (以原图为准)
        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 二值化处理 (防止边缘有灰色噪点，特别是JPG mask)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 归一化 mask (0~1) 并扩展维度
        mask_norm = mask.astype(float) / 255.0
        mask_norm = np.expand_dims(mask_norm, axis=2) # (H,W,1)

        # 叠加 (原图 * mask)，背景变全黑 (0,0,0)
        img_masked = (img.astype(float) * mask_norm).astype(np.uint8)

        # 保存 (保持原文件名，强制存为高质量 JPG)
        save_name = name_no_ext + ".jpg"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, img_masked, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        success_count += 1

    print(f"\n✅ 处理完成！共生成 {success_count} 张图片。")
    print(f"📁 结果保存在: {output_dir}")

if __name__ == "__main__":
    print("="*60)
    print(" 🖼️ 图片去背景工具 (Apply Mask)")
    print(" 功能: 读取原图和Mask，自动生成背景全黑的图片到新文件夹")
    print("="*60)
    
    while True:
        print("\n" + "-"*30)
        i_dir = input("1. 请输入【原图】文件夹路径 (q退出): ").strip()
        if i_dir.lower() == 'q': break
        
        m_dir = input("2. 请输入【Mask】文件夹路径: ").strip()
        
        process_folders(i_dir, m_dir)