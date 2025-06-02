import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

PATCH_SIZE = 128
STRIDE = 64
SAVE_DIR = "./patches"
IMAGE_DIR = "DRIVE/training/images"
MASK_DIR = "DRIVE/training/1st_manual"

def extract_patches(img, patch_size=128, stride=64):
    h, w = img.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches

def save_patches(patches, base_name, subdir):
    os.makedirs(subdir, exist_ok=True)
    for i, patch in enumerate(patches):
        filename = os.path.join(subdir, f"{base_name}_patch_{i:04d}.png")
        cv2.imwrite(filename, patch)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    image_files = sorted(os.listdir(IMAGE_DIR))
    total_patches = 0
    
    for fname in tqdm(image_files):
        if not fname.endswith(".tif"):
            continue
        base_name = fname.replace("_training.tif", "")
        
        # Load image and mask
        img_path = os.path.join(IMAGE_DIR, fname)
        mask_path = os.path.join(MASK_DIR, f"{base_name}_manual1.gif")

        img = cv2.imread(img_path)
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)

        # Extract patches
        img_patches = extract_patches(img, PATCH_SIZE, STRIDE)
        mask_patches = extract_patches(mask, PATCH_SIZE, STRIDE)

        # Update patch count
        total_patches += len(img_patches)

        # Save patches
        save_patches(img_patches, base_name, os.path.join(SAVE_DIR, "images"))
        save_patches(mask_patches, base_name, os.path.join(SAVE_DIR, "masks"))

    print(f"\n✅ Total patches extracted: {total_patches}")

if __name__ == "__main__":
    main()
