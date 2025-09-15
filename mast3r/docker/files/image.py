#!/usr/bin/env python3
"""
Image processing utilities for MASt3R
Provides resizing functions based on dust3r.utils.image logic
"""

import PIL.Image
from PIL.ImageOps import exif_transpose
from dust3r.utils.image import _resize_pil_image


def resize_image(image_path, size, square_ok=False, patch_size=16):
    """
    Resize an image using the same logic as dust3r.utils.image.load_images
    but without tensor conversion and normalization.
    
    Args:
        image_path (str): Path to the image file
        size (int): Target size for resizing
        square_ok (bool): Whether square images are acceptable
        patch_size (int): Patch size for alignment (default 16)
    
    Returns:
        PIL.Image: Resized PIL image
    """
    # Load image with EXIF handling
    img = exif_transpose(PIL.Image.open(image_path)).convert('RGB')
    W1, H1 = img.size
    
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
    else:
        # resize long side to specified size
        img = _resize_pil_image(img, size)
    
    W, H = img.size
    cx, cy = W//2, H//2
    
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw = ((2 * cx) // patch_size) * patch_size / 2
        halfh = ((2 * cy) // patch_size) * patch_size / 2
        if not square_ok and W == H:
            halfh = 3*halfw/4
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    
    return img


def resize_images(image_paths, size, square_ok=False, patch_size=16):
    """
    Resize multiple images using the same logic as dust3r.utils.image.load_images
    but without tensor conversion and normalization.
    
    Args:
        image_paths (list): List of paths to image files
        size (int): Target size for resizing
        square_ok (bool): Whether square images are acceptable
        patch_size (int): Patch size for alignment (default 16)
    
    Returns:
        list: List of resized PIL images
    """
    resized_images = []
    for image_path in image_paths:
        resized_img = resize_image(image_path, size, square_ok, patch_size)
        resized_images.append(resized_img)
    
    return resized_images


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python image.py <image_path> <size>")
        print("Example: python image.py /path/to/image.jpg 512")
        sys.exit(1)
    
    image_path = sys.argv[1]
    size = int(sys.argv[2])
    
    try:
        resized_img = resize_image(image_path, size)
        print(f"Successfully resized image to {resized_img.size}")
        
        # Save resized image
        output_path = f"resized_{size}_{image_path.split('/')[-1]}"
        resized_img.save(output_path)
        print(f"Saved resized image to: {output_path}")
        
    except Exception as e:
        print(f"Error resizing image: {e}")
        sys.exit(1)
