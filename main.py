import cv2
import numpy as np
import os

# Load an image from a given path in grayscale mode
def load_image(path):
    try:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Could not load image: {path}")
        return image
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

# Apply a predefined 5x5 Gaussian filter manually using convolution
def apply_gaussian_blur(gray_image):
    gaussian_kernel = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ], dtype=np.float32)
    
    gaussian_kernel /= 273.0  # Normalize kernel to retain brightness

    # Perform convolution with the kernel
    blurred = cv2.filter2D(gray_image, -1, gaussian_kernel)
    return blurred

# Save the processed image to disk
def save_image(path, image):
    try:
        cv2.imwrite(path, image)
    except Exception as e:
        print(f"Saving failed for {path}: {e}")

# Process all .tiff images in the input directory with Gaussian blur
def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.tiff', '.tif')):
            input_path = os.path.join(input_dir, fname)
            output_name = f"{os.path.splitext(fname)[0]}_gaussian.tiff"
            output_path = os.path.join(output_dir, output_name)

            # Load -> Filter -> Save pipeline
            img = load_image(input_path)
            if img is None:
                continue

            blurred_img = apply_gaussian_blur(img)
            save_image(output_path, blurred_img)

            print(f"Done: {output_path}")

# Main function with default folder setup
def main():
    input_dir = "input_images"
    output_dir = "output_images"
    process_folder(input_dir, output_dir)

if __name__ == "__main__":
    main()
