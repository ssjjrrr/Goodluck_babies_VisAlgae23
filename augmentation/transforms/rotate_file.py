import os
import cv2 as cv
import numpy as np
import random

# Function to perform the rotation and blending on a single image
def process_image(image_path, output_folder):
    # Load algae image and background image
    image_algae = cv.imread(image_path)
    image_bg = cv.imread('F:/github/Goodluck_babies_VisAlgae23/augmentation/random_gen/gen_augmenter_src/backgrounds/back_0.jpg')

    # Define rotation angle
    angle = random.uniform(0, 10)

    # Get rotation matrix
    image_center = tuple(np.array(image_algae.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)

    sin_cos = rot_mat[:, 1]
    size_convert = np.mat([[sin_cos[1], sin_cos[0]], [sin_cos[0], sin_cos[1]]])
    alg_size = image_algae.shape[1::-1]
    out_size = np.dot(alg_size, size_convert)
    w_convert = int(out_size[:, 0])
    h_convert = int(out_size[:, 1])
    rot_mat[0, 2] += (w_convert - image_center[0] * 2) * 0.5
    rot_mat[1, 2] += (h_convert - image_center[1] * 2) * 0.5

    # Rotate algae image
    rotated_algae = cv.warpAffine(image_algae, rot_mat, (w_convert, h_convert), flags=cv.INTER_NEAREST)

    # Crop image to reduce black border size
    h_cut = int(image_center[0] * sin_cos[0])
    w_cut = int(image_center[1] * sin_cos[0])
    h_end = int(h_convert - h_cut)
    w_end = int(w_convert - w_cut)
    cropped = rotated_algae[h_cut:h_end, w_cut:w_end]

    # Create rotated image mask
    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    mask[cropped.any(axis=-1)] = 255

    # Set seamless clone center position (adjust if needed)
    center = (image_bg.shape[1] // 2, image_bg.shape[0] // 2)

    # Perform seamless cloning
    blended_image = cv.seamlessClone(cropped, image_bg, mask, center, cv.NORMAL_CLONE)

    # Save the result
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    # cv.imwrite(output_path, blended_image)
    cv.imwrite(output_path, cropped)
    print(f"Processed and saved: {output_path}")

# Folder containing algae images
input_folder = 'F:\github\Goodluck_babies_VisAlgae23/augmentation/random_gen\gen_augmenter_src/algaes/4Porphyridium/'

# Output folder for saving processed images
output_folder = './rotate_output/algaes/4Porphyridium/'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, output_folder)
