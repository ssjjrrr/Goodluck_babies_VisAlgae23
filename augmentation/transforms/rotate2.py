import cv2
import os
import numpy as np

def read_yolo_labels(label_path, img_width, img_height):
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            labels.append([class_id, x_center, y_center, width, height])
    return labels

def write_yolo_labels(labels, label_path):
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(' '.join(map(str, label)) + '\n')

def rotate_point(x, y, angle, cx, cy):
    angle = np.radians(angle)
    x_new = np.cos(angle) * (x - cx) - np.sin(angle) * (y - cy) + cx
    y_new = np.sin(angle) * (x - cx) + np.cos(angle) * (y - cy) + cy
    return x_new, y_new

def transform_labels_for_rotation(labels, angle, img_width, img_height):
    cx, cy = img_width / 2, img_height / 2
    transformed_labels = []
    for label in labels:
        class_id, x_center, y_center, width, height = label
        x_center, y_center = rotate_point(x_center * img_width, y_center * img_height, -angle, cx, cy)
        x_center, y_center = x_center / img_width, y_center / img_height
        transformed_labels.append([class_id, x_center, y_center, width, height])
    return transformed_labels

def process_image(image_path, label_path, save_path, angle):
    # Read the image
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    # Define the center of the image (cx, cy)
    cx, cy = img_width / 2, img_height / 2

    # Read and transform labels
    labels = read_yolo_labels(label_path, img_width, img_height)
    transformed_labels = transform_labels_for_rotation(labels, angle, img_width, img_height)

    # Rotate the image
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (img_width, img_height))

    # Save the rotated image and labels
    cv2.imwrite(os.path.join(save_path, os.path.basename(image_path)), rotated_image)
    write_yolo_labels(transformed_labels, os.path.join(save_path, os.path.basename(label_path)))
# Example usage
image_path = 'path/to/your/image.jpg'
label_path = 'path/to/your/label.txt'
save_path = 'path/to/save'
angle = 5  # Rotate by 45 degrees

process_image(image_path, label_path, save_path, angle)