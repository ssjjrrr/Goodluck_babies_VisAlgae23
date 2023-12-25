import os
import xml.etree.ElementTree as ET

def convert_yolo_to_xml(yolo_file, xml_file, image_width, image_height, classes):
    with open(yolo_file, 'r') as file:
        lines = file.readlines()

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'images'
    ET.SubElement(annotation, 'filename').text = os.path.basename(yolo_file).replace('.txt', '.jpg')

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(image_width)
    ET.SubElement(size, 'height').text = str(image_height)
    ET.SubElement(size, 'depth').text = '3'

    for line in lines:
        elements = line.strip().split()
        class_id = int(elements[0])
        center_x = float(elements[1]) * image_width
        center_y = float(elements[2]) * image_height
        width = float(elements[3]) * image_width
        height = float(elements[4]) * image_height

        x_min = int(center_x - width / 2)
        y_min = int(center_y - height / 2)
        x_max = int(center_x + width / 2)
        y_max = int(center_y + height / 2)

        object = ET.SubElement(annotation, 'object')
        ET.SubElement(object, 'name').text = classes[class_id]
        ET.SubElement(object, 'pose').text = 'Unspecified'
        ET.SubElement(object, 'truncated').text = '0'
        ET.SubElement(object, 'difficult').text = '0'

        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(x_min)
        ET.SubElement(bndbox, 'ymin').text = str(y_min)
        ET.SubElement(bndbox, 'xmax').text = str(x_max)
        ET.SubElement(bndbox, 'ymax').text = str(y_max)

    tree = ET.ElementTree(annotation)
    tree.write(xml_file)

def convert_folder(yolo_folder, output_folder, image_width, image_height, classes):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(yolo_folder):
        if filename.endswith('.txt'):
            yolo_file = os.path.join(yolo_folder, filename)
            xml_file = os.path.join(output_folder, filename.replace('.txt', '.xml'))
            convert_yolo_to_xml(yolo_file, xml_file, image_width, image_height, classes)

# Example usage
yolo_folder = 'D:/Goodluck_babies_VisAlgae23/dataset/train/labels'  # Path to folder containing YOLO format files
output_folder = 'augmentation/xml_labels'  # Output directory for XML files
image_width, image_height = 1904, 608  # Dimensions of your images
classes = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6']  # List of class names

convert_folder(yolo_folder, output_folder, image_width, image_height, classes)
