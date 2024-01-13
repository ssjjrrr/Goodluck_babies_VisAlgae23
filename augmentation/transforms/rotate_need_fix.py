import math
import numpy as np
import cv2
import os

def rotate_img_bbox(img, bboxes, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_c, y_c, w, h],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        # 旋转图像
        width = img.shape[1]
        height = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * height) + abs(np.cos(rangle) * width)) * scale
        nh = (abs(np.cos(rangle) * height) + abs(np.sin(rangle) * width)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - width) * 0.5, (nh - height) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
 
        # 矫正bbox坐标
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            x_c, y_c, w, h = bbox
            xmin = (x_c - w / 2) * width
            xmax = (x_c + w / 2) * width
            ymin = (y_c - h / 2) * height
            ymax = (y_c + h / 2) * height
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.float32)
            # 得到旋转后的坐标
            rxmin, rymin, rw, rh = cv2.boundingRect(concat)
            rx_c = (rxmin + rw / 2) / width
            ry_c = (rymin + rh / 2) / height
            rw = rw / width
            rh = rh / height
            # 加入list中
            rot_bboxes.append([rx_c, ry_c, rw, rh])
 
        return rot_img, rot_bboxes


images_folder = 'D:/Goodluck_babies_VisAlgae23/aug_dataset/main/images'
labels_folder = 'D:/Goodluck_babies_VisAlgae23/aug_dataset/main/labels'
output_images_folder = 'D:/Goodluck_babies_VisAlgae23/aug_dataset/rotated/images'
output_labels_folder = 'D:/Goodluck_babies_VisAlgae23/aug_dataset/rotated/labels'

# 确保输出文件夹存在
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)


for image_file in os.listdir(images_folder):
    if image_file.endswith('.jpg'):
        # 构建完整的文件路径
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, image_file.replace('.jpg', '.txt'))

        # 读取图像和标签
        img = cv2.imread(image_path)
        class_ids = []
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = parts[1:]
                class_ids.append(class_id)
                bboxes.append([x_center, y_center, width, height])

        # 调用函数进行旋转
        rot_img, rot_bboxes = rotate_img_bbox(img, bboxes)

        # 保存旋转后的图像和标签
        cv2.imwrite(os.path.join(output_images_folder, image_file), rot_img)
        with open(os.path.join(output_labels_folder, image_file.replace('.jpg', '.txt')), 'w') as f:
            for class_id, bbox in zip(class_ids, rot_bboxes):
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            print(f"image {image_file} is rotated")