import cv2 
import numpy as np
import random
import os

def rotate_random(img, bboxes):
    
    #设置角度为random
    angle = random.uniform(0,10)
    
    # 获取旋转矩阵
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)


    sin_cos = rot_mat[:,1]  #从旋转矩阵中获取sin与cos
    size_convert = np.mat([[sin_cos[1],sin_cos[0]],[sin_cos[0],sin_cos[1]]])    #计算形状变化的矩阵（可用于bbox的w，h）
    alg_size = img.shape[1::-1]
    out_size = np.dot(alg_size , size_convert)
    w_convert = int(out_size[:,0])  #旋转后的宽
    h_convert = int(out_size[:,1])  #旋转后的高

    # 旋转藻类图像
    rotated_algae = cv2.warpAffine(img, rot_mat, (w_convert,h_convert), flags=cv2.INTER_NEAREST)   #borderMode=cv.BORDER_REPLICATE 可以加

    # 创建旋转后的图像掩码
    mask = np.zeros(rotated_algae.shape[:2], dtype=np.uint8)
    mask[rotated_algae.any(axis=-1)] = 255  #rotated_algae为旋转后的藻类图像
    
    #进行bbox的操作
    rot_bboxes = list()
    for bbox in bboxes:
        xmin = float(bbox[0])
        ymin = float(bbox[1])
        xmax = float(bbox[2])
        ymax = float(bbox[3])
        point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
        # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))
        # 改变array类型
        concat = concat.astype(np.int32)
        # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        rx_min = rx
        ry_min = ry
        rx_max = rx + rw
        ry_max = ry + rh
        # 加入list中
        rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

    return rotated_algae, rot_bboxes


