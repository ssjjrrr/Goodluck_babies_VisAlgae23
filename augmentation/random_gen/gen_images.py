from utils import AlgaeDataset, RandomGenAugmenter
import os
from PIL import Image
import numpy as np
import cv2

#随机读取数据集中的图片
path_dataset = os.path.join("dataset", "train")
algae_dataset = AlgaeDataset(path_dataset)

# image, annot = algae_dataset[0]
# algae_dataset.plot_image_annot(image, annot)

def gen_img(nums, save_dir, source, save=False):
    for i in range(nums):
        
        img_id = i + 1201
        augmenter = RandomGenAugmenter(src_root=source)
        image_new, annot_new = augmenter(img_id)
        
        #查看前两张
        if i <= 1:
            algae_dataset.plot_image_annot(image_new, annot_new)
        #获取图像宽高，类别和边界框，转成YOLO格式
        img_new = image_new.permute(1, 2, 0)
        image = Image.fromarray(img_new.numpy(), 'RGB')
        width, height = image.size
        # print(width, height)
        bboxes, classes = annot_new['boxes'][:], annot_new['labels'][:] - 1
        if save:
            try:
                save_img_path = os.path.join(save_dir, f"images/{img_id}.jpg")
                image.save(save_img_path)
                save_txt_path = os.path.join(save_dir, f"labels/{img_id}.txt")
                with open(save_txt_path, "w") as f:
                    for bbox, class_name in zip(bboxes, classes):
                        bbox = xyxy2xywh(bbox, width, height)
                        # print(class_name, bbox)

                        class_name = class_name.item()
                        bbox = [b.item() for b in bbox]
                        data_str = f"{class_name} " + " ".join([f"{b:f}" for b in bbox]) + "\n"
                        f.write(data_str)
                print(f"image {img_id} saved at {save_dir}")
            except RuntimeError:
                print("save failed!")
            
                

def xyxy2xywh(bbox_xyxy, width, height):
    x_min, y_min, x_max, y_max = bbox_xyxy
    x_c = (x_min + x_max) / (2 * width)
    y_c = (y_min + y_max) / (2 * height)
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height
    return [x_c, y_c, w, h]



if __name__ == "__main__":
    img_nums = 400
    save_dir = 'D:/Goodluck_babies_VisAlgae23/aug_dataset/main'

    path_gen_src = 'D:/Goodluck_babies_VisAlgae23/labelled_src'

    gen_img(img_nums, save_dir, path_gen_src, save=True)


