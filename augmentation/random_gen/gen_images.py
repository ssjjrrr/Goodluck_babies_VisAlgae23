from utils import AlgaeDataset, RandomGenAugmenter
import os
from PIL import Image
import numpy as np

annotations = []
save_dir = 'augmentation/random_gen/images_generated'

#随机读取数据集中的图片
path_dataset = os.path.join("dataset", "train")
algae_dataset = AlgaeDataset(path_dataset)

# image, annot = algae_dataset[0]
# algae_dataset.plot_image_annot(image, annot)
for img_id in range(3):
    path_gen_src = os.path.join("augmentation", "random_gen", "gen_augmenter_src")
    augmenter = RandomGenAugmenter(src_root=path_gen_src)
    image_new, annot_new = augmenter(img_id)
    algae_dataset.plot_image_annot(image_new, annot_new)
    img_new = image_new.permute(1, 2, 0)
    image = Image.fromarray(img_new.numpy(), 'RGB')
    print(annot_new)
    # image.show()
    save_file_path = os.path.join(save_dir, f"{img_id}.jpg")
    image.save(save_file_path)