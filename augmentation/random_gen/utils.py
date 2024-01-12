import os
import shutil
import random
import numpy as np

import torch
from torch.utils.data import random_split, Dataset

from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.tv_tensors import BoundingBoxes
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import functional as F

import cv2 as cv

from matplotlib import pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def train_test_split_dir(test_size, validate_size):
    if os.path.exists(os.path.join("datasets", "algae_dataset")):
        return
    images_path = os.path.join("train", "images")
    labels_path = os.path.join("train", "labels")
    
    images = os.listdir(images_path)
    dataset_split = random_split(images, [1-test_size-validate_size, validate_size, test_size])
    
    set_types = ["train", "validate", "test"]
    
    for set_type in set_types:
        for fd in ["images", "labels"]:
            os.makedirs(os.path.join("datasets", "algae_dataset", set_type, fd))
    
    for i in range(len(dataset_split)):
        for image in dataset_split[i]:
            label = image.split(".")[0] + ".txt"
            shutil.copy(os.path.join(images_path, image), 
                        os.path.join("datasets", "algae_dataset", set_types[i], "images"))
            shutil.copy(os.path.join(labels_path, label), 
                        os.path.join("datasets", "algae_dataset", set_types[i], "labels"))



class AlgaeDataset(Dataset):
    def __init__(self, root, transforms = None) -> None:
        self.root = root
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annots = list(sorted(os.listdir(os.path.join(root, "labels"))))
        self.label_names = ["background", "Platymonas", "Chlorella", "Dunaliella salina",
                            "Zooxanthella", "Porphyridium", "Haematococcus pluvialis"]
    
    def __getitem__(self, index) -> tuple:
        self.image_path = os.path.join(self.root, "images", self.images[index])
        self.annot_path = os.path.join(self.root, "labels", self.annots[index])
        image = read_image(self.image_path)
        _, image_h, image_w = image.size()
        
        # Parse annotations
        annot_file = open(self.annot_path)
        lines = annot_file.read().splitlines()
        num_objs = len(lines)
        boxes, labels = torch.zeros((num_objs, 4)), torch.zeros((num_objs), dtype=torch.int16)

        for index_line, line in enumerate(lines):
            label, x_center, y_center, w, h = line.split()
            xmin = image_w * (float(x_center) - float(w)/2)
            ymin = image_h * (float(y_center) - float(h)/2)
            xmax = image_w * (float(x_center) + float(w)/2)
            ymax = image_h * (float(y_center) + float(h)/2)

            boxes[index_line, :] = torch.tensor((xmin, ymin, xmax, ymax))
            labels[index_line] = int(label) + 1

        area = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        annot = {}
        annot["boxes"] = BoundingBoxes(boxes, format="XYXY", 
                                       canvas_size=(image_h, image_w))
        annot["labels"] = labels
        annot["image_id"] = index
        annot["area"] = area
        annot["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, annot = self.transforms(image, annot)
        
        return image, annot
    
    def to_ultralytics_form():
        pass


    
    def plot_image_annot(self, image, annot):
        label_strs = [self.label_names[label] for label in annot["labels"]]
        result = draw_bounding_boxes(image=image, boxes=annot["boxes"], labels=label_strs, font="arial.ttf", font_size=30)
        show(result)



class RandomGenAugmenter(torch.nn.Module):
    def __init__(self,
                 src_root,
                 label_dist=torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                 max_objs=4,
                 edge_constraint=0.2,
                 label_names=["background", "Platymonas", "Chlorella", "Dunaliella salina",
                              "Zooxanthella", "Porphyridium", "Haematococcus pluvialis"]):
        """
        Args:
            label_dist (tensor, optional): sampling prob. dist. for each label
            Defaults to torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).
            max_objs (int, optional): _description_. Defaults to 4.
            edge_constraint (float, optional): _description_. Defaults to 0.2.
        """
        super(RandomGenAugmenter, self).__init__()
        self.src_root = src_root
        self.label_dist = label_dist
        self.max_objs = max_objs
        self.edge_constraint = edge_constraint
        self.label_names = label_names
    
    def forward(self, new_id) -> tuple:
        """Generate new sample

        Args:
            image (torch.tensor): input image
            annot (dict): input annot

        Returns:
            tuple: image, annot
        """ 
        # Retrieve a background
        path_bg = os.path.join(self.src_root, "backgrounds")
        image_bg = read_image(os.path.join(path_bg, random.choice(os.listdir(path_bg))))
        _, image_bg_h, image_bg_w = image_bg.size()
        
        # Determine algae num
        num_objs = random.randint(1, self.max_objs)
        algae_pos_tensor = (1-self.edge_constraint) * torch.rand((num_objs, 2))
        
        # Retrieve algaes
        chosen_labels = torch.multinomial(self.label_dist, num_objs, replacement=True)
        chosen_label_names = [self.label_names[label+1] for label in chosen_labels.tolist()]
        print("Randomly chosen classes:", chosen_label_names)
        
        boxes, labels = (torch.zeros((num_objs, 4)), 
                         torch.zeros(num_objs, dtype=torch.int16))
        for index, label_name in enumerate(chosen_label_names):
            path_alg = os.path.join(self.src_root, "algaes", label_name)
            image_algae = read_image(os.path.join(path_alg, random.choice(os.listdir(path_alg))))
            _, image_algae_h, image_algae_w = image_algae.size()

            xmin, ymin = (algae_pos_tensor[index, :] * 
                          torch.tensor([image_bg_w, image_bg_h])).tolist()
            boxes[index, :] = torch.tensor([xmin, ymin, 
                                            xmin+image_algae_w, ymin+image_algae_h])
            labels[index] = chosen_labels[index] + 1

            # Blend algaes into the background
            image_bg = self.blend_bg_algae(image_bg, image_algae, (xmin, ymin))
        
        image_new = image_bg
        #show(image_bg)

        # Generate annotation
        area = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        annot_new = {}
        annot_new["boxes"] = BoundingBoxes(boxes, format="XYXY", 
                                           canvas_size=(image_bg_h, image_bg_w))
        annot_new["labels"] = labels
        annot_new["image_id"] = new_id
        annot_new["area"] = area
        annot_new["iscrowd"] = iscrowd

        return image_new, annot_new

    @staticmethod
    def blend_bg_algae(image_bg: torch.tensor, image_algae: torch.tensor, pos) -> torch.tensor:
        """Blend algaes into backgrounds smoothly

        Args:
            image_bg (torch.tensor): background image
            image_algae (torch.tensor): algae image
            pos (tuple): xmin, ymin

        Returns:
            torch.tensor: Blended image
        """
        
        xmin_sample, ymin_sample = pos
        _, image_algae_h, image_algae_w = image_algae.size()
        image_bg_np = image_bg[:, 
                    int(ymin_sample):int(ymin_sample)+image_algae_h, 
                    int(xmin_sample):int(xmin_sample)+image_algae_w].cpu().numpy().transpose(1, 2, 0)
        image_algae_np = image_algae.cpu().numpy().transpose(1, 2, 0)

        # Convert the images to uint8 format for seamlessClone
        image_algae_np = (image_algae_np * 255).astype(np.uint8)
        image_bg_np = (image_bg_np * 255).astype(np.uint8)

        # Create a mask for the region
        mask = (image_algae_np > 0).astype(np.uint8) * 255

        # Convert the images to BGR format for seamlessClone
        image_algae_np = cv.cvtColor(image_algae_np, cv.COLOR_RGB2BGR)
        image_bg_np = cv.cvtColor(image_bg_np, cv.COLOR_RGB2BGR)

        # Perform seamless cloning
        blended_image = cv.seamlessClone(image_algae_np, image_bg_np, mask, 
                                        (int(image_algae_w / 2), 
                                        int(image_algae_h / 2)), 
                                        cv.NORMAL_CLONE)
        blended_image = cv.cvtColor(blended_image, cv.COLOR_BGR2RGB) * 255
        blended_image_tensor = torch.from_numpy(blended_image.transpose(2, 0, 1)).float()
        image_bg[:, 
                int(ymin_sample):int(ymin_sample) + image_algae_h,
                int(xmin_sample):int(xmin_sample) + image_algae_w] = blended_image_tensor
        return image_bg
    



if __name__ == "__main__":
    algae_dataset = AlgaeDataset("train")
    sample = algae_dataset[4]
    print(sample[1])
    algae_dataset.plot_image_annot(sample[0], sample[1])