# gif extracted from http://www.wolfdynamics.com/wiki/tut_2D_cylinder.pdf
import os
import torch
import random
import pprint
from PIL import Image
import numpy as np

class GetLoader_pic_pic(torch.utils.data.Dataset):
    def __init__(self, input_path, output_path, transform=None):
        # read input output path
        self.input_path = input_path
        self.input_files = os.listdir(input_path)  # list all input files
        self.output_path = output_path
        self.label_files = os.listdir(output_path)
        self.transforms = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        # read file with index
        input_img_path = os.path.join(self.input_path, self.input_files[index])
        input_img = Image.open(input_img_path)
        input_img = input_img.convert("RGB")
        output_img_path = os.path.join(self.output_path, self.label_files[index])
        output_img = Image.open(output_img_path)
        output_img = output_img.convert("RGB")
        if self.transforms:
            input_img = self.transforms(input_img)
            output_img = self.transforms(output_img)
        return (input_img, output_img)

class GetLoader_pic(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        # read input output path
        self.paths = path
        self.files = os.listdir(path)  # list all input files
        self.transforms = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # read file with index
        img_path = os.path.join(self.paths, self.files[index])
        img = Image.open(img_path)
        img = np.array(img.convert("L")).reshape(1, -1)
        if self.transforms:
            img = self.transforms(img)
        img.type(torch.FloatTensor)
        return img

def load_img_as_matrix(path):
    files = os.listdir(path)  # list all input files
    files.sort()
    matrix = []
    for file in files:
        img_path = os.path.join(path, file)
        img = Image.open(img_path)
        img = np.array(img.convert("L")).reshape(-1)/255 # [392,490] to 
        matrix.append(img)
    return np.array(matrix).astype("float32")