import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import json


import cv2
import random
from torch.utils import data


class ADE(Dataset):
    def __init__(self, args, folders, transforms, se_transforms, split):
        self.folders = folders
        self.rgb_folder = "images"
        self.semantic_folder = "annotations"
        self.data = self.load_data(split)
        self.transform = transforms
        self.semantic = se_transforms
        self.args = args

    def load_data(self, split):
        rgb_path = os.path.join(self.folders, self.rgb_folder, split)
        semantic_path = os.path.join(self.folders, self.semantic_folder, split)

        rgb_files = sorted(os.listdir(rgb_path))
        semantic_files = sorted(os.listdir(semantic_path))

        data = [(os.path.join(rgb_path, f_rgb), os.path.join(semantic_path, f_semantic))
                for f_rgb, f_semantic in zip(rgb_files, semantic_files)]

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path, semantic_path = self.data[idx]

        rgb_image = Image.open(rgb_path).convert("RGB")
        mask = torch.tensor(np.array(self.semantic(Image.open(semantic_path))), dtype=torch.long)
        num_classes = 151
        one_hot_mask = torch.zeros(num_classes, *mask.shape, dtype=torch.float32)
        semantic_map = one_hot_mask.scatter_(0, mask.unsqueeze(0), 1)

        rgb_image = self.transform(rgb_image)
        
        return {"image": rgb_image, "semantic": semantic_map}

class MergedDataset(Dataset):
    def __init__(self, args, folders, split, transforms, target_transforms, se_transforms, alb_transforms, sh_transforms):
        self.folders = folders
        self.rgb_folder = "rgb"
        self.depth_folder = "depth"
        self.semantic_folder = "semantic"
        self.albedo_folder = "albedo"
        self.shading_folder = "shading"
        self.normal_folder = "normal"
        self.metric_scale = 512.0
        self.data = []  # List to store tuples (image_path, label)
        self.transform = transforms
        self.target = target_transforms
        self.semantic = se_transforms
        self.abledo_trans = alb_transforms
        self.shading_trans = sh_transforms
        self.args = args

        for folder in self.folders:
            rgb = os.path.join(folder, self.rgb_folder)
            depth = os.path.join(folder, self.depth_folder)
            semantic = os.path.join(folder, self.semantic_folder)
            albedo = os.path.join(folder, self.albedo_folder)
            shading = os.path.join(folder, self.shading_folder)
            normal = os.path.join(folder, self.normal_folder)
            for f_rgb in split:
                image_path, depth_path = os.path.join(rgb, f_rgb), os.path.join(depth, f_rgb)
                semantic_path, albedo_path = os.path.join(semantic, f_rgb), os.path.join(albedo, f_rgb)
                shading_path, normal_path = os.path.join(shading, f_rgb), os.path.join(normal, f_rgb)
                self.data.append((image_path, depth_path, semantic_path, albedo_path, shading_path, normal_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path, depth_path, semantic_path, albedo_path, shading_path, normal_path = self.data[idx]
        #print(normal_path, depth_path) 
        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_map = Image.open(depth_path)
        albedo_map = Image.open(albedo_path).convert("RGB")
        shading_map = Image.open(shading_path)
        normal_map = Image.open(normal_path).convert("RGB")
        mask = torch.tensor(np.array(self.semantic(Image.open(semantic_path))), dtype=torch.long)
        num_classes = 41
        one_hot_mask = torch.zeros(num_classes, *mask.shape, dtype=torch.float32)
        semantic_map = one_hot_mask.scatter_(0, mask.unsqueeze(0), 1)

        rgb_image = self.transform(rgb_image)
        depth_map = self.target(depth_map)
        depth_map[depth_map>17] = 17
        albedo_map = self.abledo_trans(albedo_map)
        normal_map = self.abledo_trans(normal_map)
        shading_map = self.shading_trans(shading_map)

        if self.args.option == "all":
            return {"image": rgb_image, "depth": depth_map, "semantic": semantic_map, "albedo": albedo_map, "shading": shading_map, \
                   "normal": normal_map}
        elif self.args.option == "depth":
            return {"image": rgb_image, "depth": depth_map}
        elif self.args.option == "semantic":
            return {"image": rgb_image, "semantic": semantic_map}
        else:
            return {"image": rgb_image, "albedo": albedo_map, "shading": shading_map}
class s2d3d(Dataset):
    def __init__(self, args, folders, transforms, target_transforms, se_transforms, alb_transforms, sh_transforms):
        self.folders = folders
        self.rgb_folder = "rgb"
        self.depth_folder = "depth"
        self.albedo_folder = "albedo"
        self.shading_folder = "shading"
        self.semantic_folder = "semantic"
        self.normal_folder = "normal"
        self.metric_scale = 512.0
        self.data = []  # List to store tuples (image_path, label)
        self.transform = transforms
        self.target = target_transforms
        self.semantic = se_transforms
        self.abledo_trans = alb_transforms
        self.sh_trans = sh_transforms
        self.args = args
        self.max_depth_meters = 10.0
        self.dataset_path = "./Data/s2d3d"
        
        with open(os.path.join(self.dataset_path, 'assets/semantic_labels.json'), 'r', encoding='utf8') as f:
            id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
        with open(os.path.join(self.dataset_path, 'assets/name2label.json'), 'r', encoding='utf8') as f:
            name2id = json.load(f)
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)
        for folder in self.folders:
            rgb = os.path.join(self.dataset_path, folder, self.rgb_folder)
            depth = os.path.join(self.dataset_path, folder, self.depth_folder)
            semantic = os.path.join(self.dataset_path, folder, self.semantic_folder)
            normal = os.path.join(self.dataset_path, folder, self.normal_folder)
            albedo = os.path.join(self.dataset_path, folder, self.albedo_folder)
            shading = os.path.join(self.dataset_path, folder, self.shading_folder)
            for f_rgb, f_rgb1, f_rgb2, f_rgb3, f_rgb4, f_rgb5 in zip(sorted(os.listdir(rgb)), sorted(os.listdir(semantic)), \
                                                     sorted(os.listdir(depth)), sorted(os.listdir(normal)), \
                                                     sorted(os.listdir(albedo)), sorted(os.listdir(shading)),):
                image_path, depth_path = os.path.join(rgb, f_rgb), os.path.join(depth, f_rgb2)
                semantic_path = os.path.join(semantic, f_rgb1)
                normal_path = os.path.join(normal, f_rgb3)
                albedo_path = os.path.join(albedo, f_rgb4)
                shading_path = os.path.join(shading, f_rgb5)
                self.data.append((image_path, depth_path, semantic_path, normal_path, albedo_path, shading_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path, depth_path, semantic_path, normal_path, albedo_path, shading_path = self.data[idx]
        
        rgb_image = Image.open(rgb_path).convert("RGB").crop((0, 310, 4096, 2048-310))
        depth_map = Image.open(depth_path).crop((0, 310, 4096, 2048-310))
        normal_map = Image.open(normal_path).convert("RGB").crop((0, 310, 4096, 2048-310))
        mask = Image.open(semantic_path).crop((0, 310, 4096, 2048-310))
        albedo_map = Image.open(albedo_path).convert("RGB")
        shading_map = Image.open(shading_path)

        
        mask = _color2id(mask, rgb_image, self.id2label)
        mask = torch.tensor(np.array(self.semantic(mask)), dtype=torch.long)
        num_classes = 14
        one_hot_mask = torch.zeros(num_classes, *mask.shape, dtype=torch.float32)
        semantic_map = one_hot_mask.scatter_(0, mask.unsqueeze(0), 1)

        rgb_image = self.transform(rgb_image)
        depth_map = self.target(depth_map)
        depth_map[depth_map > self.max_depth_meters+1] = self.max_depth_meters+1
        normal_map = self.abledo_trans(normal_map)
        albedo_map = self.abledo_trans(albedo_map)
        shading_map = self.sh_trans(shading_map)
        return {"image": rgb_image, "depth": depth_map, "semantic": semantic_map, "albedo": albedo_map, "shading": shading_map, \
                   "normal": normal_map}

def _color2id(mask, img, id2label):
    mask = np.array(mask, np.int32)
    rgb = np.array(img, np.int32)
    unk = (mask[..., 0] != 0)
    mask = id2label[mask[..., 1] * 256 + mask[..., 2]]
    mask[unk] = 0
    mask[rgb.sum(-1) == 0] = 0
    mask -= 1  # 0->255
    mask[mask == 255] = 13
    return Image.fromarray(mask)
def read_file(txt_file_path):
    missing_file = ['scene_01744_351.png', 'scene_01652_387.png', 'scene_00759_141.png']
    # Read the content of the text file
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
    
    # Process each line and generate the list of filenames
    file_list = []
    for line in lines:
        # Split the line based on space
        parts = line.strip().split()
    
        # Combine the parts to form the filename in the desired format
        if len(parts) == 2:
            filename = f"{parts[0]}_{parts[1]}.png"
            if filename in missing_file:
                pass
            else:
                file_list.append(filename)
    return file_list
    
def build_dataset(args):
    transforms, target_transforms, se_transforms, alb_transforms, sh_transforms = build_transform(args)

    if args.dataset == 's3d':
        train, test = read_file(os.path.join(args.folders[0], 'train_clean.txt')), read_file(os.path.join(args.folders[0], 'test.txt'))
        # print(train)
        train_dataset = MergedDataset(args, args.folders, train, transforms, target_transforms, se_transforms, alb_transforms, sh_transforms)
        test_dataset = MergedDataset(args, args.folders, test, transforms, target_transforms, se_transforms, alb_transforms, sh_transforms)
        
        
        # num_train = len(dataset)//200
        # split_idx = int(np.floor(args.validation_split * num_train))
        # indices = list(range(num_train))
        # np.random.shuffle(indices)
        # train_indices, valid_indices = indices[split_idx:], indices[:split_idx]
        # train_dataset = torch.utils.data.Subset(dataset, train_indices)
        # valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
        return train_dataset, test_dataset, 41
        
    elif args.dataset == 's2d3d':
        train_folders = args.folders[:-2]
        test_folders = args.folders[-2:]
        train_dataset = s2d3d(args, train_folders, transforms, target_transforms, se_transforms, alb_transforms, sh_transforms)
        test_dataset = s2d3d(args, test_folders, transforms, target_transforms, se_transforms, alb_transforms, sh_transforms)
        #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        #valid_loader = DataLoader(test_dataset, batch_size=int(args.batch_size * 1.5))
        return train_dataset, test_dataset, 14
    elif args.dataset == 'ade20k':
        train_dataset = ADE(args, "Data/ADEChallengeData2016", transforms, se_transforms, split='training')
        test_dataset = ADE(args, "Data/ADEChallengeData2016", transforms, se_transforms, split='validation')
        return train_dataset, test_dataset


def build_transform(args):
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size[0], args.input_size[1])),
        transforms.ToTensor()
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((args.input_size[0] * args.outsize, args.input_size[1] * args.outsize)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float() / 4000.0)
    ])
    se_transform = transforms.Compose([
        transforms.Resize((args.input_size[0] * args.outsize, args.input_size[1] * args.outsize)),
    ])
    abl_transform = transforms.Compose([
        transforms.Resize((args.input_size[0] * args.outsize, args.input_size[1] * args.outsize)),
        transforms.ToTensor(),
    ])
    shading_transform = transforms.Compose([
        transforms.Resize((args.input_size[0] * args.outsize, args.input_size[1] * args.outsize)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float() / 65535.0)
    ])
    return train_transform, valid_transform, se_transform, abl_transform, shading_transform
