from __future__ import absolute_import, division, print_function
import os
import comet_ml
from utils import utils as util
import numpy as np
import time
import json
import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
torch.manual_seed(100)
torch.cuda.manual_seed(100)
from utils.metrics import compute_depth_metrics, Evaluator, compute_shading_metrics, compute_alb_metrics, compute_semantic_metrics
from utils.metrics import compute_normal_metrics, mIoU
from utils.normal_reconstruct import reconstruct
from utils.losses import proportion_good_pixels, average_angular_error
import utils.loss_gradient as loss_g
import utils.losses as loss
from utils.losses import BerhuLoss
from models import *
from models import MultiViT as MVT
# from utils.stanford2d3d import Stanford2D3D
from utils.structured3d import Structured3D

import matplotlib.pyplot as plt

# Helper function to display logged assets in the Comet UI
def display(tab=None):
    experiment = comet_ml.get_global_experiment()
    experiment.display(tab=tab)


semantic_loss = nn.CrossEntropyLoss(reduction='mean')
ssim = loss.SSIMLoss()

def gradient(x):
    gradient_model = loss_g.Gradient_Net()
    g_x, g_y = gradient_model(x)
    return g_x, g_y

def gradient3d(x):
    gradient_model = loss_g.Gradient_Net_3d()
    g_x, g_y = gradient_model(x)
    return g_x, g_y

compute_loss = BerhuLoss()

def loss_behru(input, output):
    gt = input
    pred = output
    G_x, G_y = gradient(gt.float())
    p_x, p_y = gradient(pred)
    loss =compute_loss(input.float(), output) +\
                         compute_loss(G_x, p_x) +\
                         compute_loss(G_y, p_y)
    return loss

def loss_behru3d(input, output):
    gt = input
    pred = output
    G_x, G_y = gradient3d(gt.float())
    p_x, p_y = gradient3d(pred)
    loss =compute_loss(input.float(), output) +\
                         compute_loss(G_x, p_x) +\
                         compute_loss(G_y, p_y)
    return loss
def semantic_loss1(semantic, pred):
    ssloss = semantic_loss(pred, semantic) + loss.dice_coefficient_loss(semantic, pred)
    return ssloss


class Trainer:
    def __init__(self, settings):
        self.settings = settings
        comet_init_config = {
            "api_key":"QNHPKSygOyiOMg3t2DYAE1rBq",
            "project_name": "multivit-depth",
            "workspace": "uzshah"
        }
        if util.get_rank() == 0:
            comet_ml.init(api_key=comet_init_config['api_key'],  
                          workspace=comet_init_config['workspace'],
                          project_name=comet_init_config['project_name']
                         )
        print(self.settings)
        util.init_distributed_mode(self.settings)
        self.device = torch.device(self.settings.device)
    
        # Fix the seed for reproducibility
        seed = util.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True
    
        num_tasks = util.get_world_size()
        global_rank = util.get_rank()
    
        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)
        
        train_dataset = Structured3D("Data/full", "", num_classes = 41, target = settings.target, 
                                     height=settings.input_size[0], width=settings.input_size[1],
                                      disable_color_augmentation=True, disable_LR_filp_augmentation=False, 
                                     disable_yaw_rotation_augmentation=False, is_training = True)
        sampler_train = torch.utils.data.DistributedSampler(
                train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        self.train_loader = DataLoader(train_dataset, self.settings.batch_size,  sampler= sampler_train, 
                                       num_workers=self.settings.num_workers, pin_memory=self.settings.pin_mem, drop_last=True)
        
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs
        val_dataset = Structured3D("Data/full", "", num_classes = 41, target = settings.target, 
                                   height=settings.input_size[0], width=settings.input_size[1],
                                    disable_color_augmentation=True, disable_LR_filp_augmentation=True, 
                                    disable_yaw_rotation_augmentation=True, is_training = False)
        sampler_test = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, self.settings.batch_size,  sampler= sampler_test,
                                     num_workers=self.settings.num_workers, pin_memory=self.settings.pin_mem, drop_last=True)

        self.settings.num_classes = 41
        self.model = MVT.MultiViT(num_classes = self.settings.num_classes, target = self.settings.target)
        self.model.to(self.device)
        model_without_ddp = self.model
        if self.settings.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.settings.gpu], find_unused_parameters=True)
            model_without_ddp = self.model.module
        
        self.parameters_to_train = list(self.model.parameters())

        self.optimizer = optim.Adam(self.parameters_to_train, self.settings.learning_rate)

        if self.settings.load_weights_dir is not None:
            self.load_model()

        ## Print Parameters 
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")
        print("Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)
        self.evaluator = Evaluator(self.settings)

        self.writers = {}
        for mode in ["train", "val"]:
            if util.get_rank() == 0 :
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path,mode),comet_config={"disabled": False})
            else:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
    

        self.save_settings()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        # self.validate()
        for self.epoch in range(self.settings.num_epochs):
            self.train_one_epoch()
            self.validate()
            if (self.epoch + 1) % self.settings.save_frequency == 0:
                self.save_model()

    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()

        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))

        for batch_idx, inputs in enumerate(pbar):

            outputs, losses = self.process_batch(inputs, True)

            self.optimizer.zero_grad(),
            losses["loss"].backward()
            self.optimizer.step()

            # log less frequently after the first 1000 steps to save time & disk space
            early_phase = batch_idx % self.settings.log_frequency == 0 and self.step < 1000
            late_phase = self.step % 1000 == 0
            errors = []
            if early_phase or late_phase:
                if self.settings.target == "all" or self.settings.target == "depth":
                    errors.extend(compute_depth_metrics(inputs["depth"].detach(), outputs["pred_depth"].detach(), inputs["depth_mask"]))
                if self.settings.target == "all" or self.settings.target == "shading":
                    errors.extend(compute_shading_metrics(inputs["shading"].detach(), outputs["pred_shading"].detach(), inputs["val_mask"]))
                if self.settings.target == "all" or self.settings.target == "albedo":
                    errors.extend(compute_alb_metrics(inputs["albedo"].detach(), outputs["pred_albedo"].detach(), inputs["val_mask"]))
                if self.settings.target == "all" or self.settings.target == "normal":
                    errors.extend(compute_normal_metrics(inputs["normal"].detach(), outputs["pred_normal"].detach(), inputs["val_mask"]))
                if self.settings.target == "all" or self.settings.target == "semantic":
                    errors.extend(compute_semantic_metrics(inputs["semantic"].detach(), outputs["pred_semantic"].detach()))
                
                for i, key in enumerate(self.evaluator.metrics.keys()):
                    losses[key] = np.array(errors[i].cpu())
                self.log("train", inputs, outputs, losses)

            self.step += 1

    def process_batch(self, inputs, is_training = True):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
    
        losses = {}
    
        equi_inputs = inputs["rgb"]*inputs["val_mask"]
    
        outputs = self.model(equi_inputs)
        if self.settings.target == "all" or self.settings.target == "depth":
            # Depth loss
            inputs["depth"] = inputs["depth"] * inputs["depth_mask"]
            outputs["pred_depth"] = outputs["pred_depth"] * inputs["depth_mask"]
            losses["depth_loss"] = loss_behru(inputs["depth"], outputs["pred_depth"])
        if self.settings.target == "all" or self.settings.target == "shading":
            ## Shading loss
            inputs["shading"] = inputs["shading"] * inputs["val_mask"]
            outputs["pred_shading"] = outputs["pred_shading"] * inputs["val_mask"]
            losses["shading_loss"] = loss_behru(inputs["shading"], outputs["pred_shading"])
            
        if self.settings.target == "all" or self.settings.target == "albedo":
            ## Albedo loss
            inputs["albedo"] = inputs["albedo"] * inputs["val_mask"]
            outputs["pred_albedo"] = outputs["pred_albedo"] * inputs["val_mask"]
            losses["albedo_loss"] = loss_behru3d(inputs["albedo"], outputs["pred_albedo"])
        if self.settings.target == "all":
            ## ALbedo & noraml loss
            losses['joint_loss'] = ssim(equi_inputs, inputs["shading"] * inputs["albedo"]).mean()
            
        if self.settings.target == "all" or self.settings.target == "normal":
            ## Normal loss
            inputs["normal"] = inputs["normal"] * inputs["val_mask"]
            outputs["pred_normal"] = outputs["pred_normal"] * inputs["val_mask"]
            losses["normal_loss"] = loss_behru3d(inputs["normal"], outputs["pred_normal"])
            n_hat = 2*(outputs["pred_normal"])-1
            estimated_normal = reconstruct(inputs["depth"]).float().to(self.device)
            losses["re_normal_loss"] = loss_behru3d(n_hat, estimated_normal)
            
        if self.settings.target == "all" or self.settings.target == "semantic":
            ## semantic loss
            inputs["semantic"]  = inputs["semantic"] * inputs["val_mask"]
            outputs["pred_semantic"] = outputs["pred_semantic"] * inputs["val_mask"]
            losses['semantic_loss'] = semantic_loss1(inputs["semantic"], outputs["pred_semantic"])
            losses['mIoUloss'] = 1 - mIoU(inputs["semantic"], outputs["pred_semantic"])
        if is_training:
            losses['loss'] = 0
            if self.settings.target == "all" or self.settings.target == "depth":
                losses['loss'] += losses["depth_loss"]
            if self.settings.target == "all" or self.settings.target == "shading":
                losses['loss'] += losses["shading_loss"]
            if self.settings.target == "all" or self.settings.target == "albedo":
                losses['loss'] += losses["albedo_loss"]
            if self.settings.target == "all" or self.settings.target == "normal":
                losses['loss'] += losses["normal_loss"] + (0.5*losses["re_normal_loss"])
            if self.settings.target == "all" or self.settings.target == "semantic":
                losses['loss'] += losses["semantic_loss"]
            if self.settings.target == "all":
                losses['loss'] += (0.2 * losses['joint_loss']) 
            
        else:
            losses['loss'] = 0
            if self.settings.target == "all" or self.settings.target == "depth":
                losses['loss'] += losses["depth_loss"]
            if self.settings.target == "all" or self.settings.target == "shading":
                losses['loss'] += losses["shading_loss"]
            if self.settings.target == "all" or self.settings.target == "albedo":
                losses['loss'] += losses["albedo_loss"]
            if self.settings.target == "all" or self.settings.target == "normal":
                losses['loss'] += losses["normal_loss"]
            if self.settings.target == "all" or self.settings.target == "semantic":
                losses['loss'] += losses['mIoUloss']
        return outputs, losses

    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs, losses = self.process_batch(inputs, False)
                if self.settings.target == "all":
                    self.evaluator.compute_eval_metrics(inputs["depth"].detach(), outputs["pred_depth"].detach(), \
                                               inputs["shading"].detach(), outputs["pred_shading"].detach(),\
                                               inputs["albedo"].detach(), outputs["pred_albedo"].detach(), \
                                               inputs["normal"].detach(), outputs["pred_normal"].detach(), \
                                               inputs["semantic"].detach(), outputs["pred_semantic"].detach(), \
                                               dmask=inputs["depth_mask"], mask = inputs["val_mask"])
                if self.settings.target == "depth":
                   self.evaluator.compute_eval_metrics(gt_depth=inputs["depth"].detach(), pred_depth=outputs["pred_depth"].detach(), \
                                               dmask=inputs["depth_mask"], mask = inputs["val_mask"]) 
                if self.settings.target == "shading":
                   self.evaluator.compute_eval_metrics(gt_shading=inputs["shading"].detach(), pred_shading=outputs["pred_depth"].detach(), \
                                             mask = inputs["val_mask"]) 
                if self.settings.target == "normal":
                   self.evaluator.compute_eval_metrics(gt_normal=inputs["normal"].detach(), pred_normal=outputs["pred_normal"].detach(), \
                                             mask = inputs["val_mask"]) 
                if self.settings.target == "albedo":
                   self.evaluator.compute_eval_metrics(gt_albedo=inputs["albedo"].detach(), pred_albedo=outputs["pred_albedo"].detach(), \
                                             mask = inputs["val_mask"]) 
                if self.settings.target == "semantic":
                   self.evaluator.compute_eval_metrics(gt_semantic=inputs["albedo"].detach(), pred_semantic=outputs["pred_albedo"].detach(), \
                                             mask = inputs["val_mask"]) 
        self.evaluator.print()

        for i, key in enumerate(self.evaluator.metrics.keys()):
            # print(key)
            losses[key] = np.array(self.evaluator.metrics[key].avg.cpu())
        self.log("val", inputs, outputs, losses)
        del inputs, outputs, losses

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        if self.settings.target == "semantic":
            outputs["pred_semantic"] = F.softmax(outputs["pred_semantic"], dim=1)
            outputs["pred_semantic"] = torch.argmax(outputs["pred_semantic"], dim=1).unsqueeze(1)
            inputs["semantic"] = torch.argmax(inputs["semantic"], dim=1).unsqueeze(1)
        # print(gt_semantic.shape)
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(2, self.settings.batch_size)):  # write a maxmimum of four images
            writer.add_image("rgb/{}".format(j), inputs["rgb"][j].data, self.step)
            if self.settings.target == "all" or self.settings.target == "depth":
                writer.add_image("gt_depth/{}".format(j),
                                 inputs["depth"][j].data/inputs["depth"][j].data.max(),self.step)
                writer.add_image("pred_depth/{}".format(j),
                                 outputs['pred_depth'][j].data/outputs['pred_depth'][j].data.max(), self.step)
            if self.settings.target == "all" or self.settings.target == "shading":
                writer.add_image("gt_shading/{}".format(j),
                                 inputs['shading'][j].data/inputs['shading'][j].data.max(),self.step)
                writer.add_image("pred_shading/{}".format(j),
                                 outputs['pred_shading'][j].data/outputs['pred_shading'][j].data.max(), self.step)
            if self.settings.target == "all" or self.settings.target == "albedo":
                writer.add_image("gt_albedo/{}".format(j),
                                inputs['albedo'][j].data/inputs['albedo'][j].data.max(),self.step)
                writer.add_image("pred_albedo/{}".format(j),
                                 outputs['pred_albedo'][j].data/outputs['pred_albedo'][j].data.max(), self.step)
            if self.settings.target == "all" or self.settings.target == "normal":
                writer.add_image("gt_normal/{}".format(j),
                                 inputs['normal'][j].data/inputs['normal'][j].data.max(),self.step)
                writer.add_image("pred_normal/{}".format(j),
                                 outputs['pred_normal'][j].data/outputs['pred_normal'][j].data.max(), self.step)
            if self.settings.target == "all" or self.settings.target == "semantic":
                writer.add_image("gt_semantic/{}".format(j),
                                 inputs['semantic'][j].data/inputs['semantic'][j].data.max(),self.step)
                writer.add_image("pred_semantic/{}".format(j),
                                 outputs['pred_semantic'][j].data/outputs['pred_semantic'][j].data.max(), self.step)
            
            
    def save_settings(self):
        """Save settings to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.settings.__dict__.copy()

        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model"))
        to_save = self.model.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model from disk
        """
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), \
            "Cannot find folder {}".format(self.settings.load_weights_dir)
        print("loading model from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.settings.load_weights_dir, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

