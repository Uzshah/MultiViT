import torch
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import focal_loss
import utils.lovasz_losses as L
from skimage import color
import tqdm
# from torchmetrics import F1
from .normal_reconstruct import reconstruct
from .losses import proportion_good_pixels, average_angular_error
from torchvision.transforms.functional import rgb_to_grayscale
from utils.losses import BerhuLoss
import utils.losses as loss
import utils.gradient_loss as loss_g
from utils.metrics import compute_depth_metrics, Evaluator, compute_shading_metrics, compute_alb_metrics, compute_semantic_metrics
from utils.metrics import compute_normal_metrics, mIoU


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

def loss_behru(gt, pred, input, output):
    G_x, G_y = gradient(gt.float())
    p_x, p_y = gradient(pred)
    loss =compute_loss(input.float(), output) +\
                         compute_loss(G_x, p_x) +\
                         compute_loss(G_y, p_y)
    return loss

def loss_behru3d(gt, pred, input, output):
    G_x, G_y = gradient3d(gt.float())
    p_x, p_y = gradient3d(pred)
    loss =compute_loss(input.float(), output) +\
                         compute_loss(G_x, p_x) +\
                         compute_loss(G_y, p_y)
    return loss


def semantic_loss1(semantic, pred):
    ssloss = semantic_loss(pred, semantic) + loss.dice_coefficient_loss(semantic, pred)
    return ssloss
    
def process_batch(model, inputs, device, is_training = False):
    for key, ipt in inputs.items():
        inputs[key] = ipt.to(device)

    losses = {}

    equi_inputs = inputs["rgb"]*inputs["val_mask"]

    outputs = model(equi_inputs)
    # Depth loss
    inputs["depth"] = inputs["depth"] * inputs["depth_mask"]
    outputs["pred_depth"] = outputs["pred_depth"] * inputs["depth_mask"]
    losses["depth_loss"] = loss_behru(inputs["depth"].float(), outputs["pred_depth"], \
                                      inputs["depth"].float(), outputs["pred_depth"])
    ## Shading loss
    inputs["shading"] = inputs["shading"] * inputs["val_mask"]
    outputs["pred_shading"] = outputs["pred_shading"] * inputs["val_mask"]
    losses["shading_loss"] = loss_behru(inputs["shading"].float(), outputs["pred_shading"], \
                                        inputs["shading"].float(), outputs["pred_shading"])

    ## Albedo loss
    inputs["albedo"] = inputs["albedo"] * inputs["val_mask"]
    outputs["pred_albedo"] = outputs["pred_albedo"] * inputs["val_mask"]
    losses["albedo_loss"] = loss_behru3d(inputs["albedo"].float(), outputs["pred_albedo"], \
                                         inputs["albedo"].float(), outputs["pred_albedo"])
    
    ## ALbedo & noraml loss
    losses['joint_loss'] = ssim(equi_inputs, inputs["shading"] * inputs["albedo"]).mean()
    
    ## Normal loss
    inputs["normal"] = inputs["normal"] * inputs["val_mask"]
    outputs["pred_normal"] = outputs["pred_normal"] * inputs["val_mask"]
    losses["normal_loss"] = loss_behru3d(inputs["normal"].float(), outputs["pred_normal"], \
                                         inputs["normal"].float(), outputs["pred_normal"])
    n_hat = 2*(outputs["pred_normal"])-1
    estimated_normal = reconstruct(inputs["depth"]).float().to(device)
    losses["re_normal_loss"] = loss_behru3d(n_hat, estimated_normal, n_hat, estimated_normal)
    
    ## semantic loss
    inputs["semantic"]  = inputs["semantic"] * inputs["val_mask"]
    outputs["pred_semantic"] = outputs["pred_semantic"] * inputs["val_mask"]
    losses['semantic_loss'] = semantic_loss1(inputs["semantic"], outputs["pred_semantic"])
    losses['mIoUloss'] = 1 - mIoU(inputs["semantic"], outputs["pred_semantic"])
    if is_training:
        losses['loss'] = losses['semantic_loss'] + losses["depth_loss"] + \
                        losses["shading_loss"] + losses["albedo_loss"] + \
                        losses["normal_loss"] + losses["re_normal_loss"] + (0.2 * losses['joint_loss'])
    else:
        losses['loss'] = losses['mIoUloss'] + losses["depth_loss"] + \
                        losses["shading_loss"] + losses["albedo_loss"] + \
                        losses["normal_loss"]
        
    return outputs, losses

def log(args, mode, inputs, outputs, losses, step, writer):
        """Write an event to the tensorboard events file
        """
        outputs["pred_semantic"] = F.softmax(outputs["pred_semantic"], dim=1)
        outputs["pred_semantic"] = torch.argmax(outputs["pred_semantic"], dim=1).unsqueeze(1)
        inputs["semantic"] = torch.argmax(inputs["semantic"], dim=1).unsqueeze(1)
        # print(gt_semantic.shape)
        write = writer[mode]
        for l, v in losses.items():
            write.add_scalar("{}".format(l), v, step)

        for j in range(min(2, args.batch_size)):  # write a maxmimum of four images
            write.add_image("rgb/{}".format(j), inputs["rgb"][j].data, step)
            # writer.add_image("cube_rgb/{}".format(j), inputs["cube_rgb"][j].data, self.step)
            write.add_image("gt_depth/{}".format(j),
                             inputs["depth"][j].data/inputs["depth"][j].data.max(),step)
            write.add_image("pred_depth/{}".format(j),
                             outputs['pred_depth'][j].data/outputs['pred_depth'][j].data.max(), step)
            write.add_image("gt_shading/{}".format(j),
                             inputs['shading'][j].data/inputs['shading'][j].data.max(),step)
            write.add_image("pred_shading/{}".format(j),
                             outputs['pred_shading'][j].data/outputs['pred_shading'][j].data.max(), step)
            write.add_image("gt_albedo/{}".format(j),
                             inputs['albedo'][j].data/inputs['albedo'][j].data.max(),step)
            write.add_image("pred_albedo/{}".format(j),
                             outputs['pred_albedo'][j].data/outputs['pred_albedo'][j].data.max(), step)
            write.add_image("gt_normal/{}".format(j),
                             inputs['normal'][j].data/inputs['normal'][j].data.max(),step)
            write.add_image("pred_normal/{}".format(j),
                             outputs['pred_normal'][j].data/outputs['pred_normal'][j].data.max(), step)
            write.add_image("gt_semantic/{}".format(j),
                             inputs['semantic'][j].data/inputs['semantic'][j].data.max(),step)
            write.add_image("pred_semantic/{}".format(j),
                             outputs['pred_semantic'][j].data/outputs['pred_semantic'][j].data.max(), step)
            
def train_one_epoch(args, model, optimizer, scheduler, train_loader, device, evaluator, writer, step=0):
    model.train()
    pbar = tqdm.tqdm(train_loader, desc="Training Epoch_{}".format(args.epoch), position=0, leave=True)
    for i, inputs in enumerate(pbar):
        errors = []
        outputs, losses = process_batch(model, inputs, device, True)
        # print(losses)
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()
        # log less frequently after the first 1000 steps to save time & disk space
        early_phase = step % args.log_frequency == 0 and step < 10
        late_phase = step % 500 == 0
    
        if early_phase or late_phase:
            
            errors.extend(compute_depth_metrics(inputs["depth"].detach(), outputs["pred_depth"].detach(), inputs["depth_mask"]))
            errors.extend(compute_shading_metrics(inputs["shading"].detach(), outputs["pred_shading"].detach(), inputs["val_mask"]))
            errors.extend(compute_alb_metrics(inputs["albedo"].detach(), outputs["pred_albedo"].detach(), inputs["val_mask"]))
            errors.extend(compute_normal_metrics(inputs["normal"].detach(), outputs["pred_normal"].detach(), inputs["val_mask"]))
            errors.extend(compute_semantic_metrics(inputs["semantic"].detach(), outputs["pred_semantic"].detach()))
            
            for i, key in enumerate(evaluator.metrics.keys()):
                losses[key] = np.array(errors[i].cpu())
    
            log(args, "train", inputs, outputs, losses, step, writer)
            
            loss_str = f'Training_loss: {losses["loss"]:.4f}'
            pbar.set_postfix_str(loss_str)
        step +=1
    
    # scheduler.step()
    return losses, step
    
def evalute(args, model, val_loader, device, evaluator, writer, step):
    model.eval()
    pbar = tqdm.tqdm(val_loader, desc="Validating Epoch_{}".format(args.epoch), position=0, leave=True)
    # pbar = tqdm.tqdm(val_loader)
    # pbar.set_description("Validating Epoch_{}".format(args.epoch))
    with torch.no_grad():
        for i, inputs in enumerate(pbar):
            outputs, losses = process_batch(model, inputs, device)
            
            evaluator.compute_eval_metrics(inputs["depth"].detach(), outputs["pred_depth"].detach(), \
                                           inputs["shading"].detach(), outputs["pred_shading"].detach(),\
                                           inputs["albedo"].detach(), outputs["pred_albedo"].detach(), \
                                           inputs["normal"].detach(), outputs["pred_normal"].detach(), \
                                           inputs["semantic"].detach(), outputs["pred_semantic"].detach(), \
                                           dmask=inputs["depth_mask"], mask = inputs["val_mask"])
            # Update tqdm progress bar with loss information
            loss_str = f"val_loss: {losses['loss']:.4f}"
            pbar.set_postfix_str(loss_str)
            
        evaluator.print()
        for i, key in enumerate(evaluator.metrics.keys()):
            losses[key] = np.array(evaluator.metrics[key].avg.cpu())
        log(args, "val", inputs, outputs, losses, step, writer)
        
    return losses
