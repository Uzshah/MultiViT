import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
from PIL import Image


warnings.filterwarnings("ignore")


def vis(model, batch, dir, bid):
    images = batch['image'].cuda()
    labels = batch['depth'].cuda()
    semantic = batch['semantic'].cuda()
    albedo = batch['albedo'].cuda()
    shading = batch['shading'].cuda()
    normal = batch['normal'].cuda()
    mask = batch["val_mask"].cuda()
    
    
    pred_depth, pred_semantic, pred_albedo, pred_shading, pred_normal, _ = model(images)
    for idx, img in enumerate(images*mask):
        img = img.permute(1,2,0).cpu().detach().numpy()
        img = (img * 255.0).astype(np.uint8)
        plt.imsave(f"{dir}/RGB_{idx}_{bid}.png", img)
        
        
    for idx, (lbl, pred) in enumerate(zip(labels*mask, pred_depth*mask)):
        lbl = lbl.permute(1,2,0).squeeze().cpu().detach().numpy()
        pred = pred.permute(1,2,0).squeeze().cpu().detach().numpy()
        min, max = lbl.min(), lbl.max()
        # pred =np.clip(pred, min, max)
        # lbl =np.clip(lbl, 0.0,1.0)
        pred[pred>11] = 11
        plt.imsave(f"{dir}/pred_depth_{idx}_{bid}.png", pred, cmap = "plasma")
        plt.imsave(f"{dir}/gt_depth_{idx}_{bid}.png", lbl, cmap = "plasma")
        
    
    for idx, (lbl, pred) in enumerate(zip(shading*mask, pred_shading*mask)):
        lbl = lbl.permute(1,2,0).squeeze().cpu().detach().numpy()
        pred = pred.permute(1,2,0).squeeze().cpu().detach().numpy()
        pred = np.clip(pred, 0.0,1.0)
        plt.imsave(f"{dir}/gt_shading_{idx}_{bid}.png", lbl, cmap = "gray")
        plt.imsave(f"{dir}/pred_shading_{idx}_{bid}.png", pred, cmap = "gray")
        
    for idx, (lbl, pred) in enumerate(zip(albedo*mask, pred_albedo*mask)):
        lbl = lbl.permute(1,2,0).cpu().detach().numpy()
        pred = pred.permute(1,2,0).cpu().detach().numpy()
        pred =np.clip(pred, 0.0,1.0)
        lbl = (lbl * 255.0).astype(np.uint8)
        pred = (pred * 255.0).astype(np.uint8)
        plt.imsave(f"{dir}/gt_albedo_{idx}_{bid}.png", lbl)
        plt.imsave(f"{dir}/pred_albedo_{idx}_{bid}.png", pred)
        
    
    for idx, (lbl, pred) in enumerate(zip(normal*mask, pred_normal*mask)):
        lbl = lbl.permute(1,2,0).cpu().detach().numpy()
        pred = pred.permute(1,2,0).cpu().detach().numpy()
        lbl = (lbl * 255.0).astype(np.uint8)
        pred = (pred * 255.0).astype(np.uint8)
        plt.imsave(f"{dir}/gt_normal_{idx}_{bid}.png", lbl)
        plt.imsave(f"{dir}/pred_normal_{idx}_{bid}.png", pred)
        
    
    for idx, (se, se_pred) in enumerate(zip(semantic*mask, pred_semantic*mask)):
        se = torch.argmax(se, dim = 0).cpu().detach().numpy()
        se_pred = torch.argmax(se_pred , dim = 0).cpu().detach().numpy()
        
        plt.imsave(f"{dir}/gt_semantic_{idx}_{bid}.png", se)
        plt.imsave(f"{dir}/pred_semantic_{idx}_{bid}.png", se_pred)
        
