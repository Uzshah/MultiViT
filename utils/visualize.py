import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings


warnings.filterwarnings("ignore")


def batch_vis(args, model, batch, dir, bid, epoch = ''):
    if args.dataset =='s3d':
        images = batch['image'].cuda()
        labels = batch['depth'].cuda()
        semantic = batch['semantic'].cuda()
        albedo = batch['albedo'].cuda()
        shading = batch['shading'].cuda()
        normal = batch['normal'].cuda()
        _, _, H, _ = images.shape
        
        pred_depth, pred_semantic, pred_albedo, pred_shading, pred_normal, _ = model(images)
        for idx, (img) in enumerate(images):
            img = img.permute(1,2,0).cpu().detach().numpy()
            img = (img * 255.0).astype(np.uint8)
            plt.imshow(img)
            plt.imsave(f"{dir}/epoch_{epoch}_{bid}_orignal_{idx}.png", img)
            
            
        for idx, (lbl, pred) in enumerate(zip(labels, pred_depth)):
            lbl = lbl.permute(1,2,0).squeeze().cpu().detach().numpy()
            pred = pred.permute(1,2,0).squeeze().cpu().detach().numpy()
            pred[pred>11] = 0
            lbl[lbl>11] = 0
            border = np.zeros((H, 15))
            
            img = np.hstack((lbl, border, pred))
            # plt.imshow(img, cmap = "viridis")
            plt.imsave(f"{dir}/epoch_{epoch}_{bid}_depth_{idx}.png", img, cmap = "plasma")
            
    
        for idx, (lbl, pred) in enumerate(zip(shading, pred_shading)):
            lbl = lbl.permute(1,2,0).squeeze().cpu().detach().numpy()
            pred = pred.permute(1,2,0).squeeze().cpu().detach().numpy()
            pred = np.clip(pred, 0.0,1.0)
            border = np.zeros((H, 15))
            img = np.hstack((lbl, border, pred))
            # plt.imshow(img, cmap = "gray")
            plt.imsave(f"{dir}/epoch_{epoch}_{bid}_shading_{idx}.png", img, cmap = "gray")
            
        for idx, (lbl, pred) in enumerate(zip(albedo, pred_albedo)):
            lbl = lbl.permute(1,2,0).cpu().detach().numpy()
            pred = pred.permute(1,2,0).cpu().detach().numpy()
            # lbl = (lbl * 255.0).astype(np.uint8)
            # pred = (pred * 255.0).astype(np.uint8)
            
            border = np.zeros((H, 15, 3))
            
            img = np.hstack((lbl, border, pred))
            img = np.clip(img, 0.0,1.0)
            img = (img * 255.0).astype(np.uint8)
            plt.imshow(img)
            plt.imsave(f"{dir}/epoch_{epoch}_{bid}_albedo_{idx}.png", img)
    
        for idx, (lbl, pred) in enumerate(zip(normal, pred_normal)):
            lbl = lbl.permute(1,2,0).cpu().detach().numpy()
            pred = pred.permute(1,2,0).cpu().detach().numpy()
            # lbl = (lbl * 255.0).astype(np.uint8)
            # pred = (pred * 255.0).astype(np.uint8)
            
            border = np.zeros((H, 15, 3))
            
            img = np.hstack((lbl, border, pred))
            img = np.clip(img, 0.0,1.0)
            img = (img * 255.0).astype(np.uint8)
            plt.imshow(img)
            plt.imsave(f"{dir}/epoch_{epoch}_{bid}_normal_{idx}.png", img)
        
        for idx, (se, se_pred) in enumerate(zip(semantic, pred_semantic)):
            se = torch.argmax(se, dim = 0).cpu().detach().numpy()
            se_pred = torch.argmax(se_pred , dim = 0).cpu().detach().numpy()
            # print(se_pred.shape)
            border = np.zeros((H, 15))
            
            img1 = np.hstack((se, border, se_pred))
            plt.imshow(img1)
            plt.imsave(f"{dir}/epoch_{epoch}_{bid}_semantic_{idx}.png", img1)
    else:
        images = batch['image'].cuda()
        labels = batch['depth'].cuda()
        semantic = batch['semantic'].cuda()
        normal = batch['normal'].cuda()
        _, _, H, _ = images.shape
        pred_depth, pred_semantic, pred_albedo, pred_shading, pred_normal, _ = model(images)
        for idx, (img) in enumerate(images):
            img = img.permute(1,2,0).cpu().detach().numpy()
            img = (img * 255.0).astype(np.uint8)
            # plt.imshow(img)
            plt.imsave(f"{dir}/epoch_{epoch}_{bid}_orignal_{idx}.png", img)
            
            
        for idx, (lbl, pred) in enumerate(zip(labels, pred_depth)):
            lbl = lbl.permute(1,2,0).squeeze().cpu().detach().numpy()
            pred = pred.permute(1,2,0).squeeze().cpu().detach().numpy()
            pred[pred>11] = 0
            border = np.zeros((H, 15))
            
            img = np.hstack((lbl, border, pred))
            # plt.imshow(img, cmap = "viridis")
            plt.imsave(f"{dir}/epoch_{epoch}_{bid}_depth_{idx}.png", img, cmap = "plasma")
            
    
        for idx, lbl in enumerate(pred_shading):
            lbl = lbl.permute(1,2,0).squeeze().cpu().detach().numpy()
            lbl = np.clip(lbl, 0.0, 1.0)
            plt.imsave(f"{dir}/epoch_{epoch}_{bid}_shading_{idx}.png", lbl, cmap = "gray")
            
        for idx, lbl in enumerate(pred_albedo):
            lbl = lbl.permute(1,2,0).cpu().detach().numpy()
            lbl = np.clip(lbl, 0.0, 1.0)
            plt.imsave(f"{dir}/epoch_{epoch}_{bid}_albedo_{idx}.png", lbl)
    
        for idx, (lbl, pred) in enumerate(zip(normal, pred_normal)):
            lbl = lbl.permute(1,2,0).cpu().detach().numpy()
            pred = pred.permute(1,2,0).cpu().detach().numpy()
            
            border = np.zeros((H, 15, 3))
            
            img = np.hstack((lbl, border, pred))
            img = np.clip(img, 0.0,1.0)
            img = (img * 255.0).astype(np.uint8)
            plt.imshow(img)
            plt.imsave(f"{dir}/epoch_{epoch}_{bid}_normal_{idx}.png", img)
        
        for idx, (se, se_pred) in enumerate(zip(semantic, pred_semantic)):
            se = torch.argmax(se, dim = 0).cpu().detach().numpy()
            se_pred = torch.argmax(se_pred , dim = 0).cpu().detach().numpy()
            # print(se_pred.shape)
            border = np.zeros((H, 15))
            
            img1 = np.hstack((se, border, se_pred))
            plt.imshow(img1)
            plt.imsave(f"{dir}/epoch_{epoch}_{bid}_semantic_{idx}.png", img1)
        
