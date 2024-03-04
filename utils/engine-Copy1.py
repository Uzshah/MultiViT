import torch
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import focal_loss
import utils.lovasz_losses as L
from skimage import color
# from torchmetrics import F1
from .normal_reconstruct import reconstruct
from .losses import proportion_good_pixels, average_angular_error
from torchvision.transforms.functional import rgb_to_grayscale


def ssim(img1, img2, window_size=11, size_average=True):
    """
    Compute Structural Similarity Index (SSI) between two images.

    Args:
    - img1: Tensor, the first image (batch_size x channels x height x width).
    - img2: Tensor, the second image (batch_size x channels x height x width).
    - window_size: int, size of the sliding window for local SSIM computation.
    - size_average: bool, whether to compute the average SSIM across the batch.

    Returns:
    - ssim_index: Tensor, the SSIM index.
    """
    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * 255)**2
    C2 = (K2 * 255)**2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=0)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=0)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1**2, window_size, stride=1, padding=0) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2**2, window_size, stride=1, padding=0) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=0) - mu1_mu2

    SSIM_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    SSIM_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_index = SSIM_n / SSIM_d

    if size_average:
        return 1 - ssim_index.mean()
    else:
        return 1 - ssim_index.mean(1).mean(1).mean(1)


#########standard metrics from FCRN and OmniDepth
def standard_metrics(input_gt_depth_image, pred_depth_image, verbose=False):
    input_gt_depth_image, pred_depth_image = input_gt_depth_image.cpu().detach().numpy(), pred_depth_image.cpu().detach().numpy()
    input_gt_depth = input_gt_depth_image.copy()
    pred_depth = pred_depth_image.copy()

    n = np.sum(input_gt_depth > 1e-3) ####valid samples
                        
    ###invalid samples - no measures
    idxs = ( (input_gt_depth <= 1e-3) )
    pred_depth[idxs] = 1
    input_gt_depth[idxs] = 1

    # print('valid samples:',n,'masked samples:', np.sum(idxs))

    ####STEP 1: compute delta################################################################
    #######prepare mask
    pred_d_gt = pred_depth / input_gt_depth
    pred_d_gt[idxs] = 100
    gt_d_pred = input_gt_depth / pred_depth
    gt_d_pred[idxs] = 100

    Threshold_1_25 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n
    Threshold_1_25_2 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n
    Threshold_1_25_3 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25 * 1.25) / n
    ########################################################################################        

    #####STEP 2: compute mean error##########################################################
    input_gt_depth_norm = input_gt_depth / np.max(input_gt_depth)
    pred_depth_norm = pred_depth / np.max(pred_depth)
        
    log_pred = np.log(pred_depth_norm)
    log_gt = np.log(input_gt_depth_norm)
               
    ###OmniDepth: 
    RMSE_linear = ((pred_depth - input_gt_depth) ** 2).mean()
    RMSE_log = np.sqrt(((log_pred - log_gt) ** 2).mean())
    ARD = (np.abs((pred_depth_norm - input_gt_depth_norm)) / input_gt_depth_norm).mean()
    SRD = (((pred_depth_norm - input_gt_depth_norm)** 2) / input_gt_depth_norm).mean()
   
    if(verbose):
        print('Threshold_1_25: {}'.format(Threshold_1_25))
        print('Threshold_1_25_2: {}'.format(Threshold_1_25_2))
        print('Threshold_1_25_3: {}'.format(Threshold_1_25_3))
        print('RMSE_linear: {}'.format(RMSE_linear))
        print('SRD (MRE): {}'.format(SRD))
        print('ARD (MAE): {}'.format(ARD))
        
    return ARD.item(), SRD.item(), RMSE_linear.item(), RMSE_log.item(), Threshold_1_25.item(), Threshold_1_25_2.item(), Threshold_1_25_3.item() 

# def compute_errors(gt, pred):
#     thresh = torch.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25).float().mean()  # Calculate mean along spatial dimensions
#     a2 = (thresh < 1.25 ** 2).float().mean()
#     a3 = (thresh < 1.25 ** 3).float().mean()

#     rmse = torch.sqrt(torch.mean((gt - pred) ** 2)).mean()
#     rmse_log = torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2)).mean()
#     abs_rel = F.l1_loss(gt, pred)
#     sq_rel = F.mse_loss(gt, pred)
#     return abs_rel.item(), sq_rel.item(), rmse.item(), rmse_log.item(), a1.item(), a2.item(), a3.item()
    
def Dice_Loss(pred, target, smooth=1.0):
    """
    Calculate the multi-class Dice loss.

    Args:
        predictions (torch.Tensor): Predicted class probabilities with shape (batch_size, num_classes, ...)
        targets (torch.Tensor): Ground truth class depth with shape (batch_size, ..., ...)
        num_classes (int): Number of classes
        epsilon (float): Small constant to avoid division by zero

    Returns:
        torch.Tensor: Dice loss
    """
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=151):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = torch.argmax(mask, dim=1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        mask = torch.argmax(mask, dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE).

    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: Mean Absolute Error.
    """
    return F.l1_loss(y_pred, y_true)

def mean_squared_root_error(y_true, y_pred):
    """
    Calculate Mean Squared Root Error (MRSE).

    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: Mean Squared Root Error.
    """
    return torch.sqrt(F.mse_loss(y_pred, y_true))

def C_MSE(y_pred, y_true):
    criterion = nn.MSELoss()

    # Convert RGB to YCbCr
    pred_ycbcr_image = torch.from_numpy(color.rgb2ycbcr(y_pred.squeeze().permute(1, 2, 0).cpu().detach().numpy()))
    true_ycbcr_image = torch.from_numpy(color.rgb2ycbcr(y_true.squeeze().permute(1, 2, 0).cpu().detach().numpy()))

    # Separate Y, Cb, and Cr channels
    G_Y = pred_ycbcr_image[:, :, 0]
    G_Cb = pred_ycbcr_image[:, :, 1]
    G_Cr = pred_ycbcr_image[:, :, 2]

    GT_Y = true_ycbcr_image[:, :, 0]
    GT_Cb = true_ycbcr_image[:, :, 1]
    GT_Cr = true_ycbcr_image[:, :, 2]

    # Calculate MSE for each channel
    MSE_Y = criterion(GT_Y, G_Y)
    MSE_Cb = criterion(GT_Cb, G_Cb)
    MSE_Cr = criterion(GT_Cr, G_Cr)

    # Weighted MSE
    Weighted_MSE = MSE_Y + 0.01*MSE_Cb + 0.01*MSE_Cr

    return Weighted_MSE

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_pred = F.softmax(y_pred, dim=1)
    intersection = torch.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)
    

def train_one_epoch(args, model, criterion1, criterion2, criterion3, optimizer, scheduler, train_loader, device):
    model.train()
    train_loss = 0.0
    depth_loss = 0.
    semantic_loss = 0.
    miou = 0.
    dice_loss= 0.0
    accuracy = 0.
    albedo_loss = 0.
    ssi_alb = 0.
    ssi_shading = 0.
    shading_loss = 0.
    normal_loss = 0.
    normal_loss_new = 0.
    new_loss = 0.
    dloss = 0.
    total_loss = 0.
    # Start the timer for this epoch
    epoch_start_time = time.time()
    total_step = len(train_loader)
    for i, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        depth = batch["depth"].to(device)
        semantic = batch["semantic"].to(device)
        normal = batch["normal"].to(device)
        albedo = batch["albedo"].to(device)
        shading = batch["shading"].to(device)
        # print(albedo.shape)
        # Forward pass
        outputs = model(images)
        n_hat = 2*outputs[4]-1
        estimated_normal = reconstruct(outputs[0]).float()
        n_loss = criterion3(n_hat, estimated_normal.to(device))
        #out = F.softmax(outputs[1], dim=1)
        outputs[0][outputs[0]>17 ] = 17
        loss1 = criterion1(outputs[0], depth)
        loss2 = criterion2(outputs[1], semantic)
        # loss8 = criterion2(outputs[5], semantic)
        # loss3 = criterion2(outputs[-1], semantic)
        loss3 = criterion1(outputs[2], albedo)
        loss4 = criterion1(outputs[3], shading)
        loss7 = criterion1(outputs[4], normal)
        pred_rgb = outputs[3] * outputs[2]
        # rgb = albedo * shading
        rgb_gray, pred_rbg_gray = rgb_to_grayscale(images), rgb_to_grayscale(pred_rgb)
        loss6 = dice_coefficient_loss(semantic, outputs[1])
        loss5 = ssim(rgb_gray, pred_rbg_gray)
        loss = loss1+loss2+loss7+loss5+loss3+loss4+n_loss
        iou = mIoU(outputs[1], semantic, n_classes = args.num_classes)
        acc = pixel_accuracy(outputs[1], semantic)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        depth_loss +=loss1.item()
        
        semantic_loss +=loss2.item()
        albedo_loss +=loss3.item()
        shading_loss +=loss4.item()
        total_loss +=loss5.item()
        dice_loss += loss6.item()
        normal_loss += loss7.item()
        normal_loss_new += n_loss.item()
        # new_loss += loss8.item()
        # dloss += d_loss.item()
        accuracy +=acc
        miou +=iou.item()
        print(f"\r Step [{i+1}/{total_step}], training Loss: {train_loss/len(train_loader):.4f}, new_loss: {new_loss/len(train_loader):.4f}, depth_loss: {dloss/len(train_loader):.4f} Time: {time.time()-epoch_start_time:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}", end='')
    # Calculate training loss and accuracy
    #scheduler.step()
    train_loss /= len(train_loader)
    depth_loss /= len(train_loader)
    semantic_loss /= len(train_loader)
    albedo_loss /= len(train_loader)
    shading_loss /= len(train_loader)
    normal_loss /= len(train_loader)
    total_loss /= len(train_loader)
    new_loss /= len(train_loader)
    accuracy /= len(train_loader)
    miou /= len(train_loader)
    dloss /= len(train_loader)
    dice_loss /= len(train_loader)
    normal_loss_new /= len(train_loader)
    epoch_time = time.time()-epoch_start_time
    training_losses = {
        "total": train_loss,
        "depth": depth_loss,
        "depth_new": dloss,
        "semantic": semantic_loss,
        "dice_loss": dice_loss,
        "albedo": albedo_loss,
        "shading": shading_loss,
        "predrgb" : total_loss,
        "pixel_acc": accuracy,
        "mean_IoU": miou,
        "new_loss": new_loss,
        "normal": normal_loss,
        "normal_new": normal_loss_new,
        "time": epoch_time
    }
    return training_losses

def evalute(args, model, criterion1, criterion2,criterion3, valid_loader, device='cuda'):
    start = time.time()
    model.eval()
    valid_loss = 0.0
    depth_loss = 0.0
    semantic_loss = 0.0
    dice_loss = 0.0
    shading_loss = 0. 
    normal_loss = 0.
    normal_loss_new = 0.
    total_loss = 0.
    new_loss = 0.
    IoU = 0.0
    miou = 0.
    accuracy = 0.
    RMSE_linear, ARD, SRD = 0., 0., 0.
    RMSE_log, A1, A2, A3 = 0., 0., 0., 0.
    albedo_loss = 0.
    total_step= len(valid_loader)
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            images = batch["image"].to(device)
            depth = batch["depth"].to(device)
            semantic = batch["semantic"].to(device)
            normal = batch["normal"].to(device)
            albedo = batch["albedo"].to(device)
            shading = batch["shading"].to(device)
            # print(albedo.shape)
            # Forward pass
            outputs = model(images)
            # out = F.softmax(outputs[1], dim=1)
            # _, preds = out.data.max(1)
            outputs[0][outputs[0]>11 ] = 11
            n_hat = 2*outputs[4]-1
            estimated_normal = reconstruct(depth).float()
            n_loss = criterion3(n_hat, estimated_normal.to(device))
            loss1 = criterion1(outputs[0], depth)
            loss2 = criterion2(outputs[1], semantic)
            loss3 = criterion1(outputs[2], albedo)
            loss4 = criterion1(outputs[3], shading)
            loss7 = criterion1(outputs[4], normal)
            # loss8 = criterion2(outputs[5], semantic)
            dloss = dice_coefficient_loss(semantic, outputs[1])
            pred_rgb = outputs[3] * outputs[2]
            # rgb = albedo * shading
            # rgb_gray, pred_rbg_gray = rgb_to_grayscale(images), rgb_to_grayscale(pred_rgb)
            loss5 = criterion1(images, pred_rgb)
            iou = mIoU(outputs[1], semantic, n_classes = args.num_classes)
            acc = pixel_accuracy(outputs[1], semantic)
            # IoUloss = calculate_iou_multiclass_torch(outputs[1], semantic)
            loss = loss1+loss2+loss7+0.1*loss3+0.1*loss4+n_loss
            valid_loss += loss.item()
            depth_loss +=loss1.item()
            semantic_loss +=loss2.item()
            total_loss +=loss5.item()
            dice_loss +=dloss.item()
            normal_loss +=loss.item()
            normal_loss_new += n_loss.item()
            albedo_loss +=loss3.item()
            shading_loss +=loss4.item()
            # new_loss += loss8.item()
            # IoU +=IoUloss.item()
            accuracy +=acc
            miou +=iou.item()
            AR, SR, RMSE, lRMSE, a1, a2, a3 = standard_metrics(depth, outputs[0])
            RMSE_linear += RMSE; ARD += AR; SRD +=SR; A1 +=a1; A2 +=a2; A3 +=a3; RMSE_log +=lRMSE
            print(f"\r Step [{i+1}/{total_step}], valid Loss: {valid_loss/len(valid_loader):.4f} Time: {time.time()-start:.4f}", end='')
             
    # Calculate validation loss and accuracy
    valid_loss /= len(valid_loader); depth_loss /= len(valid_loader); semantic_loss /= len(valid_loader);
    RMSE_linear /= len(valid_loader); ARD /= len(valid_loader); SRD /=len(valid_loader); A1 /=len(valid_loader); A2 /=len(valid_loader); 
    A3 /=len(valid_loader); RMSE_log/=len(valid_loader); 
    dice_loss /=len(valid_loader); normal_loss /= len(valid_loader);
    albedo_loss /= len(valid_loader); shading_loss /= len(valid_loader);
    total_loss /= len(valid_loader);
    accuracy /= len(valid_loader)
    new_loss /= len(valid_loader)
    normal_loss /= len(valid_loader)
    normal_loss_new /= len(valid_loader)
    miou /= len(valid_loader)
    valid_time = time.time()-start
    
    validation_losses = {
        "total": valid_loss,
        "depth": depth_loss,
        "semantic": semantic_loss,
        "new_loss": new_loss,
        "rmse": RMSE_linear,
        "log_rmse": RMSE_log,
        "normal": normal_loss,
        "normal_new": normal_loss_new,
        "mae": ARD,
        "mse": SRD,
        "a1": A1,
        "a2": A2,
        "a3": A3,
        "dice_loss": dice_loss,
        "iou": IoU,
        "albedo": albedo_loss,
        "shading": shading_loss,
        "predrgb" : total_loss,
        "pixel_acc": accuracy,
        "mean_IoU": miou,
        "time": valid_time
    }
    return validation_losses
