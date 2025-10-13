import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def debug_vis_pred_gt_click(A, B, C, C_title, D, D_title, click_position, is_positive, path):
    # A: image
    # B: gt
    # C: 1 pred
    # D: 2 pred
    print(A.shape, B.shape, C.shape, D.shape)
    
    # 创建一个包含4个子图的画布
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # 子图1：画A
    axs[0, 0].imshow(A, cmap='gray')
    axs[0, 0].set_title('Image')
    
    # 子图2：画A和B，B透明度0.5
    axs[0, 1].imshow(A, cmap='gray')
    axs[0, 1].imshow(B, alpha=0.5)
    axs[0, 1].set_title('GT')
    
    # 子图3：画A和C，C透明度0.5，并在给定位置绘制一个点
    axs[1, 0].imshow(A, cmap='gray')
    axs[1, 0].imshow(C, alpha=0.5)
    if is_positive:
        axs[1, 0].plot(click_position[1], click_position[0], 'ro')
    else:
        axs[1, 0].plot(click_position[1], click_position[0], 'ko')
    axs[1, 0].set_title(f'pred w/o click :{round(C_title, 4)}')

    # 子图4：画A和D，D透明度0.5
    axs[1, 1].imshow(A, cmap='gray')
    axs[1, 1].imshow(D, alpha=0.5)
    axs[1, 1].set_title(f'pred with click :{round(D_title, 4)}')
    
    # 调整子图之间的间距
    plt.tight_layout()
    plt.savefig(path)
    # 显示图形
    plt.show()


def visualize_images(tensors, folder_name, name):

    # 创建一个包含两个子图的画布
    fig, axes = plt.subplots(1, len(tensors))

    for ax_id, ax in enumerate(axes):
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        # 在左边的子图上绘制第一个张量
        ax.imshow(tensors[ax_id], cmap='gray')
        ax.set_title(f'Tensor {ax_id}')

    # 显示图形
    plt.axis('off')
    os.makedirs(f"debug_imgs/{folder_name}", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{folder_name}/{name}.png", dpi=120)
    plt.cla()


def click_contrastive_loss(positive_clicks, negative_clicks, feature):
    """
    input clicks:
        形状：(bs, num_click, 3)
        3指三维坐标，顺序: (H, W, D)，范围是0~128
        NOTE： click could be Empty
    feature:
        形状：(bs, dim, size, size, size)
        顺序是：H, W, D
    """

    # Torch: grid specifies the sampling pixel locations normalized by the input spatial dimensions. Therefore, it should have most values in the range of [-1, 1]. For example, values x = -1, y = -1 is the left-top pixel of input, and values x = 1, y = 1 is the right-bottom pixel of input.
    # In the case of 5D inputs, grid[n, d, h, w] specifies the x, y, z pixel locations for interpolating output[n, :, d, h, w].
    
    positive_clicks = positive_clicks / 128 * 2 - 1
    negative_clicks = negative_clicks / 128 * 2 - 1
    
    # input feature should be (bs, dim, H, W, D) and be changed to (bs, dim, D, H, W)
    # NOTE: check this
    feature = feature.permute(0, 1, 4, 2, 3)
    
    # for idx, f in enumerate(feature.squeeze()):
    #     print(idx, f)
    # print("input positive_clicks", positive_clicks)
    
    positive_feat = F.grid_sample(feature, positive_clicks.unsqueeze(1).unsqueeze(1), padding_mode='zeros', align_corners=True).squeeze(2).squeeze(2)  # bs, dim, num_click
    negative_feat = F.grid_sample(feature, negative_clicks.unsqueeze(1).unsqueeze(1), padding_mode='zeros', align_corners=True).squeeze(2).squeeze(2)  # bs, dim, num_click
    
    # print(positive_feat.shape, negative_feat.shape)  # torch.Size([2, 256, 10]) torch.Size([2, 256, 8])
    # print("positive_feat", positive_feat)
    # print("negative_feat", negative_feat)
    
    # global
    global_pos = positive_feat.mean(-1)
    global_neg = negative_feat.mean(-1)
    loss_global = -1 * F.mse_loss(global_pos, global_neg)
    
    return loss_global



def get_click_embedding(positive_clicks, negative_clicks, feature):
    """
    input clicks:
        形状：(bs, num_click, 3)
        3指三维坐标，顺序: (H, W, D)，范围是0~128
        NOTE： click could be Empty
    feature:
        形状：(bs, dim, size, size, size)
        顺序是：H, W, D
    """

    # Torch: grid specifies the sampling pixel locations normalized by the input spatial dimensions. Therefore, it should have most values in the range of [-1, 1]. For example, values x = -1, y = -1 is the left-top pixel of input, and values x = 1, y = 1 is the right-bottom pixel of input.
    # In the case of 5D inputs, grid[n, d, h, w] specifies the x, y, z pixel locations for interpolating output[n, :, d, h, w].
    
    positive_clicks = positive_clicks / 127 * 2 - 1
    negative_clicks = negative_clicks / 127 * 2 - 1
    
    # input feature should be (bs, dim, H, W, D) and be changed to (bs, dim, D, H, W)
    # NOTE: check this
    # feature = feature.permute(0, 1, 4, 2, 3)
    
    # for idx, f in enumerate(feature.squeeze()):
    #     print(idx, f)
    # print("input positive_clicks", positive_clicks)
    
    positive_feat = F.grid_sample(feature, positive_clicks.unsqueeze(1).unsqueeze(1), mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(2).squeeze(2)  # bs, dim, num_click
    negative_feat = F.grid_sample(feature, negative_clicks.unsqueeze(1).unsqueeze(1), mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(2).squeeze(2)  # bs, dim, num_click
    
    # print(positive_feat.shape, negative_feat.shape)  # torch.Size([2, 256, 10]) torch.Size([2, 256, 8])
    # print("positive_feat", positive_feat)
    # print("negative_feat", negative_feat)
    
    return positive_feat, negative_feat


def get_random_click(seg, click_num, dis_map, device):
    """_summary_

    Args:
        seg (tensor): GT mask, shape: (bs, D, H, W)
        click_num (int): click number
        device (device): gpu

    Returns:
        ret_pos_clicks: clicks, could be []
        ret_neg_clicks: clicks, shape: (bs, click_num, 3)
    """
    ret_pos_clicks = []
    ret_neg_clicks = []
    ret_pos_disks = []
    ret_neg_disks = []
    for _seg in seg:
        _seg = _seg.unsqueeze(0)
        assert _seg.shape[-1] == 128
        
        l = len(torch.where(_seg == 1)[0])
        if l > 0:
            np.random.seed(2024)
            sample = np.random.choice(np.arange(l), click_num, replace=True)
            x = torch.where(_seg == 1)[1][sample].unsqueeze(1)  # D
            y = torch.where(_seg == 1)[2][sample].unsqueeze(1)  # H
            z = torch.where(_seg == 1)[3][sample].unsqueeze(1)  # W
            points_pos_for_disk = torch.cat([x, y, z], dim=1).float().to(device)
            #points_pos = torch.cat([y, z, x], dim=1).float().to(device)  # DHW
            disk_pos = dis_map.get_coord_features(points_pos_for_disk, 1, 128, 128, 128)
            points_pos = torch.cat([z, y, x], dim=1).float().to(device)
            points_pos = points_pos.unsqueeze(0)
        else:
            print("no target")
            points_pos = []
            disk_pos = torch.zeros(1, 1, 128, 128, 128).to(device)
        ret_pos_clicks.append(points_pos)
        ret_pos_disks.append(disk_pos)
        l = len(torch.where(_seg == 0)[0])
        np.random.seed(2024)
        sample = np.random.choice(np.arange(l), click_num, replace=True)
        x = torch.where(_seg == 0)[1][sample].unsqueeze(1)
        y = torch.where(_seg == 0)[2][sample].unsqueeze(1)
        z = torch.where(_seg == 0)[3][sample].unsqueeze(1)
        points_neg_for_disk = torch.cat([x, y, z], dim=1).float().to(device)
        #points_neg = torch.cat([y, z, x], dim=1).float().to(device)  # DHW
        disk_neg = dis_map.get_coord_features(points_neg_for_disk, 1, 128, 128, 128)
        points_neg = torch.cat([z, y, x], dim=1).float().to(device)
        points_neg = points_neg.unsqueeze(0)
        
        ret_neg_clicks.append(points_neg)
        ret_neg_disks.append(disk_neg)
    return ret_pos_clicks, ret_neg_clicks, ret_pos_disks, ret_neg_disks


def get_random_click_(seg, click_num, dis_map, device):
    """_summary_

    Args:
        seg (tensor): GT mask, shape: (bs, D, H, W)
        click_num (int): click number
        device (device): gpu

    Returns:
        ret_pos_clicks: clicks, could be []
        ret_neg_clicks: clicks, shape: (bs, click_num, 3)
    """
    ret_pos_clicks = []
    ret_neg_clicks = []
    ret_pos_disks = []
    ret_neg_disks = []
    for _seg in seg:
        _seg = _seg.unsqueeze(0)
        assert _seg.shape[-1] == 128
        
        l = len(torch.where(_seg == 1)[0])
        if l > 0:
            sample = np.random.choice(np.arange(l), click_num, replace=True)
            x = torch.where(_seg == 1)[1][sample].unsqueeze(1)  # D
            y = torch.where(_seg == 1)[2][sample].unsqueeze(1)  # H
            z = torch.where(_seg == 1)[3][sample].unsqueeze(1)  # W
            #points_pos = torch.cat([x, y, z], dim=1).unsqueeze(0).float().to(device)
            points_pos = torch.cat([y, z, x], dim=1).float().to(device)  # DHW
            disk_pos = dis_map.get_coord_features(points_pos, 1, 128, 128, 128)
            points_pos = points_pos.unsqueeze(0)
        else:
            print("no target")
            points_pos = []
            disk_pos = torch.zeros(1, 1, 128, 128, 128).to(device)
        ret_pos_clicks.append(points_pos)
        ret_pos_disks.append(disk_pos.permute(0, 1, 3, 4, 2))
        l = len(torch.where(_seg == 0)[0])
        sample = np.random.choice(np.arange(l), click_num, replace=True)
        x = torch.where(_seg == 0)[1][sample].unsqueeze(1)
        y = torch.where(_seg == 0)[2][sample].unsqueeze(1)
        z = torch.where(_seg == 0)[3][sample].unsqueeze(1)
        #points_neg = torch.cat([x, y, z], dim=1).unsqueeze(0).float().to(device)
        points_neg = torch.cat([y, z, x], dim=1).float().to(device)  # DHW
        disk_neg = dis_map.get_coord_features(points_neg, 1, 128, 128, 128)
        points_neg = points_neg.unsqueeze(0)
        
        ret_neg_clicks.append(points_neg)
        ret_neg_disks.append(disk_neg.permute(0, 1, 3, 4, 2))
    return ret_pos_clicks, ret_neg_clicks, ret_pos_disks, ret_neg_disks


def loss_pCE(outputs, targets, eps=1e-12):
    src_masks = F.softmax(outputs, dim=1)
    y_labeled = targets
    print("src_masks", src_masks.shape, "y_labeled", y_labeled.shape)
    cross_entropy = - torch.sum(y_labeled * torch.log(src_masks + eps), dim = 1)
    return - cross_entropy.mean()