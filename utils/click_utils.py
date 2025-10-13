import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt

class Clicker(object):
    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1, click_indx_offset=0):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)
    #生成下一个点击点
    def make_next_click(self, pred_mask, debug=False, iter=0):
        assert self.gt_mask is not None
        click = self._get_next_click(pred_mask, debug=debug, iter=iter)
        self.add_click(click)
        return click
    #获取点击点
    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]
    #计算下一个点击点
    def _get_next_click(self, pred_mask, padding=False, debug=False, iter=0):
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))
    #添加点击点
    def add_click(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False
    #移除最后一个点击点
    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True
    #重置点击状态
    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool_)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []
    #获取当前状态
    def get_state(self):
        return deepcopy(self.clicks_list)
    #设置点击状态
    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)
    #获取点击点数量
    def __len__(self):
        return len(self.clicks_list)

class Click:
    def __init__(self, is_positive, coords, indx=None):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy


def get_click(pred_mask, gt_mask):
    """_summary_

    Args:
        pred_mask (torch.tensor): shape: [n_slice, height, width]
        gt_mask (torch.tensor): shape: [n_slice, height, width]
    """
    def compute_iou_batch(gt_mask, pred_mask):
        if len(gt_mask.shape) == 4:  # [B, D, H, W]
            intersection = torch.logical_and(gt_mask, pred_mask).sum(dim=(2, 3))
            union = torch.logical_or(gt_mask, pred_mask).sum(dim=(2, 3))
        else:  # [D, H, W]
            intersection = torch.logical_and(gt_mask, pred_mask).sum(dim=(1, 2))
            union = torch.logical_or(gt_mask, pred_mask).sum(dim=(1, 2))
        #intersection = torch.logical_and(gt_mask, pred_mask).sum(dim=(1, 2))
        #union = torch.logical_or(gt_mask, pred_mask).sum(dim=(1, 2))
        iou = intersection.float() / union.float()
        return iou
    
        # 处理不同维度输入
    if len(gt_mask.shape) == 4:  # 4D输入 [B, D, H, W]
        # 压缩batch维度（假设batch_size=1）
        gt_mask = gt_mask.squeeze(0)
        pred_mask = pred_mask.squeeze(0)
    else:  # 3D输入 [D, H, W]
        gt_mask = gt_mask
        pred_mask = pred_mask

    # get slices with the lowest IOU, then choose one randomly
    iou_values = compute_iou_batch(gt_mask, pred_mask)
    iou_values[torch.isnan(iou_values)] = 2

    #min_ious = torch.topk(iou_values, 10, largest=False)
    min_ious = torch.topk(iou_values, 40, largest=False)
    # min_ious = torch.topk(iou_values, 1, largest=False)
    random_click_slice_idx = random.sample(min_ious.indices.tolist(), 1)[0]

    # get 2D click position
    clicker = Clicker(gt_mask=gt_mask[random_click_slice_idx].numpy())
    click_pos = clicker.make_next_click(pred_mask[random_click_slice_idx].numpy(), debug=(iter==1), iter=iter)
    
    return [random_click_slice_idx, click_pos.coords[0], click_pos.coords[1]], click_pos.is_positive
    
def get_click_batch(pred_batch, gt_batch):
    ret_pos = []
    ret_click_type = []
    for pred_mask, gt_mask in zip(pred_batch, gt_batch):
        
        if gt_mask.sum() == 0:
            # 没有目标
            ret_pos.append([12345, 12345, 12345])
            ret_click_type.append(False)
            print("no target?")
        else:
            click_pos, is_positive = get_click(pred_mask, gt_mask)
            ret_pos.append(click_pos)
            ret_click_type.append(is_positive)
    return torch.tensor(ret_pos), torch.tensor(ret_click_type)
    

# def debug_vis_pred_gt_click(pred, gt, click, is_positive):
#     # NOTE: input are 3D
#     slice_idx = click[0]
    
#     predict_mask = pred[slice_idx]
#     gt_mask = gt[slice_idx]
#     click_position = click[1:]
    
#     # 创建一个新的图像
#     fig, ax = plt.subplots()

    
    
#     # 可视化真实掩码（使用绿色）
#     ax.imshow(gt_mask, cmap='Blues', alpha=1)
    
#     # 可视化预测掩码（使用红色）
#     ax.imshow(predict_mask, cmap='Greens', alpha=0.5)

    
#     # 可视化点击位置（使用蓝色）
#     # TODO: check this, why 1 then 0?
#     if is_positive:
#         ax.plot(click_position[1], click_position[0], 'ro')
#     else:
#         ax.plot(click_position[1], click_position[0], 'ko')

#     # 设置坐标轴刻度为整数
#     ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
#     ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

#     # 添加图例
#     ax.legend(['Predicted Mask', 'Ground Truth Mask', 'Click Position'])
#     plt.savefig('visualization.png')
#     # 显示图像
#     plt.show()

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



if __name__ == "__main__":
    
    gt_mask = torch.zeros((100, 128, 128)).bool()
    gt_mask[10:70, 10:70, 10:70] = 1
    pred_mask = torch.zeros((100, 128, 128)).bool()
    pred_mask[10:70, 20:80, 20:70] = 1
    
    click_pos, is_positive = get_click_batch([pred_mask], [gt_mask])
    
    print(click_pos, is_positive)
    
    debug_vis_pred_gt_click(pred_mask, gt_mask, click_pos[0], is_positive[0])
    