import torch.nn as nn
import torch
import matplotlib.pyplot as plt

class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, use_disks=True):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.use_disks = use_disks

    def get_coord_features(self, points, batchsize, depth, rows, cols):
        """
        points: bs x num_point, 2
        """
        #创建网格坐标，深度，行，列
        depth_array = torch.arange(start=0, end=depth, step=1, dtype=torch.float32, device=points.device)
        row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
        col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

        coord_depths, coord_rows, coord_cols = torch.meshgrid(depth_array, row_array, col_array)
        #print(coord_rows.shape)
        coords = torch.stack((coord_depths, coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1, 1)
        
        #print(coords)
        #添加输入的点坐标偏移
        add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1, 1)
        coords.add_(-add_xy)
        
        #print(coords)
        
        if not self.use_disks:
            coords.div_(self.norm_radius * self.spatial_scale)
        coords.mul_(coords)

        coords = coords.sum(1)

        coords = coords.view(batchsize, -1, depth, rows, cols)

        coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
        coords = coords.view(batchsize, -1, depth, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()

        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3], x.shape[4])

if __name__ == "__main__":
    dis_map = DistMaps(3, use_disks=True)
    points = torch.tensor(
        [
            [10, 20, 30],
            [10, 35, 45],
            [50, 55, 60]
        ]
    )
    batchsize = 1
    depth = 128
    rows = 128
    cols = 128
    point_feat = dis_map.get_coord_features(points, batchsize, depth, rows, cols)
    print(point_feat.shape)

    plt.imshow(point_feat[0][0][8])
    plt.savefig("images/click_encoding_debug/click.png")
    plt.show()