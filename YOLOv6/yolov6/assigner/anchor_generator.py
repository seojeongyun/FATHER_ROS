import torch
from yolov6.utils.general import check_version

torch_1_10_plus = check_version(torch.__version__, minimum='1.10.0')


def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5,
                     device='cpu', is_eval=False, mode='af'):
    ''' Generate anchors from features '''
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None

    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij') if torch_1_10_plus else torch.meshgrid(
                shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(torch.float)
            if mode == 'af':
                # anchor-free
                # anchor_point.reshape([-1, 2]) --> shape: (w * h, 2)
                anchor_points.append(anchor_point.reshape([-1, 2]))
                # stride_tensor - shape: (h * w, 1)
                stride_tensor.append(
                    torch.full(
                        size=(h * w, 1), fill_value=stride, dtype=torch.float, device=device)
                )
            elif mode == 'ab':
                # anchor-based
                # Example of .repeat
                # >>> x = torch.tensor([1, 2, 3])
                # >>> x.repeat(4, 2)
                # tensor([[ 1,  2,  3,  1,  2,  3],
                #         [ 1,  2,  3,  1,  2,  3],
                #         [ 1,  2,  3,  1,  2,  3],
                #         [ 1,  2,  3,  1,  2,  3]])
                # The meaning of repeat(3, 1) is that three anchors are used for each grid.
                #
                # anchor_point.reshape([-1, 2]).repeat(3, 1) --> shape: (h * w * 3, 2)
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
                # stride_tensor - shape: (h * w * 3, 1)
                stride_tensor.append(
                    torch.full(
                        size=(h * w, 1), fill_value=stride, dtype=torch.float, device=device).repeat(3, 1)
                )
                # --> result of torch.full() is
                #          tensor([[stride],
                #                  [stride],
                #                     ...
                #                  [stride],
                #                  [stride]])
                # --> Shape of torch.full() is (h * w, 1), and the result of repeat(3, 1) is (h * w * 3, 1).

        # An example is that fpn_strides = [8, 16, 32] and mode == 'af'.
        # anchor_points - shape is as (8 * 8 + 16 * 16 + 32 * 32, 2).
        # stride_tensor - shape is as (8 * 8 + 16 * 16 + 32 * 32, 1).
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            # Example: w=2, h=2, stride=32, grid_cell_offset=0.5, grid_cell_size = 5.0
            #          Total number of grids = 32 * 32
            # cell_half_size = 5.0 * 32 * 0.5 = 5 * 16 = 80
            cell_half_size = grid_cell_size * stride * 0.5
            # Example: w=2, h=2, stride=32, grid_cell_offset=0.5
            # shift_x = ([0, 1] + grid_cell_offset) * stride = [0, 32] + 16 = [0 + 16, 32 + 16]
            # shift_y = ([0, 1] + grid_cell_offset) * stride = [0, 32] + 16 = [0 + 16, 32 + 16]
            # ==> These values are the center points of anchors for the size of a input image
            # That is, the center point of anchors consists of (shift_x, shift_y).
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij') \
                if torch_1_10_plus else torch.meshgrid(shift_y, shift_x)
            #   |-> shift_y.shape = shift_x.shape = [2, 2]
            #
            # For w=2, h=2
            # anchor - shape: (2, 2, 4)
            #          Shape of (shift_x - cell_half_size): (2, 2)
            #          Shape of (shift_y - cell_half_size): (2, 2)
            #          Shape of (shift_x + cell_half_size): (2, 2)
            #          Shape of (shift_y + cell_half_size): (2, 2)
            #           |-> torch.stack(, dim=-1) -> shape: (2, 2, 4)
            #
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                dim=-1).clone().to(feats[0].dtype)
            # anchor_point - shape; (2, 2, 2)
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).clone().to(feats[0].dtype)

            if mode == 'af':  # anchor-free
                # anchor.reshape([-1, 4]) - shape: (4, 4)
                anchors.append(anchor.reshape([-1, 4]))
                # anchor_point.reshape([-1, 2]) - shape: (4, 2) == (w * h, 2)
                # --> center of anchors
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == 'ab':  # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3, 1))
                # anchor_point.reshape([-1, 2]).repeat(3, 1) - shape: (12, 2) == (w * h * 3, 2)
                # --> center of anchors
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor


if __name__ == '__main__':
    # def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5,
    #                      device='cpu', is_eval=False, mode='af'):
    feats = [torch.randn(1, 100, 2, 2)]
    fpn_strides = [16]
    grid_cell_size = 5.0
    grid_cell_offset = 0.5
    anchors, anchor_points, num_anchors_list, stride_tensor = \
        generate_anchors(feats, fpn_strides, grid_cell_size, grid_cell_offset)
