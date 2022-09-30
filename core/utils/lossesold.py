# -*- coding: utf-8 -*-
# @Time    : 20-3-8 下午5:53
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class loss_functions():

    @classmethod
    def torch_warp(cls, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """

        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # print(grid.shape,flo.shape,'...')
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        # tools.check_tensor(x, 'x')
        # tools.check_tensor(vgrid, 'vgrid')
        output = nn.functional.grid_sample(x, vgrid, padding_mode='zeros')
        # mask = torch.autograd.Variable(torch.ones(x.size()))
        # if x.is_cuda:
        #     mask = mask.cuda()
        # mask = nn.functional.grid_sample(mask, vgrid, padding_mode='zeros')
        #
        # mask[mask < 0.9999] = 0
        # mask[mask > 0] = 1
        # output = output * mask
        # # nchw->>>nhwc
        # if x.is_cuda:
        #     output = output.cpu()
        # output_im = output.numpy()
        # output_im = np.transpose(output_im, (0, 2, 3, 1))
        # output_im = np.squeeze(output_im)
        return output

    @classmethod
    def photo_loss_function(cls, diff, mask, q, charbonnier_or_abs_robust, if_use_occ, averge=True):
        if charbonnier_or_abs_robust:
            if if_use_occ:
                p = ((diff) ** 2 + 1e-6).pow(q)
                p = p * mask
                if averge:
                    p = p.mean()
                    ap = mask.mean()
                else:
                    p = p.sum()
                    ap = mask.sum()
                loss_mean = p / (ap * 2 + 1e-6)
            else:
                p = ((diff) ** 2 + 1e-8).pow(q)
                if averge:
                    p = p.mean()
                else:
                    p = p.sum()
                return p
        else:
            if if_use_occ:
                diff = (torch.abs(diff) + 0.01).pow(q)
                diff = diff * mask
                diff_sum = torch.sum(diff)
                loss_mean = diff_sum / (torch.sum(mask) * 2 + 1e-6)
            else:
                diff = (torch.abs(diff) + 0.01).pow(q)
                if averge:
                    loss_mean = diff.mean()
                else:
                    loss_mean = diff.sum()
        return loss_mean

    @classmethod
    def census_loss_torch(cls, img1, img1_warp, mask, q, charbonnier_or_abs_robust, if_use_occ, averge=True, max_distance=3):
        patch_size = 2 * max_distance + 1

        def _ternary_transform_torch(image):
            R, G, B = torch.split(image, 1, 1)
            intensities_torch = (0.2989 * R + 0.5870 * G + 0.1140 * B)  # * 255  # convert to gray
            # intensities = tf.image.rgb_to_grayscale(image) * 255
            out_channels = patch_size * patch_size
            w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))  # h,w,1,out_c
            w_ = np.transpose(w, (3, 2, 0, 1))  # 1,out_c,h,w
            weight = torch.from_numpy(w_).float()
            if image.is_cuda:
                weight = weight.cuda()
            patches_torch = torch.conv2d(input=intensities_torch, weight=weight, bias=None, stride=[1, 1], padding=[max_distance, max_distance])
            transf_torch = patches_torch - intensities_torch
            transf_norm_torch = transf_torch / torch.sqrt(0.81 + transf_torch ** 2)
            return transf_norm_torch

        def _hamming_distance_torch(t1, t2):
            dist = (t1 - t2) ** 2
            dist = torch.sum(dist / (0.1 + dist), 1, keepdim=True)
            return dist

        def create_mask_torch(tensor, paddings):
            shape = tensor.shape  # N,c, H,W
            inner_width = shape[2] - (paddings[0][0] + paddings[0][1])
            inner_height = shape[3] - (paddings[1][0] + paddings[1][1])
            inner_torch = torch.ones([shape[0], shape[1], inner_width, inner_height]).float()
            if tensor.is_cuda:
                inner_torch = inner_torch.cuda()
            mask2d = F.pad(inner_torch, [paddings[0][0], paddings[0][1], paddings[1][0], paddings[1][1]])
            return mask2d

        img1 = _ternary_transform_torch(img1)
        img1_warp = _ternary_transform_torch(img1_warp)
        dist = _hamming_distance_torch(img1, img1_warp)
        transform_mask = create_mask_torch(mask, [[max_distance, max_distance],
                                                  [max_distance, max_distance]])
        census_loss = cls.photo_loss_function(diff=dist, mask=mask * transform_mask, q=q,
                                              charbonnier_or_abs_robust=charbonnier_or_abs_robust, if_use_occ=if_use_occ, averge=averge)
        return census_loss

    @classmethod
    def flow_smooth_delta(cls, flow, if_second_order=False):
        def gradient(x):
            D_dy = x[:, :, 1:] - x[:, :, :-1]
            D_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            return D_dx, D_dy

        dx, dy = gradient(flow)
        # dx2, dxdy = gradient(dx)
        # dydx, dy2 = gradient(dy)
        if if_second_order:
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            smooth_loss = dx.abs().mean() + dy.abs().mean() + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
        else:
            smooth_loss = dx.abs().mean() + dy.abs().mean()
        # smooth_loss = dx.abs().mean() + dy.abs().mean()  # + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()
        # 暂时不上二阶的平滑损失，似乎加上以后就太猛了，无法降低photo loss TODO
        return smooth_loss

    @classmethod
    def edge_aware_smoothness_per_pixel(cls, img, pred):
        def gradient_x(img):
            gx = img[:, :, :-1, :] - img[:, :, 1:, :]
            return gx

        def gradient_y(img):
            gy = img[:, :, :, :-1] - img[:, :, :, 1:]
            return gy

        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)

        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)


    # boundary dilated warping
    class boundary_dilated_warp():

        @classmethod
        def get_grid(cls, batch_size, H, W, start):
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
            ones = torch.ones_like(xx)
            grid = torch.cat((xx, yy, ones), 1).float()
            if torch.cuda.is_available():
                grid = grid.cuda()
            # print("grid",grid.shape)
            # print("start", start)
            grid[:, :2, :, :] = grid[:, :2, :, :] + start  # 加上patch在原图内的偏移量

            return grid

        @classmethod
        def transformer(cls, I, vgrid, train=True):
            # I: Img, shape: batch_size, 1, full_h, full_w
            # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
            # outsize: (patch_h, patch_w)

            def _repeat(x, n_repeats):

                rep = torch.ones([n_repeats, ]).unsqueeze(0)
                rep = rep.int()
                x = x.int()

                x = torch.matmul(x.reshape([-1, 1]), rep)
                return x.reshape([-1])

            def _interpolate(im, x, y, out_size, scale_h):
                # x: x_grid_flat
                # y: y_grid_flat
                # out_size: same as im.size
                # scale_h: True if normalized
                # constants
                
                num_batch, num_channels, height, width = im.size()

                out_height, out_width = out_size[0], out_size[1]
                # zero = torch.zeros_like([],dtype='int32')
                zero = 0
                max_y = height - 1
                max_x = width - 1
                if scale_h:
                    # scale indices from [-1, 1] to [0, width or height]
                    # print('--Inter- scale_h:', scale_h)
                    x = (x + 1.0) * (height) / 2.0
                    y = (y + 1.0) * (width) / 2.0

                # do sampling
                x0 = torch.floor(x).int()
                x1 = x0 + 1
                y0 = torch.floor(y).int()
                y1 = y0 + 1

                x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
                x1 = torch.clamp(x1, zero, max_x)
                y0 = torch.clamp(y0, zero, max_y)
                y1 = torch.clamp(y1, zero, max_y)

                dim1 = torch.from_numpy(np.array(width * height))
                dim2 = torch.from_numpy(np.array(width))
                
                base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)  # In fact, it is simply to mark the subscript position of each graph in the batch
                # base = torch.arange(0,num_batch) * dim1
                # base = base.reshape(-1, 1).repeat(1, out_height * out_width).reshape(-1).int()
                # the difference? expand does not copy the data. .reshape(-1,1).expand(-1,out_height * out_width).reshape(-1)
                if torch.cuda.is_available():
                    dim2 = dim2.cuda()
                    dim1 = dim1.cuda()
                    y0 = y0.cuda()
                    y1 = y1.cuda()
                    x0 = x0.cuda()
                    x1 = x1.cuda()
                    base = base.cuda()

                print(x0.size())
                base_y0 = base + y0 * dim2 #base.size() - 365056
                base_y1 = base + y1 * dim2
                idx_a = base_y0 + x0    #x0<etc>.size() - 4380672
                idx_b = base_y1 + x0
                idx_c = base_y0 + x1
                idx_d = base_y1 + x1

                # use indices to lookup pixels in the flat image and restore
                # channels dim
                im = im.permute(0, 2, 3, 1)
                im_flat = im.reshape([-1, num_channels]).float()

                idx_a = idx_a.unsqueeze(-1).long()
                idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
                Ia = torch.gather(im_flat, 0, idx_a)

                idx_b = idx_b.unsqueeze(-1).long()
                idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
                Ib = torch.gather(im_flat, 0, idx_b)

                idx_c = idx_c.unsqueeze(-1).long()
                idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
                Ic = torch.gather(im_flat, 0, idx_c)

                idx_d = idx_d.unsqueeze(-1).long()
                idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
                Id = torch.gather(im_flat, 0, idx_d)

                # and finally calculate interpolated values
                x0_f = x0.float()
                x1_f = x1.float()
                y0_f = y0.float()
                y1_f = y1.float()

                wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
                wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
                wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
                wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
                output = wa * Ia + wb * Ib + wc * Ic + wd * Id

                return output

            def _transform(I, vgrid, scale_h):

                C_img = I.shape[1] # bilo je 1
                
                B, C, H, W = vgrid.size()
                print(vgrid.size())
                x_s_flat = vgrid[ :, 0, ...].reshape([-1])
                y_s_flat = vgrid[ :, 1, ...].reshape([-1])
                out_size = vgrid.shape[2:]
                input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size, scale_h)

                output = input_transformed.reshape([B, H, W, C_img])
                return output

            # scale_h = True
            output = _transform(I, vgrid, scale_h=False)
            if train:
                output = output.permute(0, 3, 1, 2)
            return output

        @classmethod
        def warp_im(cls, I_nchw, flow_nchw, start_n211):
            _, batch_size, _, img_h, img_w = I_nchw.size()
            _, _, _, patch_size_h, patch_size_w = flow_nchw.size()
            patch_indices = cls.get_grid(batch_size, patch_size_h, patch_size_w, start_n211)
            vgrid = patch_indices[:, :2, ...]
            
            # grid_warp = vgrid - flow_nchw
            grid_warp = vgrid + flow_nchw
            pred_I2 = cls.transformer(I_nchw, grid_warp)
            return pred_I2