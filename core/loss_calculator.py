import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils.spectral_norm import spectral_norm

import numpy as np
#from utils.tools import tools
from utils.losses import loss_functions


class LossCalculatior():
    class config():
        def __init__(self):
            # occ loss choose
            self.occ_type = 'for_back_check'
            self.alpha_1 = 0.1
            self.alpha_2 = 0.5
            self.occ_check_obj_out_all = 'obj'  # if boundary dilated warping is used, here should be obj
            self.stop_occ_gradient = False
            self.smooth_level = 'final'  # final or 1/4
            self.smooth_type = 'edge'  # edge or delta
            self.smooth_order_1_weight = 1
            # smooth loss
            self.smooth_order_2_weight = 0
            # photo loss type add SSIM
            self.photo_loss_type = 'SSIM'  # abs_robust, charbonnier,L1, SSIM
            self.photo_loss_delta = 0.4
            self.photo_loss_use_occ = True
            self.photo_loss_census_weight = 1
            # use cost volume norm
            self.if_norm_before_cost_volume = False
            self.norm_moments_across_channels = True
            self.norm_moments_across_images = True
            self.multi_scale_distillation_weight = 0
            self.multi_scale_distillation_style = 'upup'  # down,upup,
            # 'down', 'upup', 'updown'
            self.multi_scale_distillation_occ = True  # if consider occlusion mask in multiscale distilation
            self.if_froze_pwc = False
            self.input_or_sp_input = 1  # use raw input or special input for photo loss
            self.if_use_boundary_warp = True  # if use the boundary dilated warping

            self.if_sgu_upsample = False  # if use sgu upsampling
            self.if_use_cor_pytorch = False  # use my implementation of correlation layer by pytorch. only for test model in cpu(corr layer cuda is not compiled)

        def __call__(self, ):
            # return PWCNet_unsup_irr_bi_v5_4(self)
            return UPFlow_net(self)


    def __init__(self, conf = config()):
        '''
        Class that deals with all types of loss calculation, Parameters are defined using the config class.
        '''
        self.conf = conf

    def __call__(self, output_dict, im1_ori, im2_ori, flow_f, flow_b):
        
        #flow_f = output_dict['flow_f']
        #flow_b = output_dict['flow_b']
        #iters = int(flow_f.size()[0]/im1_ori.size()[0])

       # im1_ori = torch.repeat_interleave(im1_ori, iters, dim=0)
       # im2_ori = torch.repeat_interleave(im2_ori, iters, dim=0)


        #occ_fw = torch.repeat_interleave(output_dict['occ_fw'], iters, dim=0)
        #occ_bw = torch.repeat_interleave(output_dict['occ_bw'], iters, dim=0)
        occ_fw = output_dict['occ_fw']
        occ_bw = output_dict['occ_bw']

        this_loss = {}
        # === smooth loss
        if self.conf.smooth_level == 'final':
            s_flow_f, s_flow_b = flow_f, flow_b
            s_im1, s_im2 = im1_ori, im2_ori
        elif self.conf.smooth_level == '1/4':
            s_flow_f, s_flow_b = flows[0]  # flow in 1/4 scale
            _, _, temp_h, temp_w = s_flow_f.size()
            s_im1 = F.interpolate(im1_ori, (temp_h, temp_w), mode='area')
            s_im2 = F.interpolate(im2_ori, (temp_h, temp_w), mode='area')
        else:
            raise ValueError('wrong smooth level choosed: %s' % self.smooth_level)
        smooth_loss = 0
        # 1 order smooth loss
        if self.conf.smooth_order_1_weight > 0:
            if self.conf.smooth_type == 'edge':
                smooth_loss += self.conf.smooth_order_1_weight * self.edge_aware_smoothness_order1(img=s_im1, pred=s_flow_f)
                smooth_loss += self.conf.smooth_order_1_weight * self.edge_aware_smoothness_order1(img=s_im2, pred=s_flow_b)
            elif self.conf.smooth_type == 'delta':
                smooth_loss += self.conf.smooth_order_1_weight * self.flow_smooth_delta(flow=s_flow_f, if_second_order=False)
                smooth_loss += self.conf.smooth_order_1_weight * self.flow_smooth_delta(flow=s_flow_b, if_second_order=False)
            else:
                raise ValueError('wrong smooth_type: %s' % self.conf.smooth_type)

        # 2 order smooth loss
        if self.conf.smooth_order_2_weight > 0:
            if self.conf.smooth_type == 'edge':
                smooth_loss += self.conf.smooth_order_2_weight * self.edge_aware_smoothness_order2(img=s_im1, pred=s_flow_f)
                smooth_loss += self.conf.smooth_order_2_weight * self.edge_aware_smoothness_order2(img=s_im2, pred=s_flow_b)
            elif self.conf.smooth_type == 'delta':
                smooth_loss += self.conf.smooth_order_2_weight * self.flow_smooth_delta(flow=s_flow_f, if_second_order=True)
                smooth_loss += self.conf.smooth_order_2_weight * self.flow_smooth_delta(flow=s_flow_b, if_second_order=True)
            else:
                raise ValueError('wrong smooth_type: %s' % self.conf.smooth_type)
        this_loss['smooth_loss'] = smooth_loss
        torch.cuda.empty_cache()
        # === photo loss
        if self.conf.if_use_boundary_warp:
            im1_s, im2_s, start_s = im1_ori, im2_ori, 0  # the image before cropping
            im1_warp = loss_functions.boundary_dilated_warp.warp_im(im2_s, flow_f, start_s)  # warped im1 by forward flow and im2
            im2_warp = loss_functions.boundary_dilated_warp.warp_im(im1_s, flow_b, start_s)
        else:
            im1_warp = loss_functions.torch_warp(im2_ori, flow_f)  # warped im1 by forward flow and im2
            im2_warp = loss_functions.torch_warp(im1_ori, flow_b)
        # photo loss
        torch.cuda.empty_cache()
        if self.conf.stop_occ_gradient:
            occ_fw, occ_bw = occ_fw.clone().detach(), occ_bw.clone().detach()
        photo_loss = self.photo_loss_multi_type(im1_ori, im1_warp, occ_fw, photo_loss_type=self.conf.photo_loss_type,
                                                            photo_loss_delta=self.conf.photo_loss_delta, photo_loss_use_occ=self.conf.photo_loss_use_occ)
        photo_loss += self.photo_loss_multi_type(im2_ori, im2_warp, occ_bw, photo_loss_type=self.conf.photo_loss_type,
                                                            photo_loss_delta=self.conf.photo_loss_delta, photo_loss_use_occ=self.conf.photo_loss_use_occ)
        this_loss['photo_loss'] = photo_loss
        #this_loss['im1_warp'] = im1_warp
        #this_loss['im2_warp'] = im2_warp
        torch.cuda.empty_cache()
        # === census loss
        if self.conf.photo_loss_census_weight > 0:
            census_loss = loss_functions.census_loss_torch(img1=im1_ori, img1_warp=im1_warp, mask=occ_fw, q=self.conf.photo_loss_delta,
                                                            charbonnier_or_abs_robust=False, if_use_occ=self.conf.photo_loss_use_occ, averge=True) + \
                            loss_functions.census_loss_torch(img1=im2_ori, img1_warp=im2_warp, mask=occ_bw, q=self.conf.photo_loss_delta,
                                                            charbonnier_or_abs_robust=False, if_use_occ=self.conf.photo_loss_use_occ, averge=True)
            census_loss *= self.conf.photo_loss_census_weight
        else:
            census_loss = None
        this_loss['census_loss'] = census_loss

        # === multi scale distillation loss
        # REMOVED
        torch.cuda.empty_cache()
        return this_loss


    @classmethod
    def weighted_ssim(cls, x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
        """Computes a weighted structured image similarity measure.
        Args:
          x: a batch of images, of shape [B, C, H, W].
          y:  a batch of images, of shape [B, C, H, W].
          weight: shape [B, 1, H, W], representing the weight of each
            pixel in both images when we come to calculate moments (means and
            correlations). values are in [0,1]
          c1: A floating point number, regularizes division by zero of the means.
          c2: A floating point number, regularizes division by zero of the second
            moments.
          weight_epsilon: A floating point number, used to regularize division by the
            weight.

        Returns:
          A tuple of two pytorch Tensors. First, of shape [B, C, H-2, W-2], is scalar
          similarity loss per pixel per channel, and the second, of shape
          [B, 1, H-2. W-2], is the average pooled `weight`. It is needed so that we
          know how much to weigh each pixel in the first tensor. For example, if
          `'weight` was very small in some area of the images, the first tensor will
          still assign a loss to these pixels, but we shouldn't take the result too
          seriously.
        """

        def _avg_pool3x3(x):
            # tf kernel [b,h,w,c]
            return F.avg_pool2d(x, (3, 3), (1, 1))
            # return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')

        if c1 == float('inf') and c2 == float('inf'):
            raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                             'likely unintended.')
        average_pooled_weight = _avg_pool3x3(weight)
        weight_plus_epsilon = weight + weight_epsilon
        inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

        def weighted_avg_pool3x3(z):
            wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
            return wighted_avg * inverse_average_pooled_weight

        mu_x = weighted_avg_pool3x3(x)
        mu_y = weighted_avg_pool3x3(y)
        sigma_x = weighted_avg_pool3x3(x ** 2) - mu_x ** 2
        sigma_y = weighted_avg_pool3x3(y ** 2) - mu_y ** 2
        sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
        if c1 == float('inf'):
            ssim_n = (2 * sigma_xy + c2)
            ssim_d = (sigma_x + sigma_y + c2)
        elif c2 == float('inf'):
            ssim_n = 2 * mu_x * mu_y + c1
            ssim_d = mu_x ** 2 + mu_y ** 2 + c1
        else:
            ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
            ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        result = ssim_n / ssim_d
        return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight

    @classmethod
    def edge_aware_smoothness_order1(cls, img, pred):
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

    @classmethod
    def edge_aware_smoothness_order2(cls, img, pred):
        def gradient_x(img, stride=1):
            gx = img[:, :, :-stride, :] - img[:, :, stride:, :]
            return gx

        def gradient_y(img, stride=1):
            gy = img[:, :, :, :-stride] - img[:, :, :, stride:]
            return gy

        pred_gradients_x = gradient_x(pred)
        pred_gradients_xx = gradient_x(pred_gradients_x)
        pred_gradients_y = gradient_y(pred)
        pred_gradients_yy = gradient_y(pred_gradients_y)

        image_gradients_x = gradient_x(img, stride=2)
        image_gradients_y = gradient_y(img, stride=2)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_x = torch.abs(pred_gradients_xx) * weights_x
        smoothness_y = torch.abs(pred_gradients_yy) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

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
    def photo_loss_multi_type(cls, x, y, occ_mask, photo_loss_type='abs_robust',  # abs_robust, charbonnier,L1, SSIM
                              photo_loss_delta=0.4, photo_loss_use_occ=False,
                              ):
        occ_weight = occ_mask
        if photo_loss_type == 'abs_robust':
            photo_diff = x - y
            loss_diff = (torch.abs(photo_diff) + 0.01).pow(photo_loss_delta)
        elif photo_loss_type == 'charbonnier':
            photo_diff = x - y
            loss_diff = ((photo_diff) ** 2 + 1e-6).pow(photo_loss_delta)
        elif photo_loss_type == 'L1':
            photo_diff = x - y
            loss_diff = torch.abs(photo_diff + 1e-6)
        elif photo_loss_type == 'SSIM':
            loss_diff, occ_weight = cls.weighted_ssim(x, y, occ_mask)
        else:
            raise ValueError('wrong photo_loss type: %s' % photo_loss_type)

        if photo_loss_use_occ:
            photo_loss = torch.sum(loss_diff * occ_weight) / (torch.sum(occ_weight) + 1e-6)
        else:
            photo_loss = torch.mean(loss_diff)
        return photo_loss

    