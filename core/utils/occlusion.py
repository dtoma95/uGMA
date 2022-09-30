import torch
from utils.losses import loss_functions
#import torch.nn as nn
#import torch.nn.functional as F

class OcclusionChecker():

    def __init__(self, occ_type='for_back_check', occ_alpha_1=1.0, occ_alpha_2=0.05, sum_abs_or_squar=True, obj_out_all='all'):
        '''
        :param occ_type: method to check occ mask: bidirection check, or froward warping check(not implemented)
        :param occ_alpha_1: threshold
        :param occ_alpha_2: threshold
        :param obj_out_all: occ mask for: (1) all occ area; (2) only moving object occ area; (3) only out-plane occ area.
        '''
        self.occ_type_ls = ['for_back_check', 'forward_warp']
        assert occ_type in self.occ_type_ls
        assert obj_out_all in ['obj', 'out', 'all']
        self.occ_type = occ_type
        self.occ_alpha_1 = occ_alpha_1
        self.occ_alpha_2 = occ_alpha_2
        self.sum_abs_or_squar = True  # found that false is not OK
        self.obj_out_all = obj_out_all

    def __call__(self, flow_f, flow_b, scale=1):
        # 输入进来是可使用的光流
        if self.obj_out_all == 'all':
            if self.occ_type == 'for_back_check':
                occ_1, occ_2 = self._forward_backward_occ_check(flow_fw=flow_f, flow_bw=flow_b, scale=scale)
            elif self.occ_type == 'forward_warp':
                raise ValueError('not implemented')
            else:
                raise ValueError('occ type should be in %s, get %s' % (self.occ_type_ls, self.occ_type))
            return occ_1, occ_2
        elif self.obj_out_all == 'obj':
            if self.occ_type == 'for_back_check':
                occ_1, occ_2 = self._forward_backward_occ_check(flow_fw=flow_f, flow_bw=flow_b, scale=scale)
            elif self.occ_type == 'forward_warp':
                raise ValueError('not implemented')
            elif self.occ_type == 'for_back_check&forward_warp':
                raise ValueError('not implemented')
            else:
                raise ValueError('occ type should be in %s, get %s' % (self.occ_type_ls, self.occ_type))
            out_occ_fw = self.torch_outgoing_occ_check(flow_f)
            out_occ_bw = self.torch_outgoing_occ_check(flow_b)
            obj_occ_fw = self.torch_get_obj_occ_check(occ_mask=occ_1, out_occ=out_occ_fw)
            obj_occ_bw = self.torch_get_obj_occ_check(occ_mask=occ_2, out_occ=out_occ_bw)
            return obj_occ_fw, obj_occ_bw
        elif self.obj_out_all == 'out':
            out_occ_fw = self.torch_outgoing_occ_check(flow_f)
            out_occ_bw = self.torch_outgoing_occ_check(flow_b)
            return out_occ_fw, out_occ_bw
        else:
            raise ValueError("obj_out_all should be in ['obj','out','all'], but get: %s" % self.obj_out_all)

    def _forward_backward_occ_check(self, flow_fw, flow_bw, scale=1):
        """
        In this function, the parameter alpha needs to be improved
        """

        def length_sq_v0(x):
            # torch.sum(x ** 2, dim=1, keepdim=True)
            # temp = torch.sum(x ** 2, dim=1, keepdim=True)
            # temp = torch.pow(temp, 0.5)
            return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
            # return temp

        def length_sq(x):
            # torch.sum(x ** 2, dim=1, keepdim=True)
            temp = torch.sum(x ** 2, dim=1, keepdim=True)
            temp = torch.pow(temp, 0.5)
            # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
            return temp

        if self.sum_abs_or_squar:
            sum_func = length_sq_v0
        else:
            sum_func = length_sq
        mag_sq = sum_func(flow_fw) + sum_func(flow_bw)
        flow_bw_warped = loss_functions.torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
        flow_fw_warped = loss_functions.torch_warp(flow_fw, flow_bw)
        flow_diff_fw = flow_fw + flow_bw_warped
        flow_diff_bw = flow_bw + flow_fw_warped
        occ_thresh = self.occ_alpha_1 * mag_sq + self.occ_alpha_2 / scale
        occ_fw = sum_func(flow_diff_fw) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
        occ_bw = sum_func(flow_diff_bw) < occ_thresh
        # if IF_DEBUG:
        #     temp_ = sum_func(flow_diff_fw)
        #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_fw')
        #     temp_ = sum_func(flow_diff_bw)
        #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_bw')
        #     tools.check_tensor(data=mag_sq, name='check occlusion mask mag_sq')
        #     tools.check_tensor(data=occ_thresh, name='check occlusion mask occ_thresh')
        return occ_fw.float(), occ_bw.float()

    def forward_backward_occ_check(self, flow_fw, flow_bw, alpha1, alpha2, obj_out_all='obj'):
        """
        In this function, the parameter alpha needs to be improved
        """

        def length_sq_v0(x):
            # torch.sum(x ** 2, dim=1, keepdim=True)
            # temp = torch.sum(x ** 2, dim=1, keepdim=True)
            # temp = torch.pow(temp, 0.5)
            
            return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
            # return temp

        def length_sq(x):
            # torch.sum(x ** 2, dim=1, keepdim=True)
            temp = torch.sum(x ** 2, dim=1, keepdim=True)
            temp = torch.pow(temp, 0.5)
            # return torch.sum(torch.pow(x ** 2, 0.5), dim=1, keepdim=True)
            return temp

        if self.sum_abs_or_squar:
            sum_func = length_sq_v0
        else:
            sum_func = length_sq
        mag_sq = sum_func(flow_fw) + sum_func(flow_bw)
        flow_bw_warped = loss_functions.torch_warp(flow_bw, flow_fw)  # torch_warp(img,flow)
        flow_fw_warped = loss_functions.torch_warp(flow_fw, flow_bw)
        flow_diff_fw = flow_fw + flow_bw_warped
        flow_diff_bw = flow_bw + flow_fw_warped
        occ_thresh = alpha1 * mag_sq + alpha2
        occ_fw = sum_func(flow_diff_fw) < occ_thresh  # 0 means the occlusion region where the photo loss we should ignore
        occ_bw = sum_func(flow_diff_bw) < occ_thresh
        occ_fw = occ_fw.float()
        occ_bw = occ_bw.float()
        # if IF_DEBUG:
        #     temp_ = sum_func(flow_diff_fw)
        #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_fw')
        #     temp_ = sum_func(flow_diff_bw)
        #     tools.check_tensor(data=temp_, name='check occlusion mask sum_func flow_diff_bw')
        #     tools.check_tensor(data=mag_sq, name='check occlusion mask mag_sq')
        #     tools.check_tensor(data=occ_thresh, name='check occlusion mask occ_thresh')
        if obj_out_all == 'obj':
            out_occ_fw = self.torch_outgoing_occ_check(flow_fw)
            out_occ_bw = self.torch_outgoing_occ_check(flow_bw)
            occ_fw = self.torch_get_obj_occ_check(occ_mask=occ_fw, out_occ=out_occ_fw)
            occ_bw = self.torch_get_obj_occ_check(occ_mask=occ_bw, out_occ=out_occ_bw)
        return occ_fw, occ_bw

    def _forward_warp_occ_check(self, flow_bw):  # TODO
        return 0

    @classmethod
    def torch_outgoing_occ_check(cls, flow):

        B, C, H, W = flow.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
        flow_x, flow_y = torch.split(flow, 1, 1)
        if flow.is_cuda:
            xx = xx.cuda()
            yy = yy.cuda()
        # tools.check_tensor(flow_x, 'flow_x')
        # tools.check_tensor(flow_y, 'flow_y')
        # tools.check_tensor(xx, 'xx')
        # tools.check_tensor(yy, 'yy')
        pos_x = xx + flow_x
        pos_y = yy + flow_y
        # tools.check_tensor(pos_x, 'pos_x')
        # tools.check_tensor(pos_y, 'pos_y')
        # print(' ')
        # check mask
        outgoing_mask = torch.ones_like(pos_x)
        outgoing_mask[pos_x > W - 1] = 0
        outgoing_mask[pos_x < 0] = 0
        outgoing_mask[pos_y > H - 1] = 0
        outgoing_mask[pos_y < 0] = 0
        return outgoing_mask.float()

    @classmethod
    def torch_get_obj_occ_check(cls, occ_mask, out_occ):
        outgoing_mask = torch.zeros_like(occ_mask)
        if occ_mask.is_cuda:
            outgoing_mask = outgoing_mask.cuda()
        outgoing_mask[occ_mask == 1] = 1
        outgoing_mask[out_occ == 0] = 1
        return outgoing_mask