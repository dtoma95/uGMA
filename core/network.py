import torch
import torch.nn as nn
import torch.nn.functional as F

from update import GMAUpdateBlock
from extractor import BasicEncoder
from corr import CorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from gma import Attention, Aggregate
from utils.occlusion import OcclusionChecker
from loss_calculator import LossCalculatior

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFTGMA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)
        self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)
   
        self.occ_check_model = OcclusionChecker()
        self.loss_calculator = LossCalculatior()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def oneway_forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            # attention, att_c, att_p = self.att(inp)
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
        return flow_predictions

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow in both directions and calculate losses"""

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        output_dict = {}
        output_dict['image1'] = image1
        output_dict['image2'] = image2
        
        
        predictions_f = self.oneway_forward(image1, image2, iters=iters, flow_init=flow_init, upsample=upsample, test_mode=test_mode)
        output_dict['flow_f'] = predictions_f[-1]
        output_dict['predictions_f'] = predictions_f
        if test_mode == True:
            return [], output_dict['flow_f'] 

        predictions_b = self.oneway_forward(image2, image1, iters=iters, flow_init=flow_init, upsample=upsample, test_mode=test_mode)
        output_dict['predictions_b'] = predictions_b
        output_dict['flow_b'] = predictions_b[-1]

        # OCCLUSION MASK (always use the most recent prediction for mask)
        occ_fw, occ_bw = self.occ_check_model(flow_f=output_dict['flow_f'], flow_b=output_dict['flow_b'] )

        output_dict['occ_fw'] = occ_fw
        output_dict['occ_bw'] = occ_bw

        self.sequence_loss(output_dict, image1, image2, predictions_f, predictions_b)

        return output_dict
       

    def sequence_loss(self, output_dict, image1, image2, predictions_f, predictions_b):
         #  ===========SEQUENCE LOSS=======================================
        output_dict['losses'] = []

        #predictions_f = torch.stack(predictions_f)
        #predictions_b = torch.stack(predictions_b)
       # predictions_f = torch.flatten(predictions_f, start_dim=0, end_dim=1)
        #predictions_b = torch.flatten(predictions_b, start_dim=0, end_dim=1)
        
        #print(predictions_f.size())
        # CALCULATE LOSSES
        gamma = 0.8
        n_predictions = len(predictions_f)
        
        this_loss = {}
        for i in range(len(predictions_f)):
            
            i_weight = gamma**(n_predictions - i - 1)
            this_loss_temp = self.loss_calculator(output_dict, image1, image2, predictions_f[0], predictions_b[0])
            if i == 0:
                for key in this_loss_temp.keys():
                    this_loss[key] = i_weight*this_loss_temp[key]
            else:
                for key in this_loss_temp.keys():
                    this_loss[key] += i_weight*this_loss_temp[key]

        output_dict['losses'].append(this_loss)
        return output_dict
    