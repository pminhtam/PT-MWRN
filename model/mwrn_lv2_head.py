from model import common
import torch
import torch.nn as nn

class MWRN_lv2_head(nn.Module):
    def __init__(self):
        super(MWRN_lv2_head, self).__init__()
        self.color_channel = 3

        self.conv2_head_1 = common.BBlock(common.default_conv0,16*self.color_channel,256,3,bn=True)
        self.conv2_head_2 = common.BBlock(common.default_conv0,512,256,3,bn=True)
        self.res2_head = nn.Sequential(*[common.ResBlock(common.default_conv0,256,3) for i in range(4)])
        self.conv2_head_3 = common.BBlock(common.default_conv0,1024,512,3,bn=True)

        self.DWT = common.DWT()

    def forward(self, y2, lv1_head_out=None):
        y = self.conv2_head_1(y2)
        if lv1_head_out == None:
            lv1_head_out = torch.zeros_like(y)
        y = torch.cat([y,lv1_head_out],dim=1)
        y = self.conv2_head_2(y)
        y = self.res2_head(y)
        lv2_head_out_0 = self.DWT(y)
        lv2_head_out = self.conv2_head_3(lv2_head_out_0)
        return lv2_head_out_0, lv2_head_out
