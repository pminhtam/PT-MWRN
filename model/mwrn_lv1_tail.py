from model import common
import torch
import torch.nn as nn

class MWRN_lv1_tail(nn.Module):
    def __init__(self):
        super(MWRN_lv1_tail, self).__init__()
        self.color_channel = 3

        self.conv1_tail_1 = common.BBlock(common.default_conv0,256,640,3,bn=True)
        self.IWT = common.IWT()
        self.res1_tail = nn.Sequential(*[common.ResBlock(common.default_conv0,160,3) for i in range(4)])
        self.conv1_tail_2 = common.BBlock(common.default_conv0,160,4*self.color_channel,3,bn=True)


    def forward(self, lv2_out, lv1_head_out_0):
        y = self.conv1_tail_1(lv2_out)
        y = y + lv1_head_out_0
        y = self.IWT(y)
        y = self.res1_tail(y)
        lv1_out = self.conv1_tail_2(y)
        return lv1_out
