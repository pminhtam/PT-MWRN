from model import common
import torch
import torch.nn as nn

class MWRN_lv1_head(nn.Module):
    def __init__(self):
        super(MWRN_lv1_head, self).__init__()
        self.color_channel = 1

        self.conv1_head_1 = common.BBlock(common.default_conv,4*self.color_channel,160*self.color_channel,3,bn=True)
        self.res1_head = nn.Sequential(*[common.ResBlock(common.default_conv,160*self.color_channel,3) for i in range(4)])
        self.conv1_head_2 = common.BBlock(common.default_conv,640*self.color_channel,256*self.color_channel,3,bn=True)

        self.DWT = common.DWT()

    def forward(self, y1):
        y = self.conv1_head_1(y1)
        y = self.res1_head(y)
        lv1_head_out_0 = self.DWT(y)
        lv1_head_out = self.conv1_head_2(lv1_head_out_0)
        return lv1_head_out_0, lv1_head_out
