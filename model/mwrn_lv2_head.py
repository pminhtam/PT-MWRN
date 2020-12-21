from model import common
import torch
import torch.nn as nn

class MWRN_lv2_head(nn.Module):
    def __init__(self):
        super(MWRN_lv2_head, self).__init__()
        self.color_channel = 1

        self.conv2_head_1 = common.BBlock(common.default_conv,16*self.color_channel,256*self.color_channel,3,bn=True)
        self.conv2_head_2 = common.BBlock(common.default_conv,512*self.color_channel,256*self.color_channel,3,bn=True)
        self.res2_head = nn.Sequential(*[common.ResBlock(common.default_conv,256*self.color_channel,3) for i in range(4)])
        self.conv2_head_3 = common.BBlock(common.default_conv,1024*self.color_channel,512*self.color_channel,3,bn=True)

        self.DWT = common.DWT()

    def forward(self, y2, lv1_head_out):
        y2 = self.conv2_head_1(y2)
        y = torch.cat([y2,lv1_head_out],dim=1)
        y = self.conv2_head_2(y)
        y = self.res2_head(y)
        lv2_head_out_0 = self.DWT(y)
        lv2_head_out = self.conv2_head_3(lv2_head_out_0)
        return lv2_head_out_0, lv2_head_out
