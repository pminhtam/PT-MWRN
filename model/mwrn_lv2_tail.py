from model import common
import torch
import torch.nn as nn

class MWRN_lv2_tail(nn.Module):
    def __init__(self):
        super(MWRN_lv2_tail, self).__init__()
        self.color_channel = 3

        self.conv2_tail_1 = common.BBlock(common.default_conv,512*self.color_channel,1024*self.color_channel,3,bn=True)
        self.IWT = common.IWT()
        self.res2_tail = nn.Sequential(*[common.ResBlock(common.default_conv,256*self.color_channel,3) for i in range(4)])
        self.conv2_tail_img = common.BBlock(common.default_conv,256*self.color_channel,16*self.color_channel,3,bn=True)


    def forward(self, lv3_out, lv2_head_out_0):
        y = self.conv2_tail_1(lv3_out)
        y = y + lv2_head_out_0
        y = self.IWT(y)
        lv2_out = self.res2_tail(y)
        img_lv2 = self.conv2_tail_img(lv2_out)
        return lv2_out, img_lv2
