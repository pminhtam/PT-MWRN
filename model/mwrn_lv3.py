from model import common
import torch
import torch.nn as nn

class MWRN_lv3(nn.Module):
    def __init__(self):
        super(MWRN_lv3, self).__init__()
        self.color_channel = 3

        self.conv3_1 = common.BBlock(common.default_conv0,64*self.color_channel,512,3,bn=True)
        self.conv3_2 = common.BBlock(common.default_conv0,1024,512,3,bn=True)
        self.res3 = nn.Sequential(*[common.ResBlock(common.default_conv0,512,3) for i in range(8)])
        self.conv3_img = common.BBlock(common.default_conv0,512,64*self.color_channel,3,bn=True)


    def forward(self, y3, lv2_head_out=None):
        y = self.conv3_1(y3)
        if lv2_head_out == None:
            lv2_head_out = torch.zeros_like(y)
        y = torch.cat([y,lv2_head_out],dim=1)
        y = self.conv3_2(y)
        lv3_out = self.res3(y)
        img_lv3 = self.conv3_img(y) + y3
        return lv3_out, img_lv3
