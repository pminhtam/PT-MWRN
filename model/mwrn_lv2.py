from model import common
import torch
import torch.nn as nn
from model.mwrn_lv2_head import MWRN_lv2_head
from model.mwrn_lv2_tail import MWRN_lv2_tail
from model.mwrn_lv3 import MWRN_lv3

class MWRN_lv2(nn.Module):
    def __init__(self):
        super(MWRN_lv2, self).__init__()
        self.color_channel = 1

        self.lv2_head = MWRN_lv2_head()
        self.lv3 = MWRN_lv3()
        self.lv2_tail = MWRN_lv2_tail()
        self.DWT = common.DWT()
        self.IWT = common.IWT()


    def forward(self, y2, lv1_head_out):
        lv2_head_out_0, lv2_head_out = self.lv2_head(y2, lv1_head_out)
        y3 = self.DWT(y2)
        lv3_out, _ = self.lv3(y3, lv2_head_out)
        lv2_out, img_lv2 = self.lv2_tail(lv3_out, lv2_head_out_0)
        return lv2_out, img_lv2
