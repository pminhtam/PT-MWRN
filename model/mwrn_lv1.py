from model import common
import torch
import torch.nn as nn
from model.mwrn_lv1_head import MWRN_lv1_head
from model.mwrn_lv1_tail import MWRN_lv1_tail
from model.mwrn_lv2 import MWRN_lv2

class MWRN_lv1(nn.Module):
    def __init__(self):
        super(MWRN_lv1, self).__init__()
        self.color_channel = 1
        self.lv1_head = MWRN_lv1_head()
        self.lv2 = MWRN_lv2()
        self.lv1_tail = MWRN_lv1_tail()
        self.DWT = common.DWT()
        self.IWT = common.IWT()


    def forward(self, img):
        y1 = self.DWT(img)
        lv1_head_out_0, lv1_head_out = self.lv1_head(y1)
        y2 = self.DWT(y1)
        lv2_out, img_lv2 = self.lv2(y2,lv1_head_out)
        lv1_out = self.lv1_tail(lv2_out, lv1_head_out_0)
        pred = y1 + lv1_out
        pred = self.IWT(pred)
        return pred

if __name__ == "__main__":
    model = MWRN_lv1()
    from torchsummary import summary
    summary(model,(1,256,256))