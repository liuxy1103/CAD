from network_parts import Inconv, Down, Up, Outconv, Upself
import torch
import torch.nn as nn


class CoDetectionCNN(nn.Module):
    def __init__(self, n_channels=1, n_classes=16, filter_channel=16, sig=False):
        super().__init__()
        self.inc = Inconv(n_channels, filter_channel)

        self.down = nn.ModuleList([Down(filter_channel, filter_channel*2)])
        self.down.append(Down(filter_channel*4, filter_channel*4))
        self.down.append(Down(filter_channel*4, filter_channel*8))
        self.down.append(Down(filter_channel*8, filter_channel*8))

        self.up1 = Up(filter_channel*16, filter_channel*4)
        self.up2 = Up(filter_channel*8, filter_channel*2)
        self.up3_t = Up(filter_channel*4, filter_channel)
        self.up3_tn = Up(filter_channel*4, filter_channel)
        self.up4_t = Up(filter_channel*2, 32)
        self.up4_tn = Up(filter_channel*2, 32)
        self.outc_t = Outconv(32, n_classes, sig=sig)
        self.outc_tn = Outconv(32, n_classes, sig=sig)

    def forward(self, x):
        x_inp1 = x[:, 0:1, :, :]
        x_inp2 = x[:, 1::, :, :]
        t_enc, tn_enc, dec = ([0] * 2 for _ in range(3))
        enc = [0] * 4

        t_enc[0] = self.inc(x_inp1)

        tn_enc[0] = self.inc(x_inp2)
        t_enc[1] = self.down[0](t_enc[0])
        tn_enc[1] = self.down[0](tn_enc[0])
        enc[0] = torch.cat([t_enc[1], tn_enc[1]], dim=1)
        for i in range(3):
            enc[i + 1] = self.down[i + 1](enc[i])

        dec = self.up1(enc[-1], enc[-2])
        dec = self.up2(dec, enc[-3])

        t_dec = self.up3_t(dec, t_enc[-1])
        tn_dec = self.up3_tn(dec, tn_enc[-1])

        t_dec = self.up4_t(t_dec, t_enc[-2])
        tn_dec = self.up4_tn(tn_dec, tn_enc[-2])

        pred_t = self.outc_t(t_dec)
        pred_tn = self.outc_tn(tn_dec)
        return pred_t, pred_tn

if __name__ == "__main__":
    import numpy as np
    # from model.model_para import model_structure
    from ptflops import get_model_complexity_info
    # from model.model_para import model_structure
    x = torch.rand((1, 2, 1024, 1024)).cuda()
    # x = torch.rand((1, 2, 520, 520)).cuda()
    net = CoDetectionCNN(n_channels=1, n_classes=16, filter_channel=16,sig=False).cuda()
    # pred_t, pred_tn = net(x)
    # # model_structure(net)
    # print(pred_t.shape)
    # print(pred_tn.shape)
    macs, params = get_model_complexity_info(net, (2, 1024, 1024), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<20}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<20}'.format('Number of parameters: ', params))


    from torchprofile import profile_macs
    macs = profile_macs(net, x)
    print('model flops (G):', macs / 1.e9, 'input_size:', x.shape)