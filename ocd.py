import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OCDNet(nn.Module):
    def __init__(self, cfg: dict, num_classes: int = 1000, init_weights = True):
        super(OCDNet, self).__init__()

        self.ocds = {}

        self.ocd_emerge(cfg)

        self.ocd0 = self.ocds[0][0]
        self.ocd1 = self.ocds[1][0]
        self.ocd2 = self.ocds[1][1]
        self.ocd3 = self.ocds[2][0]
        self.ocd4 = self.ocds[2][1]
        self.ocd5 = self.ocds[2][2]
        self.ocd6 = self.ocds[3][0]
        self.ocd7 = self.ocds[3][1]
        self.ocd8 = self.ocds[3][2]
        self.ocd9 = self.ocds[4][0]
        self.ocd10 = self.ocds[4][1]
        self.ocd11 = self.ocds[5][0]
        # self.ocd12 = self.ocds[5][2]
        # self.ocd13 = self.ocds[5][0]
        # self.ocd14 = self.ocds[5][1]
        # self.ocd15 = self.ocds[6][0]
        # self.ocd16 = self.ocds[6][1]
        # self.ocd17 = self.ocds[7][0]



        
        self.classfier = nn.Sequential(
            nn.Linear(2 * 2 * 64, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

        if init_weights:
            self._initialize_weights()


    def ocd_emerge(self, cfg):
        for i, kernels in enumerate(cfg['kernel'].values()):
            self.ocds[i] = []
            for kernel in kernels:
                in_channel = cfg['channel'][i - 1] if i != 0 else 3
                self.ocds[i].append(OCD(kernel, 2, in_channel, cfg['channel'][i]))



    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.sequencing(x)

    
    def sequencing(self, x):
        outs = x
        new_outs = {}

        #get each layer
        for i, ocds in enumerate(self.ocds.values()):
            new_outs[i] = []
            #ocds for each layer
            if i == 0:
                outs = [outs]
            for ocd in ocds:
                x = ocd(outs)
                new_outs[i].append(x)
                    
            outs = new_outs[i]

        print('final out', len(outs), outs[0].shape)
        out = outs[0].view(-1, 2 * 2* 64)
        x = self.classfier(out)
        return x
            

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




class OCD_Strider(nn.Module):
    def __init__(self, kernel, stride, in_channel, out_channel):
        super(OCD_Strider, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride = stride , padding = 1)

    def forward(self, x):
        outs = x
        new_outs = []
        for out in outs:
            out = self.conv(out)
            new_outs.append(out)
        return new_outs


class OCD_Bottle(nn.Module):
    def __init__(self, out_channel):
        super(OCD_Bottle, self).__init__()
        
        self.sq = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        
    def forward(self, x):
        outs = x
        new_outs = []
        mins = []

        for i in outs:
            a = i.size(-1)
            mins.append(a)
        min_size = min(mins)

        for out in outs:
            out = nn.AdaptiveAvgPool2d((min_size, min_size))(out)
            out = self.sq(out)
            new_outs.append(out)
        
        new_out = 0
        for l in new_outs:
            new_out += l
        return new_out

        

class OCD(nn.Module):
    def __init__(self, kernel, stride, in_channel, out_channel):
        super(OCD, self).__init__()

        self.OCD_Strider = OCD_Strider(kernel, stride, in_channel, out_channel)
        self.OCD_Bottle = OCD_Bottle(out_channel)

    def forward(self, x):
        return nn.Sequential(
            self.OCD_Strider,
            self.OCD_Bottle
        )(x)





cfg = {
    'advanced': {
        'kernel': {
            0: [3],
            1: [5, 3],
            2: [9, 7, 5],
            3: [9, 7, 5],
            4: [5, 3],
            5: [3],
        },
        'channel': {
            0: 64,
            1: 128,
            2: 512,
            3: 512,
            4: 128,
            5: 64,
        },
    }
}

def OCD10(cfg = cfg['advanced'], num_classes = 1000, init_weights = True):
    return OCDNet(cfg, num_classes, init_weights)