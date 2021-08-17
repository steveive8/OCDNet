import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OCDNet(nn.Module):
    def __init__(self, cfg: dict, num_classes: int = 1000, init_weights = True):
        super(OCDNet, self).__init__()

        self.ocds = {}
        self.ids = []
        self.convs = []

        self.cfg = cfg

        self.identitier(cfg)
        self.ocd_emerge(cfg)

        #Advanced
        # self.ocd0 = self.ocds[0][0]
        # self.ocd1 = self.ocds[0][1]
        # self.ocd2 = self.ocds[0][2]
        # self.ocd3 = self.ocds[0][3]
        # self.ocd4 = self.ocds[0][4]
        # self.ocd5 = self.ocds[1][0]
        # self.ocd6 = self.ocds[1][1]
        # self.ocd7 = self.ocds[1][2]
        # self.ocd8 = self.ocds[1][3]
        # self.ocd9 = self.ocds[1][4]
        # self.ocd10 = self.ocds[2][0]
        # self.ocd11 = self.ocds[2][1]
        # self.ocd12 = self.ocds[2][2]
        # self.ocd13 = self.ocds[3][0]
        # self.ocd14 = self.ocds[3][1]
        # self.ocd15 = self.ocds[3][2]
        # self.ocd16 = self.ocds[4][0]
        # self.ocd17 = self.ocds[4][1]
        # self.ocd18 = self.ocds[4][2]
        # self.ocd19 = self.ocds[5][0]
        # self.ocd20 = self.ocds[5][1]
        # self.ocd21 = self.ocds[6][0]
        # self.ocd22 = self.ocds[6][1]
        # self.ocd23 = self.ocds[7][0]

        #Intermediate

        # self.ocd0 = self.ocds[0][0]
        # self.ocd1 = self.ocds[0][1]
        # self.ocd2 = self.ocds[0][2]
        # self.ocd3 = self.ocds[0][3]
        # self.ocd4 = self.ocds[0][4]
        # self.ocd5 = self.ocds[1][0]
        # self.ocd6 = self.ocds[1][1]
        # self.ocd7 = self.ocds[1][2]
        # self.ocd8 = self.ocds[1][3]
        # self.ocd9 = self.ocds[1][4]
        # self.ocd10 = self.ocds[2][0]
        # self.ocd11 = self.ocds[2][1]
        # self.ocd12 = self.ocds[2][2]
        # self.ocd13 = self.ocds[3][0]
        # self.ocd14 = self.ocds[3][1]
        # self.ocd15 = self.ocds[3][2]
        # self.ocd16 = self.ocds[4][0]
        # self.ocd17 = self.ocds[4][1]
        # self.ocd18 = self.ocds[4][2]
        # self.ocd19 = self.ocds[5][0]
        # self.ocd20 = self.ocds[5][1]
        # self.ocd21 = self.ocds[5][2]
        # self.ocd22 = self.ocds[5][3]
        # self.ocd23 = self.ocds[5][4]
        # self.ocd24 = self.ocds[6][0]
        # self.ocd25 = self.ocds[6][1]
        # self.ocd26 = self.ocds[6][2]
        # self.ocd27 = self.ocds[6][3]
        # self.ocd28 = self.ocds[6][4]
        # self.ocd29 = self.ocds[7][0]
        # self.ocd30 = self.ocds[7][1]
        # self.ocd31 = self.ocds[7][2]

        #light

        # self.ocd0 = self.ocds[0][0]
        # self.ocd1 = self.ocds[0][1]
        # self.ocd2 = self.ocds[0][2]
        # self.ocd3 = self.ocds[0][3]
        # self.ocd4 = self.ocds[0][4]
        # self.ocd5 = self.ocds[1][0]
        # self.ocd6 = self.ocds[1][1]
        # self.ocd7 = self.ocds[1][2]
        # self.ocd8 = self.ocds[2][0]
        # self.ocd9 = self.ocds[2][1]
        # self.ocd10 = self.ocds[2][2]
        # self.ocd11 = self.ocds[3][0]
        # self.ocd12 = self.ocds[3][1]
        # self.ocd13 = self.ocds[3][2]
        # self.ocd14 = self.ocds[4][0]
        # self.ocd15 = self.ocds[4][1]
        # self.ocd16 = self.ocds[4][2]
        # self.ocd17 = self.ocds[4][3]
        # self.ocd18 = self.ocds[4][4]
        # self.ocd19 = self.ocds[5][0]
        # self.ocd20 = self.ocds[5][1]
        # self.ocd21 = self.ocds[5][2]

        #light2, 3

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

        
        self.id0 = self.ids[0]
        self.id1 = self.ids[1]
        self.id2 = self.ids[2]
        self.id3 = self.ids[3]
        #self.id4 = self.ids[4]
        #self.id5 = self.ids[5]

        self.conv0 = self.convs[0]
        self.conv1 = self.convs[1]
        self.conv2 = self.convs[2]
        self.conv3 = self.convs[3]
        #self.conv4 = self.convs[4]
        #self.conv5 = self.convs[5]

        
        self.classfier = nn.Sequential(
            nn.Linear(2 * 2 * 64, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()


    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.sequencing(x)

    
    def identitier(self, cfg):
        for i in range(0, len(cfg['kernel']) - 2):
            self.ids += [nn.Parameter(torch.tensor(0.02), requires_grad=True)]
            self.convs += [nn.Conv2d(cfg['channel'][i], cfg['channel'][i+2], 1)]

    
    def sequencing(self, x):
        outs = x
        new_outs = {}

        #get each layer
        for i, ocds in enumerate(self.ocds.values()):
            
            #Initialize
            new_outs[i] = []

            #Set first Tensor as List to Adopt OCD
            if i == 0:
                outs = [outs]

            #Slide all OCDS
            for ocd in ocds:                    
                x = ocd(outs)
                
                #Identitier
                if i > 1:
                    _out = 0
                    for out in new_outs[i - 2]:
                        id = nn.AdaptiveAvgPool2d((x.size(-1), x.size(-1)))(out)
                        _out += id
                    
                    if self.cfg['batch_expanding_gamma'] == True:
                        id = _out * self.ids[i -2]

                    id = self.convs[i - 2](id)
                    x = x + id

                new_outs[i].append(x)
                    
            outs = new_outs[i]

        #print('final out', len(outs), outs[0].shape)
        out = outs[0].view(-1, 2 * 2 * 64)
        x = self.classfier(out)
        return x


    def ocd_emerge(self, cfg):
        for i, kernels in enumerate(cfg['kernel'].values()):
            self.ocds[i] = []

            in_batch = 1 if i == 0 else len(cfg['kernel'][i - 1])

            for kernel in kernels:
                in_channel = cfg['channel'][i - 1] if i != 0 else 3
                if i in cfg['batcher']:
                    self.ocds[i].append(OCD_Strider_Batcher(kernel, 2, in_channel, cfg['channel'][i], in_batch, cfg['batch_expanding_gamma'], cfg['random_batcher']))
                else:
                    self.ocds[i].append(OCD_Strider(kernel, 2, in_channel, cfg['channel'][i], in_batch, cfg['batch_expanding_gamma'], cfg['random_batcher']))
                
                
            

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



class OCD_Strider_Batcher(nn.Module):
    def __init__(self, kernel, stride, in_channel, out_channel, in_batch, gamma, random_batcher):
        super(OCD_Strider_Batcher, self).__init__()

        self.gamma = gamma
        self.random_batcher = random_batcher
        if gamma == True:
            self.w1 = nn.Parameter(torch.tensor(1 / in_batch), requires_grad=True)
            #self.w2 = nn.Parameter(torch.tensor(1 / in_batch), requires_grad=True)

        self.in_batch = in_batch
        self.out_channel = out_channel
        
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding = 1)
        self.sq = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        outs = x
        mins = []
        new_outs = []
        for out in outs:            
            out = self.conv(out)
            new_outs.append(out)
            _min = out.size(-1)
            mins.append(_min)
        min_size = min(mins)

        _outs = 0
        if self.random_batcher:
            randoms = []
            for out in new_outs:
                out = nn.AdaptiveAvgPool2d((min_size, min_size))(out)
                randoms.append(out)
            cats = torch.cat(randoms, 0)
            indices = torch.randint(0, cats.size(0), (x[0].size(0),))
            _outs = []
            for index in indices:
                _outs.append(cats[index])
            _outs = torch.stack(_outs)
            #_outs = randoms[0]
        else:
            for out in new_outs:
                out = nn.AdaptiveAvgPool2d((min_size, min_size))(out)
                _outs+= out

            if self.gamma:
                _outs = _outs * self.w1# * self.w2
        _outs = self.sq(_outs)
        return _outs


class OCD_Strider(nn.Module):
    def __init__(self, kernel, stride, in_channel, out_channel, in_batch, gamma, random_batcher):
        super(OCD_Strider, self).__init__()
        
        self.gamma = gamma
        self.random_batcher = random_batcher
        self.in_batch = in_batch
        if gamma == True:
            self.w1 = nn.Parameter(torch.tensor(1 / in_batch), requires_grad=True)
            #self.w2 = nn.Parameter(torch.tensor(1 / in_batch), requires_grad=True)

        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding = 1)
        self.out_channel = out_channel

    def forward(self, x):
        outs = x
        mins = []
        new_outs = []
        for out in outs:            
            out = self.conv(out)
            new_outs.append(out)

            _min = out.size(-1)
            mins.append(_min)
        min_size = min(mins)

        _outs = 0
        if self.random_batcher:
            randoms = []
            for out in new_outs:
                out = nn.AdaptiveAvgPool2d((min_size, min_size))(out)
                randoms.append(out)
            cats = torch.cat(randoms, 0)
            #print(cats.shape, cats.size(0))
            indices = torch.randint(0, cats.size(0), (x[0].size(0),))
            _outs = []
            for index in indices:
                _outs.append(cats[index])
            _outs = torch.stack(_outs)
            #_outs = randoms[0]
        else:
            for out in new_outs:
                out = nn.AdaptiveAvgPool2d((min_size, min_size))(out)
                _outs+= out

            if self.gamma:
                _outs = _outs * self.w1# * self.w2
        return _outs




cfg = {
    'advanced': {
        'kernel': {
            0: [23, 19, 15, 11, 7],
            1: [11, 7, 3, 1, 1],
            2: [5, 3, 1],
            3: [3, 1, 1],
            4: [3, 2, 1],
            5: [3, 1],
            6: [3 ,1],
            7: [1]
        },
        'channel': {
            0: 64,
            1: 64,
            2: 128,
            3: 256,
            4: 512,
            5: 256,
            6: 128,
            7: 64,
        },
        'batcher': [0, 1, 2, 3, 4, 5, 6, 7],
        'batch_expanding_gamma': True,
        'random_batcher': False,
    },
    'intermediate': {
        'kernel': {
            0: [3, 3, 2, 2, 1],
            1: [5, 3, 2, 2, 1],
            2: [7, 5, 3],
            3: [9, 7, 9],
            4: [7, 5, 3],
            5: [3, 3, 2, 2, 1],
            6: [5, 3, 2, 2, 1],
            7: [1, 1, 1]
        },
        'channel': {
            0: 64,
            1: 64,
            2: 128,
            3: 256,
            4: 512,
            5: 256,
            6: 128,
            7: 64,
        },
        'batcher': [0, 1, 2, 3, 4, 5, 6, 7],
        'batch_expanding_gamma': True,
        'random_batcher': False,
    },
    'intermediate_light': {
        'kernel': {
            0: [2, 2, 2, 1, 1],
            1: [3, 3, 2, 2, 1],
            2: [7, 5, 3],
            3: [9, 7, 9],
            4: [7, 5, 3],
            5: [3, 3, 2, 2, 1],
            6: [2, 2, 2, 2, 1],
            7: [1, 1, 1]
        },
        'channel': {
            0: 64,
            1: 64,
            2: 128,
            3: 256,
            4: 512,
            5: 256,
            6: 128,
            7: 64,
        },
        'batcher': [0, 1, 2, 3, 4, 5, 6, 7],
        'batch_expanding_gamma': True,
        'random_batcher': False,
    },
    'light': {
        'kernel': {
            0: [5, 3, 2, 2, 1],
            1: [7, 5, 3],
            2: [9, 7, 9],
            3: [7, 5, 3],
            4: [3, 3, 2, 2, 1],
            5: [1, 1, 1]
        },
        'channel': {
            0: 64,
            1: 128,
            2: 512,
            3: 512,
            4: 128,
            5: 64,
        },
        'batcher': [1, 2, 3, 4, 5, 6],
        'batch_expanding_gamma': True,
        'random_batcher': False,
    },
    'light_2': {
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
        'batcher': [1, 2, 3, 4, 5],
        'batch_expanding_gamma': True,
        'random_batcher': False,
    },
    'light_3': {
        'kernel': {
            0: [1],
            1: [3, 1],
            2: [3, 2, 1],
            3: [3, 2, 1],
            4: [3, 1],
            5: [1],
        },
        'channel': {
            0: 64,
            1: 128,
            2: 512,
            3: 512,
            4: 128,
            5: 64,
        },
        'batcher': [0, 1, 2, 3, 4, 5],
        'batch_expanding_gamma': True,
        'random_batcher': False,
    },
    'light_v': {
        'kernel': {
            0: [2],
            1: [3, 2],
            2: [5, 3, 2],
            3: [3, 2, 1],
            4: [3, 1],
            5: [1],
        },
        'channel': {
            0: 64,
            1: 128,
            2: 512,
            3: 512,
            4: 128,
            5: 64,
        },
        'batcher': [0, 1, 2, 3, 4, 5],
        'batch_expanding_gamma': True,
        'random_batcher': False,
    }
}

def OCD10(cfg = cfg['light_v'], num_classes = 1000, init_weights = True):
    return OCDNet(cfg, num_classes, init_weights)


            #new_outs = new_outs.view(-1, self.out_channel * min_size * min_size)
        #print(new_outs.shape)
        #new_outs = torch.transpose(new_outs, 0, 1)
        #print(new_outs.shape)
        #new_outs = self.linear(new_outs)
        #print(new_outs.shape)
        #new_outs = torch.transpose(new_outs, 0, 1)
        #print(new_outs.shape)
        #new_outs = new_outs.view(-1, self.out_channel, min_size, min_size)
        #print(new_outs.shape)


        