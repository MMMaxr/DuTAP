import torch.nn as nn
import torch
from utils.vgg import vgg13_bn, vgg16_bn, vgg19_bn, vgg11_bn
from utils.tools import normalize_timewise
from utils.vit import ViT,TFormer

class DuTAP96(nn.Module):
    def __init__(self, n_class, channels):
        super().__init__()
        self.adpavg = nn.AdaptiveAvgPool2d((4, 10))
        self.prompt1 = nn.Parameter(torch.randn(1, 1,1,103))
        self.prompt2 = nn.Parameter(torch.randn(1, 1,1,206))
        self.prompt3 = nn.Parameter(torch.randn(1, 1,1,197))
        self.transformer1 = ViT(num_patch=512, patch_dim=468, dim=128, depth=1, heads=4, mlp_dim=256,
                                pool='cls', dim_head=128, dropout=0., emb_dropout=0.)
        self.transformer2 = TFormer(num_patch=512, patch_dim=48, dim=128, depth=1, heads=4, mlp_dim=256,
                                 pool='cls', dim_head=128, dropout=0., emb_dropout=0.)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

        self.elu = nn.ELU(inplace=True)
        self.classifier = nn.Sequential(
            nn.Linear(256+128*3, n_class),

        )
        self.softmax = nn.Softmax(dim=1)
        self.vgg1 = vgg16_bn()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.block1 = nn.Sequential(
                    nn.Conv2d(513, 256, kernel_size=(1, 4), stride=(1, 2)),
                    nn.ELU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=(1, 2), stride=(1, 1)),
                    nn.ELU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 48)),
                )

    def forward(self, pharse1, phrase2, phrase3):
        B = phrase2.size(0)
        C = 512
        phrase = torch.cat((pharse1, phrase2, phrase3), dim=2)
        input = phrase.permute(0, 1,3,2)
        input = normalize_timewise(input)
        feature0 = torch.cat(
            [input[:, :, :, :3000],  input[:, :, :, 3000:9000], input[:, :, :, 9000:]], dim=-1)
        #global stream
        feature1_ = self.vgg1(feature0)
        feature1 = self.adpavg(feature1_)
        feature = self.cat_conv(feature1)

        #task stream
        data = feature1_
        phrase1 = data[:,:,:,:103]
        phrase2 = data[:,:,:,84:290]
        phrase3 = data[:,:,:,271:]
        phrase1 = torch.cat([phrase1,self.prompt1.expand(input.size(0), -1, -1,-1)],dim=1)
        phrase2 = torch.cat([phrase2,self.prompt2.expand(input.size(0), -1, -1,-1)],dim=1)
        phrase3 = torch.cat([phrase3,self.prompt3.expand(input.size(0), -1, -1,-1)],dim=1)
        p1 = self.block1(phrase1).view(B,C,-1)
        p2 = self.block1(phrase2).view(B,C,-1)
        p3 = self.block1(phrase3).view(B,C,-1)

        # TChannel
        gf = feature1_.view(B, C, -1)
        cls = self.transformer1(gf)[:, 0, ]
        cls1 = self.transformer2(p1,cls)[:,0,]
        cls2 = self.transformer2(p2,cls1)[:,0,]
        cls3 = self.transformer2(p3,cls2)[:,0,]

        cls_total = torch.cat([cls1,cls2,cls3,feature], dim=1)
        logits = self.classifier(cls_total)
        logits = self.softmax(logits)
        return logits,logits,feature,cls1,cls2,cls3


class DuTAP(nn.Module):
    def __init__(self, n_class, channels):
        super().__init__()
        self.prompt1 = nn.Parameter(torch.randn(1, 3,1,748))
        self.prompt2 = nn.Parameter(torch.randn(1, 3,1,1496))
        self.prompt3 = nn.Parameter(torch.randn(1, 3,1,1428))
        self.transformer1 = ViT(num_patch=512, patch_dim=106, dim=128, depth=3, heads=4, mlp_dim=256, pool='cls', dim_head=128, dropout=0., emb_dropout=0.)
        self.transformer2 = TFormer(num_patch=512, patch_dim=108, dim=128, depth=3, heads=4, mlp_dim=256, pool='cls', dim_head=128, dropout=0., emb_dropout=0.)
        self.adpavg = nn.AdaptiveAvgPool2d((4, 4))
        self.cat_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.elu = nn.ELU(inplace=True)
        self.classifier = nn.Sequential(
            nn.Linear(256+128*3, n_class),
        )
        self.softmax = nn.Softmax(dim=1)
        self.vgg1 = vgg16_bn()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.block1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=(2, 8), stride=(1, 4)),
                    nn.ELU(inplace=True),
                    nn.Conv2d(64, 128, kernel_size=(2, 6), stride=(1, 3)),
                    nn.ELU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=(2, 4), stride=(1, 2)),
                    nn.ELU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=(2, 2), stride=(1, 1)),
                    nn.ELU(inplace=True),
                    nn.AdaptiveAvgPool2d((12, 9)),
                )

    def forward(self, pharse1, phrase2, phrase3):
        B = phrase2.size(0)
        C = 512
        phrase = torch.cat((pharse1, phrase2, phrase3), dim=2)
        input = phrase.permute(0, 1,3,2)
        input = normalize_timewise(input)
        feature0 = torch.cat(
            [input[:, :, :, :680], input[:, :, :, 680:2040], input[:, :, :, 2040:]], dim=-1)
        feature1_ = self.vgg1(feature0) #B 512,1,108
        feature1 = self.adpavg(feature1_) #B 512 4,4
        feature = self.cat_conv(feature1)
        #task stream
        phrase1 = input[:,:,:,:748]  #add 3s
        phrase2 = input[:,:,:,612:2108]
        phrase3 = input[:,:,:,1972:]
        phrase1 =  torch.cat([phrase1,self.prompt1.expand(input.size(0), -1, -1,-1)],dim=2)
        phrase2 =  torch.cat([phrase2,self.prompt2.expand(input.size(0), -1, -1,-1)],dim=2)
        phrase3 =  torch.cat([phrase3,self.prompt3.expand(input.size(0), -1, -1,-1)],dim=2)
        p1 = self.block1(phrase1).view(B,C,-1)
        p2 = self.block1(phrase2).view(B,C,-1)
        p3 = self.block1(phrase3).view(B,C,-1)
        #TCHANNEL
        gf = feature1_.view(B, C, -1)
        cls = self.transformer1(gf)[:, 0, ]
        cls1 = self.transformer2(p1,cls)[:,0,]
        cls2 = self.transformer2(p2,cls1)[:,0,]
        cls3 = self.transformer2(p3,cls2)[:,0,]

        cls_total = torch.cat([cls1,cls2,cls3,feature], dim=1)
        logits = self.classifier(cls_total)
        logits = self.softmax(logits)
        return logits,logits,feature,cls1,cls2,cls3