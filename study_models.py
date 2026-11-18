"""The three pre-specified US-only paths; no external pretrained weights."""
import torch
from torch import nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self,channels=1):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(channels,16,3,stride=2,padding=1),nn.ReLU(),
                                   nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),
                                   nn.Conv2d(32,64,3,stride=2,padding=1),nn.ReLU(),
                                   nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Linear(64,3)
    def forward(self,x): return self.head(self.encoder(x))


class StudyModel(nn.Module):
    def __init__(self,method):
        super().__init__(); self.method=method
        self.classifier=Classifier(2 if method=='xray_sr' else 1)
        if method=='xray_sr':
            self.sr=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),
                                  nn.Conv2d(8,8,3,padding=1),nn.ReLU(),
                                  nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
                                  nn.Conv2d(8,1,3,padding=1))
        if method=='enhanced':
            axis=torch.arange(-6,7,dtype=torch.float32)
            g=torch.exp(-0.5*(axis/2)**2); g=g/g.sum()
            self.register_buffer('kernel',(g[:,None]*g[None]).reshape(1,1,13,13))
    def forward(self,x):
        if self.method=='direct': return self.classifier(x),None
        up=F.interpolate(x,scale_factor=2,mode='bicubic',align_corners=False)
        if self.method=='enhanced':
            blur=F.conv2d(F.pad(up,(6,6,6,6),mode='reflect'),self.kernel)
            return self.classifier(up+(up-blur)),None
        logits=self.sr(x)
        return self.classifier(torch.cat((up,torch.sigmoid(logits)),dim=1)),logits
