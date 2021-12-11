import os
from torch import nn

def init_weights(modules):
    for module in modules:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.weight, 0)

def ensuredir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def logging(msg, *args, out=True):
    dir = './logs'
    ensuredir(dir)
    if out:
        print(msg)
    for file in args:
        f = open(dir + '/' + file, 'a')
        f.write(msg + '\n')
        f.flush()
