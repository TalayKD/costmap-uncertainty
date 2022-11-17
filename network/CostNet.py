import torch
import torch.nn as nn
from .modules import BasicBlock, make_layer, Conv, Linear

class CostResNet(nn.Module):
    def __init__(self, inputnum, outputnum, velinputlen=0):
        super().__init__()
        # blocknums = [2,2,3,3,3]
        # outputnums = [16,32,64,64,128,128,256]
        blocknums = [1,1,1,1,1]
        outputnums = [16,32,32,64,64,64,128]
        self.velinputlen = velinputlen

        self.firstconv = Conv(inputnum, outputnums[0], 3, 2) # 112 x 112

        self.layer0 = make_layer(BasicBlock, outputnums[0], outputnums[1], blocknums[0], 2, 1, 1) # 56 x 56 (32)
        self.layer1 = make_layer(BasicBlock, outputnums[1], outputnums[2], blocknums[1], 2, 1, 1) # 28 x 28 (64)
        self.layer2 = make_layer(BasicBlock, outputnums[2], outputnums[3], blocknums[2], 2, 1, 1) # 14 x 14 (64)
        self.layer3 = make_layer(BasicBlock, outputnums[3], outputnums[4], blocknums[3], 2, 1, 1) # 7 x 7 (128)
        self.layer4 = make_layer(BasicBlock, outputnums[4], outputnums[5], blocknums[4], 2, 1, 1) # 4 x 4 (128)

        self.lastconv = Conv(outputnums[5],outputnums[6], kernel_size=4, stride=1, padding=0) # 1 x 1

        fc1 = Linear(outputnums[6] + velinputlen, 64)
        fc2 = Linear(64,16)
        fc3 = nn.Linear(16, outputnum)

        self.cost_out = nn.Sequential(fc1, fc2, fc3)


    def forward(self, x, vel=None):
        '''
        x: N x c x h x w
        vel: N x 2
        '''
        x = self.firstconv(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.lastconv(x)
        x = x.view(x.shape[0], -1)

        # import ipdb;ipdb.set_trace()
        if vel is not None:
            vel = vel.repeat(1,self.velinputlen//2)
            x = torch.cat((x,vel),dim=-1)
        x = self.cost_out(x)
        x = x.squeeze(0)
        return x


class TwoHeadCostResNet(nn.Module):
    def __init__(self, inputnum1, inputnum2, outputnum, velinputlen=0):
        super().__init__()
        # blocknums = [2,2,3,3,3]
        # outputnums = [16,32,64,64,128,128,256]
        blocknums = [1,1,1,1,1]
        outputnums = [16,16,32,32,64,64,128]
        self.velinputlen = velinputlen
        self.inputnum1 = inputnum1

        self.firstconv_rgb = Conv(inputnum1, outputnums[0], 3, 2) # 112 x 112
        self.firstconv_hei = Conv(inputnum2, outputnums[0], 3, 2) # 112 x 112

        self.layer0_rgb = make_layer(BasicBlock, outputnums[0], outputnums[1], blocknums[0], 2, 1, 1) # 56 x 56 (16)
        self.layer1_rgb = make_layer(BasicBlock, outputnums[1], outputnums[2], blocknums[1], 2, 1, 1) # 28 x 28 (32)
        self.layer2_rgb = make_layer(BasicBlock, outputnums[2], outputnums[3], blocknums[2], 2, 1, 1) # 14 x 14 (32)

        self.layer0_hei = make_layer(BasicBlock, outputnums[0], outputnums[1], blocknums[0], 2, 1, 1) # 56 x 56 (16)
        self.layer1_hei = make_layer(BasicBlock, outputnums[1], outputnums[2], blocknums[1], 2, 1, 1) # 28 x 28 (32)
        self.layer2_hei = make_layer(BasicBlock, outputnums[2], outputnums[3], blocknums[2], 2, 1, 1) # 14 x 14 (32)

        self.layer3 = make_layer(BasicBlock, outputnums[3]*2, outputnums[4], blocknums[3], 2, 1, 1) # 7 x 7 (64)
        self.layer4 = make_layer(BasicBlock, outputnums[4], outputnums[5], blocknums[4], 2, 1, 1) # 4 x 4 (64)

        self.lastconv = Conv(outputnums[5],outputnums[6], kernel_size=4, stride=1, padding=0) # 1 x 1 (128)

        fc1 = Linear(outputnums[6] + velinputlen, 64)
        fc2 = Linear(64,16)
        fc3 = nn.Linear(16, outputnum)

        self.cost_out = nn.Sequential(fc1, fc2, fc3)


    def forward(self, x, vel=None):
        '''
        x1: N x c x h x w
        vel: N x 2
        '''
        x1 = x[:,:self.inputnum1,:,:] 
        x2 = x[:,self.inputnum1:,:,:]

        x1 = self.firstconv_rgb(x1)
        x1 = self.layer0_rgb(x1)
        x1 = self.layer1_rgb(x1)
        x1 = self.layer2_rgb(x1)

        x2 = self.firstconv_hei(x2)
        x2 = self.layer0_hei(x2)
        x2 = self.layer1_hei(x2)
        x2 = self.layer2_hei(x2)

        x = torch.cat((x1,x2), dim = 1)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.lastconv(x)
        x = x.view(x.shape[0], -1)

        # import ipdb;ipdb.set_trace()
        if vel is not None:
            vel = vel.repeat(1,self.velinputlen//2)
            x = torch.cat((x,vel),dim=-1)
        x = self.cost_out(x)
        x = x.squeeze(0)
        return x

if __name__ == "__main__":
    import time
    model = CostResNet(8, 1)
    model = TwoHeadCostResNet(3, 5, 1)
    model.cuda()
    # print(model)

    batchlist = [1,10,100,500]

    for batch in batchlist:
        # batch = 10  
        channel = 8
        imgTensor = torch.rand(batch, channel, 224, 224).cuda()
        velTensor = torch.rand(batch, 2).cuda()

        output = model(imgTensor)
        # output = model({"patches": imgTensor, "fourier_vels": velTensor})
        # import ipdb;ipdb.set_trace()
        print(output.shape)

        starttime = time.time()
        for k in range(100):
            with torch.no_grad():
                output = model(imgTensor)
                # output = model({"patches": imgTensor, "fourier_vels": velTensor})
                
            # print(output.shape)
        print((time.time()-starttime)/100)