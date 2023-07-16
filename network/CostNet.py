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
        x = x.squeeze(-1)
        return x


class TwoHeadCostResNet(nn.Module):
    def __init__(self, inputnum1, inputnum2, outputnum, velinputlen=0, config=0, finetune=False):
        super().__init__()
        # blocknums = [2,2,3,3,3]
        # outputnums = [16,32,64,64,128,128,256]
        if config == 0:
            blocknums = [1,1,1,1,1]
            outputnums = [16,16,32,32,64,64,128]
            linearnum = [64, 16]
        else:
            blocknums = [1,1,1,1,1]
            outputnums = [8,16,16,16,32,32,64]
            linearnum = [32, 8]

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

        fc1 = Linear(outputnums[6] + velinputlen, linearnum[0])
        fc2 = Linear( linearnum[0], linearnum[1])
        fc3 = nn.Linear( linearnum[1], outputnum)

        self.cost_out = nn.Sequential(fc1, fc2, fc3)

        if finetune:
            fixlist = [self.firstconv_rgb, self.firstconv_hei, 
                        self.layer0_rgb, self.layer1_rgb, self.layer2_rgb, 
                        self.layer0_hei, self.layer1_hei, self.layer2_hei,
                        self.layer3, self.layer4] # , self.lastconv, fc1
            for layer in fixlist:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x, vel=None):
        '''
        x1: N x c x h x w
        vel: N x 2 or N x k x 2
        if vel is N x k x 2, it outputs multiple costs for multiple speeds

        return N     if vel is N x 2
               N x k if vel is N x k x 2
        '''
        x1 = x[:,:self.inputnum1,:,:]           # B x 3 x 224 x 224
        x2 = x[:,self.inputnum1:,:,:]           # B x 5 x 224 x 224

        x1 = self.firstconv_rgb(x1)             # B x 16 x 112 x 112    or  B x outputnums[0] x 112 x 112
        x1 = self.layer0_rgb(x1)                # B x 16 x 56 x 56      or  B x outputnums[1] x 56 x 56
        x1 = self.layer1_rgb(x1)                # B x 32 x 28 x 28      or  B x outputnums[2] x 28 x 28
        x1 = self.layer2_rgb(x1)                # B x 32 x 14 x 14      or  B x outputnums[3] x 14 x 14

        x2 = self.firstconv_hei(x2)             # B x 16 x 112 x 112    or  B x outputnums[0] x 112 x 112 
        x2 = self.layer0_hei(x2)                # B x 16 x 56 x 56      or  B x outputnums[1] x 56 x 56
        x2 = self.layer1_hei(x2)                # B x 32 x 28 x 28      or  B x outputnums[2] x 28 x 28
        x2 = self.layer2_hei(x2)                # B x 32 x 14 x 14      or  B x outputnums[3] x 14 x 14

        x = torch.cat((x1,x2), dim = 1)         # B x 64 x 14 x 14      or  B x outputnums[3]*2 x 14 x 14

        x = self.layer3(x)                      # B x 64 x 7 x 7        or  B x outputnums[4] x 7 x 7
        x = self.layer4(x)                      # B x 64 x 4 x 4        or  B x outputnums[5] x 4 x 4
        x = self.lastconv(x)                    # B x 128 x 1 x 1       or  B x outputnums[6] x 1 x 1
        x = x.view(x.shape[0], -1)              # B x 128               or  B x outputnums[6]      

        # import ipdb;ipdb.set_trace()
        if vel is not None:                                 # vel is either N x 2 or N x k x 2
            if len(vel.shape) == 2: # N x 2
                vel = vel.repeat(1,self.velinputlen//2)                         # N x 32
                x = torch.cat((x,vel),dim=-1)                                   # N x 160   or  N x (outputnums[6]+velinputlen)
            elif len(vel.shape) == 3:
                velnum = vel.shape[1]
                vel = vel.repeat(1, 1, self.velinputlen//2)                     # N x k x 32
                x = x.unsqueeze(1).repeat(1, velnum, 1)                         # N x k x 128   or  N x k x outputnums[6]
                x = torch.cat((x,vel), dim=-1)                                  # N x k x 160   or  N x k x (outputnums[6]+velinputlen)
            else:
                print("unsupport vel dimention {}".format(vel.shape))
        x = self.cost_out(x) # N x 1 if vel is N x 2    and     N x k x 1   if vel is N x k x 2
        x = x.squeeze(-1)    # N if vel is N x 2        and     N x k       if vel is N x k x 2
        return x
    


class AleatoricTwoHeadCostResNet(nn.Module):
    def __init__(self, inputnum1, inputnum2, outputnum, velinputlen=0, config=0, finetune=False):
        super().__init__()
        # blocknums = [2,2,3,3,3]
        # outputnums = [16,32,64,64,128,128,256]
        if config == 0:
            self.blocknums = [1,1,1,1,1]
            self.outputnums = [16,16,32,32,64,64,128]
            self.linearnum = [64, 32]
            self.munum = [16, 8]
            self.sigmanum = [16,8]

        else:
            self.blocknums = [1,1,1,1,1]
            self.outputnums = [8,16,16,16,32,32,64]
            self.linearnum = [32, 16]
            self.munum = [8, 4]
            self.sigmanum = [8, 4]

        self.velinputlen = velinputlen
        self.inputnum1 = inputnum1

        self.firstconv_rgb = Conv(inputnum1, self.outputnums[0], 3, 2) # 112 x 112
        self.firstconv_hei = Conv(inputnum2, self.outputnums[0], 3, 2) # 112 x 112

        self.layer0_rgb = make_layer(BasicBlock, self.outputnums[0], self.outputnums[1], self.blocknums[0], 2, 1, 1) # 56 x 56 (16)
        self.layer1_rgb = make_layer(BasicBlock, self.outputnums[1], self.outputnums[2], self.blocknums[1], 2, 1, 1) # 28 x 28 (32)
        self.layer2_rgb = make_layer(BasicBlock, self.outputnums[2], self.outputnums[3], self.blocknums[2], 2, 1, 1) # 14 x 14 (32)

        self.layer0_hei = make_layer(BasicBlock, self.outputnums[0], self.outputnums[1], self.blocknums[0], 2, 1, 1) # 56 x 56 (16)
        self.layer1_hei = make_layer(BasicBlock, self.outputnums[1], self.outputnums[2], self.blocknums[1], 2, 1, 1) # 28 x 28 (32)
        self.layer2_hei = make_layer(BasicBlock, self.outputnums[2], self.outputnums[3], self.blocknums[2], 2, 1, 1) # 14 x 14 (32)

        self.layer3 = make_layer(BasicBlock, self.outputnums[3]*2, self.outputnums[4], self.blocknums[3], 2, 1, 1) # 7 x 7 (64)
        self.layer4 = make_layer(BasicBlock, self.outputnums[4], self.outputnums[5], self.blocknums[4], 2, 1, 1) # 4 x 4 (64)

        self.lastconv = Conv(self.outputnums[5],self.outputnums[6], kernel_size=4, stride=1, padding=0) # 1 x 1 (128)

        fc1 = Linear(self.outputnums[6] + velinputlen, self.linearnum[0])
        fc2 = Linear(self.linearnum[0], self.linearnum[1])

        layer0_mu = Linear(self.linearnum[1]//2,  self.munum[0])
        layer1_mu = Linear(self.munum[0], self.munum[1])
        layer2_mu = nn.Linear(self.munum[1], outputnum)

        layer0_sigma = Linear(self.linearnum[1]//2, self.sigmanum[0])
        layer1_sigma = Linear(self.sigmanum[0], self.sigmanum[1])
        layer2_sigma = nn.Linear(self.sigmanum[1], outputnum)

        self.fc_out = nn.Sequential(fc1, fc2)

        self.mu_out = nn.Sequential(layer0_mu, layer1_mu, layer2_mu)
        self.sigma_out = nn.Sequential(layer0_sigma, layer1_sigma, layer2_sigma)

        if finetune:
            fixlist = [self.firstconv_rgb, self.firstconv_hei, 
                        self.layer0_rgb, self.layer1_rgb, self.layer2_rgb, 
                        self.layer0_hei, self.layer1_hei, self.layer2_hei,
                        self.layer3, self.layer4] # , self.lastconv, fc1
            for layer in fixlist:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x, vel=None):
        '''
        x1: N x c x h x w
        vel: N x 2 or N x k x 2
        if vel is N x k x 2, it outputs multiple costs for multiple speeds

        return N     if vel is N x 2
               N x k if vel is N x k x 2
        '''
        x1 = x[:,:self.inputnum1,:,:]           # B x 3 x 224 x 224
        x2 = x[:,self.inputnum1:,:,:]           # B x 5 x 224 x 224

        x1 = self.firstconv_rgb(x1)             # B x 16 x 112 x 112    or  B x outputnums[0] x 112 x 112
        x1 = self.layer0_rgb(x1)                # B x 16 x 56 x 56      or  B x outputnums[1] x 56 x 56
        x1 = self.layer1_rgb(x1)                # B x 32 x 28 x 28      or  B x outputnums[2] x 28 x 28
        x1 = self.layer2_rgb(x1)                # B x 32 x 14 x 14      or  B x outputnums[3] x 14 x 14

        x2 = self.firstconv_hei(x2)             # B x 16 x 112 x 112    or  B x outputnums[0] x 112 x 112 
        x2 = self.layer0_hei(x2)                # B x 16 x 56 x 56      or  B x outputnums[1] x 56 x 56
        x2 = self.layer1_hei(x2)                # B x 32 x 28 x 28      or  B x outputnums[2] x 28 x 28
        x2 = self.layer2_hei(x2)                # B x 32 x 14 x 14      or  B x outputnums[3] x 14 x 14

        x = torch.cat((x1,x2), dim = 1)         # B x 64 x 14 x 14      or  B x outputnums[3]*2 x 14 x 14

        x = self.layer3(x)                      # B x 64 x 7 x 7        or  B x outputnums[4] x 7 x 7
        x = self.layer4(x)                      # B x 64 x 4 x 4        or  B x outputnums[5] x 4 x 4
        x = self.lastconv(x)                    # B x 128 x 1 x 1       or  B x outputnums[6] x 1 x 1
        x = x.view(x.shape[0], -1)              # B x 128               or  B x outputnums[6]      

        # import ipdb;ipdb.set_trace()
        if vel is not None:                                 # vel is either N x 2 or N x k x 2
            if len(vel.shape) == 2: # N x 2
                vel = vel.repeat(1,self.velinputlen//2)                         # N x 32
                x = torch.cat((x,vel),dim=-1)                                   # N x 160   or  N x (outputnums[6]+velinputlen)
            elif len(vel.shape) == 3:
                velnum = vel.shape[1]
                vel = vel.repeat(1, 1, self.velinputlen//2)                     # N x k x 32
                x = x.unsqueeze(1).repeat(1, velnum, 1)                         # N x k x 128   or  N x k x outputnums[6]
                x = torch.cat((x,vel), dim=-1)                                  # N x k x 160   or  N x k x (outputnums[6]+velinputlen)
            else:
                print("unsupport vel dimension {}".format(vel.shape))
        x = self.fc_out(x)   # N x linearnums[1] if vel is N x 2    and     N x k x linearnums[1]   if vel is N x k x 2
        if (len(x.shape) == 2):
            x_mu = x[:, :self.linearnum[1]//2]
            x_sigma = x[:, self.linearnum[1]//2:]
        elif (len(x.shape) == 3):
            x_mu = x[:, :, :self.linearnum[1]//2]
            x_sigma = x[:, :, self.linearnum[1]//2:]
        else:
            print("unsupport tensor dimension {}".format(x.shape))

        sig = nn.Sigmoid()
        mu = self.mu_out(x_mu)              # N x 1 if vel is N x 2         and         N x k x 1 if vel is N x k x 2
        mu = sig(mu)
        sigma = self.sigma_out(x_sigma)     # N x 1 if vel is N x 2         and         N x k x 1 if vel is N x k x 2
        sigma = torch.exp(sigma)

        mu = mu.squeeze(-1)             # N if vel is N x 2     and     N x k       if vel is N x k x 2
        sigma = sigma.squeeze(-1)       # N if vel is N x 2     and     N x k       if vel is N x k x 2
        return mu, sigma



if __name__ == "__main__":
    import time

    import ipdb;ipdb.set_trace()
    # model = CostResNet(8, 1)
    # model = TwoHeadCostResNet(3, 5, 1, velinputlen=32, config=0, finetune=False)
    model = AleatoricTwoHeadCostResNet(3, 5, 1, velinputlen=32, config=0, finetune=False)
    # print(model)

    batchlist = [1,10,100,400]

    for batch in batchlist:
        # batch = 10  
        channel = 8
        velnum = 4
        imgTensor = torch.rand(batch, channel, 224, 224)
        velTensor = torch.rand(batch, 2)                        # for a single velocity, used during training
        # velTensor = torch.rand(batch, velnum, 2)                # for multiple velocities
        targetTensor = torch.rand(batch)                        # for a single velocity, used during training
        # targetTensor = torch.rand(batch, velnum)                # for multiple velocities

        # output = model(imgTensor,velTensor)                         # for normal Costnets
        mu, sigma = model(imgTensor, velTensor)                     # for Aleatoric Costnets
        # output = model({"patches": imgTensor, "fourier_vels": velTensor})
        # import ipdb;ipdb.set_trace()
        # print(output.shape)

        # starttime = time.time()
        # for k in range(100):
        #     with torch.no_grad():
        #         output = model(imgTensor,velTensor)
                # output = model({"patches": imgTensor, "fourier_vels": velTensor})
                
            # print(output.shape)
        # print((time.time()-starttime)/100)