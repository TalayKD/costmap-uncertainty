# (done) only close crops
# (done) close crops + distant crops
# (done) + vel input
# + patch distance
# separate rgb and height encoders
# (done) filter the crops
# (done) filter out the bad trajectories in terms of mapping
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from workflow import WorkFlow, TorchFlow
from arguments import get_args
import numpy as np

from dataloader.TartanDriveDataset import TartanCostDataset
from network.CostNet import CostResNet

import time
import cv2
from math import sqrt

# from scipy.io import savemat
np.set_printoptions(precision=4, threshold=10000)

import time # for testing

def denormalize_vis_rgbmap(rgbmap):
    # rgbmap: c x H x W
    # denormalize the rgbmap
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,1,3)
    rgbmap_np = rgbmap.transpose(1,2,0)
    rgbmap_np = rgbmap_np * std + mean
    rgbmap_np = np.clip(rgbmap_np * 255, 0 ,255)
    rgbmap_np = rgbmap_np.astype(np.uint8)
    return rgbmap_np

class TrainCostmap(TorchFlow.TorchFlow):
    def __init__(self, workingDir, args, prefix = "", suffix = "", plotterType = 'Visdom'):
        super(TrainCostmap, self).__init__(workingDir, prefix, suffix, disableStreamLogger = False, plotterType = plotterType)
        self.args = args    
        self.saveModelName = 'costnet'
        self.costnet = CostResNet(inputnum=8, outputnum=1,velinputlen=32)

        # import ipdb;ipdb.set_trace()
        if self.args.load_model:
            modelname = self.args.working_dir + '/models/' + self.args.model_name
            self.load_model(self.costnet, modelname)

        if self.args.multi_gpu>1:
            self.costnet = nn.DataParallel(self.costnet)

        self.costnet.cuda()

        self.LrDecrease = [int(self.args.train_step/2), int(self.args.train_step*3/4), int(self.args.train_step*7/8)]
        self.lr = self.args.lr

        map_metadata = {
            'height': 12.0,
            'width': 12.0,
            'resolution': 0.02,
            'origin': [-2.0, -6.0],
        }

        crop_params ={
            'crop_size': [4, 4],
            'output_size': [224, 224]
        }


        stride = args.stride
        skip = args.skip
        cropnum = args.crop_num
        framelistfile = 'data/' + args.data_file
        testframelistfile = 'data/' + args.val_file
        datarootdir = args.data_root
        datatypes = "heightmap,rgbmap,odom,patches,cost,vels" # "heightmap"
        modalitylens = [1,1,cropnum,cropnum,cropnum,cropnum] #[1]
        if not (args.test or args.test_traj):
            trainDataset = TartanCostDataset(framelistfile, \
                                map_metadata=map_metadata,
                                crop_params=crop_params,
                                dataroot= datarootdir, \
                                datatypes = datatypes, \
                                modalitylens = modalitylens, \
                                transform=None, \
                                imu_freq = 10, \
                                frame_skip = skip, frame_stride=stride, new_odom_flag=False)
            self.trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker_num)
            self.trainDataiter = iter(self.trainDataloader)

        # while True:
        #     starttime = time.time()
        #     try:
        #         sample = next(self.trainDataiter)
        #     except StopIteration:
        #         print('New epoch..')
        #         self.trainDataiter = iter(self.trainDataloader)
        #         sample = next(self.trainDataiter)
        #     print('Sample load time: {}'.format(time.time() - starttime))

        if args.test_traj: # evaluate on the whole map
            testDataset = TartanCostDataset(testframelistfile, \
                                map_metadata=map_metadata,
                                crop_params=crop_params,
                                dataroot= datarootdir, \
                                datatypes = datatypes, \
                                modalitylens = modalitylens, \
                                transform=None, \
                                imu_freq = 10, \
                                frame_skip = skip, frame_stride=stride, 
                                new_odom_flag=False, coverage=True)
            test_shuffle = False
        else:
            testDataset = TartanCostDataset(testframelistfile, \
                                map_metadata=map_metadata,
                                crop_params=crop_params,
                                dataroot= datarootdir, \
                                datatypes = datatypes, \
                                modalitylens = modalitylens, \
                                transform=None, \
                                imu_freq = 10, \
                                frame_skip = skip, frame_stride=stride, new_odom_flag=False)
            test_shuffle = True

        self.testDataloader = DataLoader(testDataset, batch_size=args.test_batch_size, shuffle=test_shuffle, num_workers=args.worker_num)
        self.testDataiter = iter(self.testDataloader)
        
        self.criterion = nn.L1Loss()
        self.costOptimizer = optim.Adam(self.costnet.parameters(), lr = self.lr)


    def initialize(self):
        super(TrainCostmap, self).initialize()

        self.AV['loss'].avgWidth = 100
        self.add_accumulated_value('test_loss', 1)
        self.append_plotter("loss", ['loss', 'test_loss'], [True, False])

        logstr = ''
        for param in self.args.__dict__.keys(): # record useful params in logfile 
            logstr += param + ': '+ str(self.args.__dict__[param]) + ', '
        self.logger.info(logstr) 

        self.count = 0
        self.test_count = 0
        self.epoch = 0
        self.test_batch = 0

        super(TrainCostmap, self).post_initialize()

    def dumpfiles(self):
        self.save_model(self.costnet, self.saveModelName+'_'+str(self.count))
        self.write_accumulated_values()
        self.draw_accumulated_values()

    def filter_valid_num(self, input_tensor, vels_tensor, target_tensor, thresh=0.5):
        '''
        input_tensor: B x 8 x H x W
        vels_tensor: B x 2
        target_tensor: B
        '''
        threshnum = thresh * input_tensor.shape[-1] * input_tensor.shape[-2] # 224 * 224 / 2 
        heightmask = input_tensor[:,-1,:,:] # B x H x W
        validnum = torch.sum(heightmask, dim=(2,1))
        mask = validnum > threshnum
        return input_tensor[mask,...], vels_tensor[mask,...], target_tensor[mask]

    def forward(self, sample): 
        # import ipdb;ipdb.set_trace()

        patches = sample['patches'] # N x M x 8 x H x W
        targetcost = sample['cost'] # N x M 
        vels = sample['vels'] # N x M x 2
        # patches = torch.rand((vels.shape[0], vels.shape[1], 8, 224, 224))

        input_tensor = patches.view((-1,) + patches.shape[2:] ).cuda()
        vels_tensor = vels.view(-1,2).cuda()
        target_tensor = targetcost.view(-1).cuda()

        # # filter the patch based on valid map pixel number
        # input_tensor, vels_tensor, target_tensor = self.filter_valid_num(input_tensor, vels_tensor, target_tensor)
        # if len(input_tensor) == 0:
        #     return None, None

        output = self.costnet(input_tensor,vels_tensor)

        loss = self.criterion(output, target_tensor)

        return loss, output


    def train(self):
        super(TrainCostmap, self).train()

        self.count = self.count + 1
        self.costnet.train()

        starttime = time.time()

        try:
            sample = next(self.trainDataiter)
        except StopIteration:
            print('New epoch..')
            self.trainDataiter = iter(self.trainDataloader)
            sample = next(self.trainDataiter)

        loadtime = time.time()

        # import ipdb;ipdb.set_trace()
        loss, _ = self.forward(sample)
        self.costOptimizer.zero_grad()
        loss.backward()
        self.costOptimizer.step()
        lossnum = loss.item()
        # lossnum = 0.0 # for debug

        nntime = time.time()

        # import ipdb;ipdb.set_trace()
        self.AV['loss'].push_back(lossnum, self.count)

        # update Learning Rate
        if self.args.lr_decay:
            if self.count in self.LrDecrease:
                self.lr = self.lr*0.4
                for param_group in self.costOptimizer.param_groups: # ed_optimizer is defined in derived class
                    param_group['lr'] = self.lr

        if self.count % self.args.print_interval == 0:
            losslogstr = self.get_log_str()
            self.logger.info("%s #%d - %s lr: %.6f - time load/bp (%.2f, %.2f)"  % (self.args.exp_prefix[:-1], 
                self.count, losslogstr, self.lr, loadtime-starttime, nntime-loadtime))

        if self.count % self.args.plot_interval == 0: 
            self.plot_accumulated_values()

        if self.count % self.args.test_interval == 0:
            if not (self.count)%self.args.snapshot==0:
                self.test()

        if (self.count)%self.args.snapshot==0:
            self.dumpfiles()
        sample = None

    def test(self):
        super(TrainCostmap, self).test()
        self.test_count += 1

        try:
            sample = next(self.testDataiter)
        except StopIteration:
            self.testDataiter = iter(self.testDataloader)
            sample = next(self.testDataiter)

        self.costnet.eval()
        with torch.no_grad():
            loss, output = self.forward(sample)

        lossnum = loss.item()
        # lossnum = 0
        self.AV['test_loss'].push_back(lossnum, self.count)

        if self.test_count % self.args.print_interval == 0:
            self.logger.info("  TEST %s #%d - loss: %.4f "  % (self.args.exp_prefix[:-1], 
                self.test_count, lossnum))
        sample = None
        return lossnum, output

    def test_traj(self,maxpatch_size = 300):
        super(TrainCostmap, self).test()
        self.test_count += 1
        # import ipdb;ipdb.set_trace()
        self.costnet.eval()

        try:
            sample = self.testDataiter.next()
        except StopIteration:
            self.testDataiter = iter(self.testDataloader)
            sample = self.testDataiter.next()

        patches = sample['patches']
        batchsize, patchsize, _, _, _ = patches.shape
        subsample = {}
        outputlist = []
        for k in range(0, patchsize, maxpatch_size):
            endind = min(maxpatch_size+k, patchsize)
            subsample['patches'] = patches[:,k:endind]
            subpatchsize = endind - k
            subsample['cost'] = torch.zeros((batchsize, subpatchsize), dtype=torch.float32) # N x M 
            subsample['vels'] = torch.zeros((batchsize, subpatchsize, 2), dtype=torch.float32) # N x M x 2
            subsample['vels'][:,:,0] = args.test_vel / 5.0 

            with torch.no_grad():
                _, output = self.forward(subsample)
            outputlist.extend(output.cpu().numpy())
        vismap = sample['rgbmap'][0][0].numpy()
        # vid.write(disp_color)
        sample = None
        return np.array(outputlist), vismap

    def finalize(self):
        super(TrainCostmap, self).finalize()
        if self.count < self.args.train_step and not self.args.test:
            self.dumpfiles()

        elif not self.args.test:
            self.logger.info('The average loss values:')
            self.logger.info('%.4f \t %.4f ' % (self.AV['loss'].last_avg(100), self.AV['test_loss'].last_avg(100)))



if __name__ == '__main__':
    args = get_args()
    try:
        # Instantiate an object for MyWF.
        trainCostmap = TrainCostmap(args.working_dir, args, prefix = args.exp_prefix, plotterType = 'Int')
        trainCostmap.initialize()

        if args.test_traj:
            outvidfile = 'test_20220531_lowvel_0.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            fout=cv2.VideoWriter(outvidfile, fourcc, 10, (1800, 600))
            while True:
                output, vismap = trainCostmap.test_traj()
                # import ipdb;ipdb.set_trace()
                # save the file for debug
                patchsize = len(output)                
                mapsize = int(sqrt(patchsize))
                outputnp = output.reshape(mapsize,mapsize)
                disp = (outputnp*255).astype(np.uint8)
                disp = cv2.resize(disp, (400, 400), interpolation=cv2.INTER_NEAREST)
                disp_pad = np.zeros((600,600),dtype=np.uint8)
                disp_pad[100:500,100:500] = disp
                disp_color = cv2.applyColorMap(disp_pad, cv2.COLORMAP_JET)
                # disp_color = cv2.flip(disp_color, -1)
                mapvis = denormalize_vis_rgbmap(vismap)
                disp_overlay = (mapvis //2 + disp_color//2)
                visimg = np.concatenate((disp_color, mapvis, disp_overlay), axis=1)
                cv2.imshow('img', visimg)
                cv2.waitKey(10)
                fout.write(visimg)

                if trainCostmap.test_count >= args.test_num:
                    break
            print("Test reaches the maximum test number (%d)." % (args.test_num))
            fout.release()

        elif args.test:
            errorlist = []
            while True:
                loss, output = trainCostmap.test()
                errorlist.append(loss)

                if trainCostmap.test_batch >= args.test_num:
                    break

            print("Test reaches the maximum test number (%d)." % (args.test_num))
            errorlist = np.array(errorlist)
            trainCostmap.logger.info("Loss statistics: mean %.4f " % (errorlist.mean()))

        else: # Training
            while True:
                trainCostmap.train()
                if (trainCostmap.count >= args.train_step):
                    break

        trainCostmap.finalize()

    except WorkFlow.SigIntException as sie:
        print( sie.describe() )
        print( "Quit after finalize." )
        trainCostmap.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")


