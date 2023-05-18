# (done) only close crops
# (done) close crops + distant crops
# (done) + vel input
# + patch distance
# (done) separate rgb and height encoders
# (done) filter the crops
# (done) filter out the bad trajectories in terms of mapping
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from workflow import WorkFlow, TorchFlow
from arguments import get_args
import numpy as np

from dataloader.TartanDriveDataset import TartanCostDataset

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

def denormalize_vis_heightmap(heightmap):
    # heightmap: c x H x W
    # denormalize the heightmap
    heightmap_np = heightmap.transpose(1,2,0)
    heightmap_np = np.clip((heightmap_np[:,:,2]+1) * 25, 0 ,255)
    heightmap_np = heightmap_np.astype(np.uint8)
    heightmap_vis = cv2.applyColorMap(heightmap_np, cv2.COLORMAP_JET)
    return heightmap_vis

def get_coverage_path(map_metadata, stride, crop_size):
    '''
    '''
    map_height = map_metadata['height'] # 12
    map_width = map_metadata['width'] # 12
    resolution = map_metadata['resolution'] # 0.02

    cropx_px = int(crop_size[0]/resolution)
    cropy_px = int(crop_size[1]/resolution)

    map_height_px = int(map_height/resolution) # 600
    map_width_px = int(map_width/resolution) # 600
    stride_px = int(stride/resolution) # 0.2 -> 10

    pathlist = []
    for x in np.arange(0, map_height_px-cropx_px+stride_px, stride_px):
        for y in np.arange(0, map_width_px-cropy_px+stride_px, stride_px):
            pathlist.append([x, y, x+cropx_px, y+cropy_px])
    # import ipdb;ipdb.set_trace()
    return np.array(pathlist)

class TrainCostmap(TorchFlow.TorchFlow):
    def __init__(self, workingDir, args, prefix = "", suffix = "", plotterType = 'Visdom'):
        super(TrainCostmap, self).__init__(workingDir, prefix, suffix, disableStreamLogger = False, plotterType = plotterType)
        self.args = args    
        self.saveModelName = 'costnet'
        self.network = args.network
        if args.network == 1:
            from network.CostNet import CostResNet
            self.costnet = CostResNet(inputnum=8, outputnum=1,velinputlen=32)
        elif args.network ==2:
            from network.CostNet import TwoHeadCostResNet
            self.costnet = TwoHeadCostResNet(inputnum1=3, inputnum2=5, outputnum=1,velinputlen=32, config=args.net_config, finetune=args.finetune)

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
                                frame_skip = skip, frame_stride=stride, \
                                new_odom_flag=False, data_augment=True)
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
            self.fpvimg = 'img0' # for warthog it's 'img0', yamaha 'imgc
            testDataset = TartanCostDataset(testframelistfile, \
                                map_metadata=map_metadata,
                                crop_params=crop_params,
                                dataroot= datarootdir, \
                                datatypes = "heightmap,rgbmap,"+self.fpvimg, \
                                modalitylens = [1,1,1], \
                                transform=None, \
                                imu_freq = 10, \
                                frame_skip = 0, frame_stride=3, 
                                new_odom_flag=False, data_augment=False)
            test_shuffle = False
            cropstide = 0.4 # m
            self.test_vel_list = [0.5,1,2,4] #[2,4,6,8] #[0.5,1,2,4]
            self.pathlist = get_coverage_path(map_metadata, cropstide, crop_params['crop_size'])
            self.patchnum = len(self.pathlist)
            self.scale_factor = [crop_params['output_size'][0]/(self.pathlist[0][2]-self.pathlist[0][0]), 
                                crop_params['output_size'][1]/(self.pathlist[0][3]-self.pathlist[0][1])]
            print("Crop {} patches for each frame".format(self.patchnum))
        else:
            testDataset = TartanCostDataset(testframelistfile, \
                                map_metadata=map_metadata,
                                crop_params=crop_params,
                                dataroot= datarootdir, \
                                datatypes = datatypes, \
                                modalitylens = modalitylens, \
                                transform=None, \
                                imu_freq = 10, \
                                frame_skip = skip, frame_stride=stride, \
                                new_odom_flag=False, data_augment=True)
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

    def vis_patches(self, patches):
        patches_numpy_list = []
        plotstride = 2
        for ind in range(0, len(patches), plotstride):
            ppp = denormalize_vis_rgbmap(patches[ind][:3].numpy())
            patches_numpy_list.append(ppp)
        patchvis = np.concatenate(patches_numpy_list, axis=1)
        # cv2.imshow('map',patchvis)
        # cv2.waitKey(0)
        cv2.imwrite('map'+str(self.count) + '.png', patchvis)

    def forward(self, sample, mask=False): 
        # import ipdb;ipdb.set_trace()

        patches = sample['patches'] # N x M x 8 x H x W
        targetcost = sample['cost'] # N x M or N x M x K
        vels = sample['vels'] # N x M x 2 or N x M x K x 2
        # patches = torch.rand((vels.shape[0], vels.shape[1], 8, 224, 224))
        batchsize = patches.shape[0]
        patchsize = patches.shape[1]
        batchcombine = batchsize*patchsize
        
        # self.vis_patches(patches[0])

        input_tensor = patches.view((batchcombine,) + patches.shape[2:] ).cuda()
        vels_tensor = vels.view((batchcombine,) + vels.shape[2:]).cuda()
        target_tensor = targetcost.view((batchcombine,) + targetcost.shape[2:]).cuda()

        # filter the patch based on valid map pixel number
        if mask:
            input_tensor, vels_tensor, target_tensor = self.filter_valid_num(input_tensor, vels_tensor, target_tensor)
            self.logger.info("Filter: {} -> {}".format(batchcombine, input_tensor.shape[0]))
            if len(input_tensor) == 0:
                return None, None

        output = self.costnet(input_tensor,vels_tensor)
        loss = self.criterion(output, target_tensor)
        if not mask: # hacking, assume the mask=False in testing! 
            output = output.view((batchsize, patchsize) + output.shape[1:] )

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
        loss, _ = self.forward(sample, mask=False)

        if loss is None:
            self.count -= 1
            sample = None
            return 

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

    def test_traj(self,maxpatch_size = 500):
        super(TrainCostmap, self).test()
        self.test_count += 1
        self.costnet.eval()

        try:
            sample = next(self.testDataiter)
        except StopIteration:
            return None, None, None, None 
            # self.testDataiter = iter(self.testDataloader)
            # sample = self.testDataiter.next()

        rgbmap = sample['rgbmap']
        heightmap = sample['heightmap']
        batchsize = rgbmap.shape[0]
        assert batchsize == 1, 'when test_traj, test-batch-size should be 1'
        maps = torch.cat((rgbmap[0][0], heightmap[0][0]), dim=0) # 8 x H x W

        outputlist = []
        vels = torch.tensor(self.test_vel_list, dtype=torch.float32).view(1,1,4)
        velnum = vels.shape[-1]
        for k in range(0, self.patchnum, maxpatch_size):
            endind = min(maxpatch_size+k, self.patchnum)
            subsample = {} 
            patches = []
            for w in range(k, endind): 
                patchloc = self.pathlist[w]
                patch = maps[:, patchloc[0]:patchloc[2], patchloc[1]:patchloc[3]]
                patches.append(patch) # F.interpolate(output[0], scale_factor=4, mode='bilinear')
            # import ipdb;ipdb.set_trace()
            patches = torch.stack(patches, dim=0)
            patches = F.interpolate(patches, scale_factor=self.scale_factor)
            subsample['patches'] = patches.unsqueeze(0)
            subpatchsize = endind - k
            subsample['cost'] = torch.zeros((1, subpatchsize, velnum), dtype=torch.float32) # N x M x K
            subsample['vels'] = torch.zeros((1, subpatchsize, velnum, 2), dtype=torch.float32) # N x M x K x 2
            subsample['vels'][:,:,:,0] = vels / 5.0 

            with torch.no_grad():
                _, output = self.forward(subsample) # output N x M x K
            outputlist.append(output.cpu().numpy().reshape(batchsize, subpatchsize, velnum))
        costlist = np.concatenate(outputlist, axis=1) # N x M x K
            
        print("Test {}".format(self.test_count))
        visrgbmap = rgbmap[0][0].numpy()
        visheightmap = heightmap[0][0].numpy()
        visfpv = sample[self.fpvimg][0][0].numpy()
        # vid.write(disp_color)
        sample = None
        return costlist[0], visrgbmap, visheightmap, visfpv

    def finalize(self):
        super(TrainCostmap, self).finalize()
        if self.count < self.args.train_step and not self.args.test:
            self.dumpfiles()

        elif not self.args.test:
            self.logger.info('The average loss values:')
            self.logger.info('%.4f \t %.4f ' % (self.AV['loss'].last_avg(100), self.AV['test_loss'].last_avg(100)))

def visualize_output(outputlist, visrgbmap, visheightmap, visfpv, test_vel_list): 
    '''
    outputlist: ( H x W ) x 4 cost estimation
    visrgbmap: H x W x 3 
    visheightmap: H x W 
    visfpv: H x W x 3
    '''
    visfpv = cv2.resize(visfpv, (800, 400), interpolation=cv2.INTER_LINEAR)

    mapvis = denormalize_vis_rgbmap(visrgbmap)
    mapvis_400 = cv2.resize(mapvis, (400, 400), interpolation=cv2.INTER_LINEAR)
    mapvis_200 = cv2.resize(mapvis, (200, 200), interpolation=cv2.INTER_LINEAR)
    mapvis_200 = cv2.flip(mapvis_200, -1)

    maphvis = denormalize_vis_heightmap(visheightmap)
    maphvis_400 = cv2.resize(maphvis, (400, 400), interpolation=cv2.INTER_LINEAR)

    mapsvis = np.concatenate((maphvis_400,mapvis_400), axis=0) # 800 x 400
    mapsvis = cv2.flip(mapsvis, -1)

    costvislist, costoverlaylist = [], []
    vellist = ['Vel='+str(k)+' m/s' for k in test_vel_list]
    # import ipdb;ipdb.set_trace()
    for k in range(4): # hard code
        output = outputlist[:,k]
    # save the file for debug
        patchsize = len(output)                
        mapsize = int(sqrt(patchsize))
        outputnp = output.reshape(mapsize, mapsize)
        disp = np.clip((outputnp*255),0,255).astype(np.uint8)
        disp = cv2.resize(disp, (133, 133), interpolation=cv2.INTER_LINEAR) #
        disp_pad = np.zeros((200,200),dtype=np.uint8)
        disp_pad[33:166,33:166] = disp
        disp_color = cv2.applyColorMap(disp_pad, cv2.COLORMAP_JET)
        disp_color = cv2.flip(disp_color, -1)
        disp_color = cv2.putText(disp_color, vellist[k], (45,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,210,245),thickness=1)
        disp_overlay = (mapvis_200 * 0.6 + disp_color * 0.4).astype(np.uint8)
        costvislist.append(disp_color)
        costoverlaylist.append(disp_overlay)
    costvis = np.concatenate(
                (costvislist[0],costvislist[1],costvislist[2],costvislist[3]),axis=1) # 
    costoverlay = np.concatenate(
                (costoverlaylist[0],costoverlaylist[1],costoverlaylist[2],costoverlaylist[3]),axis=1)
    costvisimg = np.concatenate((costvis, costoverlay), axis=0) # 400 x 800

    visimg = np.concatenate((mapsvis, 
                            np.concatenate((visfpv, costvisimg),axis=0)), axis=1)
    return visimg

if __name__ == '__main__':
    args = get_args()
    try:
        # Instantiate an object for MyWF.
        trainCostmap = TrainCostmap(args.working_dir, args, prefix = args.exp_prefix, plotterType = 'Int')
        trainCostmap.initialize()

        if args.test_traj:
            outvidfile = args.model_name.split('.pkl')[0] +'_'+ args.out_vid_file #'test_20220531_lowvel_0.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            fout=cv2.VideoWriter(outvidfile, fourcc, 10, (1200, 800))
            while True:
                outputlist, visrgbmap, visheightmap, visfpv = trainCostmap.test_traj()
                if outputlist is None:
                    break
                visimg = visualize_output(outputlist, visrgbmap, visheightmap, visfpv, trainCostmap.test_vel_list)
                # cv2.imshow('img', visimg)
                # cv2.waitKey(10)
                fout.write(visimg)

                # if trainCostmap.test_count >= args.test_num:
                #     break
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


