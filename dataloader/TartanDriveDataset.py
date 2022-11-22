from __future__ import print_function

import numpy as np
import cv2
from scipy.spatial.transform import Rotation
np.set_printoptions(precision=2, suppress=True)
import random

from torch.utils.data import Dataset
import torch

from .terrain_map_tartandrive import TerrainMap, get_local_path

# TODO: probably want to filter the initial zero-speed section and the last intervention section
class TartanCostDataset(Dataset):
    '''
    Loader for multi-modal data
    -----
    framelistfile: 
    TRAJNAME FRAMENUM
    FRAMESTR0
    FRAMESTR1
    ...
    -----
    Requirements: 
    The actural data path consists three parts: DATAROOT+TRAJNAME+CONVERT(FRAMESTR)
    The frames under the same TRAJNAME should be consequtive. So when the next frame is requested, it should return the next one in the same sequence. 
    The frames should exists on the harddrive. 
    Sequential data: 
    When a sequence of data is required, the code will automatically adjust the length of the dataset, to make sure the longest modality queried exists. 
    The IMU has a higher frequency than the other modalities. The frequency is imu_freq x other_freq. 
    '''
    def __init__(self, \
        framelistfile, \
        map_metadata, \
        crop_params, \
        dataroot = "", \
        datatypes = "img0,img1,imgc,disp0,heightmap,rgbmap,cmd,odom,cost,patches,imu", \
        modalitylens = [1,1,1,1,1,1,1,1,1,1,10], \
        transform=None, \
        imu_freq = 10, \
        frame_skip = 0, \
        frame_stride = 1, \
        new_odom_flag = False, \
        coverage = False):

        super(TartanCostDataset, self).__init__()
        self.framelistfile = framelistfile
        self.map_metadata = map_metadata
        self.crop_params = crop_params
        self.dataroot = dataroot
        self.transform = transform
        self.imu_freq = imu_freq
        self.frame_skip = frame_skip # sample not consequtively, skip a few frames within a sequences
        self.frame_stride = frame_stride # sample less sequence, skip a few frames between two sequences 
        self.new_odom_flag = new_odom_flag # somehow, the odom axis of tartandrive is different from the odom axis in recent bagfiles
        self.coverage = coverage

        self.datatypelist = datatypes.split(',')
        self.modalitylenlist = modalitylens
        assert len(self.datatypelist)==len(modalitylens), "Error: datatype len {}, modalitylens len {}".format(len(self.datatypelist),len(modalitylens))
        self.trajlist, self.trajlenlist, self.framelist, self.imulenlist, self.startframelist = self.parse_inputfile(framelistfile, frame_skip)
        self.sample_seq_len = self.calc_seq_len(self.datatypelist, modalitylens, imu_freq)
        self.seqnumlist = self.parse_length(self.trajlenlist, frame_skip, frame_stride, self.sample_seq_len)

        self.framenumFromFile = len(self.framelist)
        self.N = sum(self.seqnumlist)
        self.trajnum = len(self.trajlenlist)
        self.acc_trajlen = [0,] + np.cumsum(self.trajlenlist).tolist()
        self.acc_seqlen = [0,] + np.cumsum(self.seqnumlist).tolist() # [0, num[0], num[0]+num[1], ..]
        self.acc_imulen = [0,] + np.cumsum(self.imulenlist).tolist() # [0, num[0], num[0]+num[1], ..]
        print('Loaded {} sequences from {}...'.format(self.N, framelistfile))

        if 'cmd' in self.datatypelist:
            self.cmdlist = self.loadDataFromFile(self.trajlist, self.startframelist, self.trajlenlist, 'cmd/twist.npy')
        if 'odom' in self.datatypelist:
            self.odomlist = self.loadDataFromFile(self.trajlist, self.startframelist, self.trajlenlist, 'odom/odometry.npy')
            self.odomlist_tartanvo = self.loadDataFromFile(self.trajlist, self.startframelist, self.trajlenlist, 'tartanvo_odom/poses.npy')
        if 'imu' in self.datatypelist:
            self.imulist = self.loadDataFromFile(self.trajlist, self.startframelist, self.trajlenlist, 'imu/imu.npy')
        if 'cost' in self.datatypelist:
            self.costlist = self.loadDataFromFile(self.trajlist, self.startframelist, self.trajlenlist, 'cost2/cost.npy')
            # self.cost2list = self.loadDataFromFile(self.trajlist, 'cost2/cost.npy') # for debug
        

    def parse_inputfile(self, inputfile, frame_skip):
        '''
        trajlist: [TRAJ0, TRAJ1, ...]
        trajlenlist: [TRAJLEN0, TRAJLEN1, ...]
        framelist: [FRAMESTR0, FRAMESTR1, ...]
        imulenlist: length of imu frames in each trajectory
                       [IMULen0, IMULen1, ...]
                       this is used to calculate the IMU frame index in __item__()                        
        '''
        with open(inputfile,'r') as f:
            lines = f.readlines()
        trajlist, trajlenlist, framelist, imulenlist, startframelist = [], [], [], [], []
        ind = 0
        while ind<len(lines):
            line = lines[ind].strip()
            traj, trajlen = line.split(' ')
            trajlen = int(trajlen)
            trajlist.append(traj)
            trajlenlist.append(trajlen)
            imulenlist.append(trajlen*self.imu_freq)
            ind += 1
            for k in range(trajlen):
                if ind>=len(lines):
                    print("Datafile Error: {}, line {}...".format(self.framelistfile, ind))
                    raise Exception("Datafile Error: {}, line {}...".format(self.framelistfile, ind))
                line = lines[ind].strip()
                framelist.append(line)
                ind += 1
                if k==0:
                    startframelist.append(int(line))

        print('Read {} trajectories, including {} frames'.format(len(trajlist), len(framelist)))
        return trajlist, trajlenlist, framelist, imulenlist, startframelist

    def calc_seq_len(self, datatypelist, seqlens, imu_freq):
        '''
        decide what is the sequence length for cutting the data, considering the different length of different modalities
        For now, all the modalities are at the same frequency except for the IMU which is faster by a factor of 'imu_freq'
        seqlens: the length of seq for each modality
        '''
        maxseqlen = 0
        for ttt, seqlen in zip(datatypelist, seqlens):
            if ttt=='imu': # IMU has a higher freqency than other modalities
                seqlen = int((float(seqlen+imu_freq-1)/imu_freq))
            if seqlen > maxseqlen:
                maxseqlen = seqlen
        return maxseqlen

    def parse_length(self, trajlenlist, skip, stride, sample_length): 
        '''
        trajlenlist: the length of each trajectory in the dataset
        skip: skip frames within sequence
        stride: skip frames between sequence
        sample_length: the sequence length 
        Return: 
        seqnumlist: the number of sequences in each trajectory
        the length of the whole dataset is the sum of the seqnumlist
        '''
        seqnumlist = []
        # sequence length with skip frame 
        # e.g. x..x..x (sample_length=3, skip=2, seqlen_w_skip=1+(2+1)*(3-1)=7)
        seqlen_w_skip = (skip + 1) * sample_length - skip
        # import ipdb;ipdb.set_trace()
        for trajlen in trajlenlist:
            # x..x..x---------
            # ----x..x..x-----
            # --------x..x..x-
            # ---------x..x..x <== last possible sequence
            #          ^-------> this starting frame number is (trajlen - seqlen_w_skip + 1)
            # stride = 4, skip = 2, sample_length = 3, seqlen_w_skip = 7, trajlen = 16
            # seqnum = (16 - 7)/4 + 1 = 3
            seqnum = int((trajlen - seqlen_w_skip)/ stride) + 1
            if trajlen<seqlen_w_skip:
                seqnum = 0
            seqnumlist.append(seqnum)
        return seqnumlist


    def getDataPath(self, trajstr, framestrlist, datatype):
        '''
        return the file path name wrt the data type and framestr
        '''
        datapathlist = []

        for framestr in framestrlist: 
            if datatype == 'img0':
                datapathlist.append(trajstr + '/image_left/' + framestr + '.png')
            if datatype == 'img1':
                datapathlist.append(trajstr + '/image_right/' + framestr + '.png')
            if datatype == 'imgc':
                datapathlist.append(trajstr + '/image_left_color/' + framestr + '.png')
            if datatype == 'disp0':
                datapathlist.append(trajstr + '/depth_left/' + framestr + '.npy')
            if datatype == 'heightmap':
                datapathlist.append(trajstr + '/height_map/' + framestr + '.npy')
            if datatype == 'rgbmap':
                datapathlist.append(trajstr + '/rgb_map/' + framestr + '.npy')

        return datapathlist

    def idx2traj(self, idx):
        '''
        handle the stride and the skip
        return: 1. the index of trajectory 
                2. the indexes of all the frames in a sequence
        '''
        # import ipdb;ipdb.set_trace()
        for k in range(self.trajnum):
            if idx < self.acc_seqlen[k+1]:
                break

        remainingframes = (idx-self.acc_seqlen[k]) * self.frame_stride
        frameind = self.acc_trajlen[k] + remainingframes
        imuframeind = self.acc_imulen[k] + remainingframes * self.imu_freq

        # put all the frames in the seq into a list
        frameindlist = []
        for w in range(self.sample_seq_len):
            frameindlist.append(frameind)
            frameind += self.frame_skip + 1
        return self.trajlist[k], frameindlist, imuframeind

    def odom2vel(self, odom):
        '''
        odom: N x 13 numpy array (x, y, z, rx, ry, rz, rw, vx, vy, xz, vrx, vry, vrz) in the global frame
        res: (forward vel, yaw)
        '''
        # import ipdb;ipdb.set_trace()
        ori_T = Rotation.from_quat(odom[:,3:7]).as_matrix()
        ori_T_inv = np.linalg.inv(ori_T)

        vel_trans = np.matmul(ori_T_inv, odom[:,7:10,np.newaxis]).squeeze(axis=-1) # N x 3
        vel_rot = np.matmul(ori_T_inv, odom[:,10:13,np.newaxis]).squeeze(axis=-1) # N x 3

        # if self.new_odom_flag:
        #     vel = vel_trans[:,0:1]
        # else:
        vel = vel_trans[:,1:2] # TODO: why??
        yaw = vel_rot[:,2:3]
        # print(vel_trans)
        # print(vel_rot)
        return np.concatenate((vel,yaw),axis=1).astype(np.float32)

    def filter_add_mask(self, heightlist, normalize=True):
        '''
        1. set unknown values to zero
        2. concatenate a mask channel: valid pixels are 1, invalid pixels are 0
        3. hard code normalization
        '''
        reslist = []
        for heightmap in heightlist:
            mask = heightmap[:,:,0] > 1000
            heightmap[mask,:] = 0.
            masknp = (1-mask).astype(np.float32).reshape(mask.shape + (1,))
            if normalize:
                heightmap = heightmap * 10
                heightmap[:,:,3] = heightmap[:,:,3] * 10 # std channel
                heightmap = np.clip(heightmap, -20, 20)
            heightmap_mask = np.concatenate((heightmap, masknp), axis=-1)
            heightmap_mask = heightmap_mask.transpose(2,0,1) # C x H x W
            reslist.append(torch.from_numpy(heightmap_mask)) # convert to tensor
        return torch.stack(reslist, dim=0)

    def random_hsv(self, rgblist):
        reslist = []
        for rgbmap in rgblist:
            h = (random.random()*2-1) * 30
            s = (random.random()*2-1) * 30
            v = (random.random()*2-1) * 30
            imghsv = cv2.cvtColor(rgbmap, cv2.COLOR_BGR2HSV)
            # import ipdb;ipdb.set_trace()
            imghsv = imghsv.astype(np.int16)
            imghsv[:,:,0] = np.clip(imghsv[:,:,0]+h, 0, 255)
            imghsv[:,:,1] = np.clip(imghsv[:,:,1]+s, 0, 255)
            imghsv[:,:,2] = np.clip(imghsv[:,:,2]+v, 0, 255)
            imghsv = imghsv.astype(np.uint8)
            reslist.append(cv2.cvtColor(imghsv,cv2.COLOR_HSV2BGR))
        return reslist

    def normalize_rgbmap(self, rgblist):
        reslist = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,1,3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,1,3)
        for rgbmap in rgblist:
            rgbmap = rgbmap.astype(np.float32)/255.
            rgbmap = (rgbmap - mean)/std
            rgbmap = rgbmap.transpose(2,0,1)
            reslist.append(torch.from_numpy(rgbmap))
        return torch.stack(reslist, dim=0)

    def normalize_vel(self, vel):
        vel[...,0] = vel[...,0] / 5.
        vel[...,1] = vel[...,1] * 5.
        return vel

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # import ipdb;ipdb.set_trace()
        # parse the idx to trajstr
        trajstr, frameindlist, imuframeind = self.idx2traj(idx)
        framestrlist = [self.framelist[k] for k in frameindlist]

        # hacking way to solve the odom inconsistency issue
        if trajstr.startswith('tartandrive_trajs'): # this is a trajectory from tartandrive
            new_odom = False
        elif trajstr.startswith('2022_traj') or trajstr.startswith('20220531'):
            new_odom = True
        else:
            assert False, "Unsupported trajectory {}".format(trajstr)

        # print(trajstr, new_odom)
        sample = {}
        for datatype, datalen in zip(self.datatypelist, self.modalitylenlist): 
            datafilelist = self.getDataPath(trajstr, framestrlist[:datalen], datatype)
            if datatype == 'img0' or datatype == 'img1' or datatype == 'imgc':
                imglist = self.load_image(datafilelist)
                if imglist is None:
                    print("!!!READ IMG ERROR {}, {}, {}".format(idx, trajstr, framestrlist, datafilelist))
                sample[datatype] = imglist
            elif datatype == 'disp0' or datatype == 'heightmap' or datatype=='rgbmap':
                datalist = self.load_numpy(datafilelist)
                if datatype == 'heightmap': 
                    datalist = self.filter_add_mask(datalist) # filter the very large and small numbers, add a mask channel
                if datatype == 'rgbmap':
                    datalist = self.random_hsv(datalist)
                    datalist = self.normalize_rgbmap(datalist) 
                sample[datatype] = datalist
            elif datatype == 'odom':
                odomlist, odomlist_tartanvo = self.load_odom(frameindlist, datalen)
                sample[datatype] = odomlist
                sample['odom_tartanvo'] = odomlist_tartanvo
            elif datatype == 'cmd':
                cmdlist = self.load_cmd(frameindlist, datalen)
                sample[datatype] = cmdlist
            elif datatype == 'cost':
                costlist = self.load_cost(frameindlist, datalen)
                sample[datatype] = costlist
                # sample['cost2'] = self.load_cost2(frameindlist, datalen) # for debug
            elif datatype == 'imu': 
                imulist = self.load_imu(imuframeind, datalen)
                sample[datatype] = imulist
            else:
                # print('Unknow Datatype {}'.format(datatype))
                pass

        # Load patches only after everything else is loaded
        if "patches" in self.datatypelist:
            datalen = self.modalitylenlist[self.datatypelist.index("patches")]
            assert 'odom' in sample and 'rgbmap' in sample and 'heightmap' in sample, "Data type error, cropping patches need odom, heightmap and rgbmap! "
            datalen_odom = self.modalitylenlist[self.datatypelist.index("odom")]
            assert datalen <= datalen_odom, "Data length error, odom should be more than patches!"

            patcheslist, masks = self.get_crops(sample["heightmap"], sample["rgbmap"], sample["odom"], 
                                                self.map_metadata, self.crop_params, coverage=self.coverage, new_odom=new_odom)
            sample["patches"] = patcheslist
            sample["masks"] = masks

        if "vels" in self.datatypelist:
            vels = self.odom2vel(sample["odom"])
            vels = self.normalize_vel(vels)
            sample['vels'] = vels

        # Transform.
        if ( self.transform is not None):
            sample = self.transform(sample)

        return sample

    def load_image(self, fns):
        imglist = []
        for fn in fns: 
            img = cv2.imread(self.dataroot + '/' + fn, cv2.IMREAD_UNCHANGED)
            imglist.append(img)
            assert img is not None, "Error loading image {}".format(fn)
        return imglist

    def load_numpy(self, fns):
        displist = []
        for fn in fns:
            disp = np.load(self.dataroot + '/' + fn)
            displist.append(disp)

        return displist

    def load_imu(self, startidx, len):
        return self.imulist[startidx: startidx+(len*(self.frame_skip+1)): self.frame_skip+1]

    def load_odom(self, frameindlist, datalen):
        odom_np = self.odomlist[frameindlist[:datalen]]
        odom_tartanvo_np = self.odomlist_tartanvo[frameindlist[:datalen]]
        return odom_np, odom_tartanvo_np 

    def load_cmd(self, frameindlist, datalen):
        return self.cmdlist[frameindlist[:datalen]]

    def load_cost(self, frameindlist, datalen):
        return self.costlist[frameindlist[:datalen]]

    # def load_cost2(self, frameindlist, datalen): # for debug
    #     return self.cost2list[frameindlist[:datalen]]

    def loadDataFromFile(self, trajlist, startframelist, trajlenlist, data_folder_and_filename):
        print('Loading data from {}...'.format(data_folder_and_filename))
        datalist = []
        for k, trajdir in enumerate(trajlist): 
            start_ind = int(startframelist[k])
            end_ind = start_ind + trajlenlist[k]
            trajpath = self.dataroot + '/' + trajdir
            cmds = np.load(trajpath + '/' + data_folder_and_filename).astype(np.float32) # framenum
            datalist.extend(cmds[start_ind:end_ind,...])
            if k%1000==0:
                print('    Processed {} trajectories...'.format(k))
        datalist = np.array(datalist)
        print('load size', datalist.shape)
        return datalist

    def get_crops(self, heightmaps, rgbmaps, odom, map_metadata, crop_params, coverage=False, new_odom=False):
        '''Returns (patches, costs)
        '''
        # rgb_map_tensor = torch.from_numpy(rgbmaps[0]).permute(2,0,1) # (C,H,W)
        # height_map_tensor = torch.from_numpy(heightmaps[0]).permute(2,0,1) # (C,H,W)
        rgb_map_tensor = rgbmaps[0] # (C,H,W)
        height_map_tensor = heightmaps[0] # (C,H,W)

        maps = {
            'rgb_map':rgb_map_tensor,
            'height_map':height_map_tensor
        }

        device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
        tm = TerrainMap(maps=maps, map_metadata=map_metadata, device=device)

        if coverage:
            local_path = get_coverage_path(map_metadata, stride=0.4, crop_size=crop_params['crop_size'])
        else:
            local_path = get_local_path(odom)
            if not new_odom:
                # if using GPS odom
                local_path = local_path[:, [1,0,2]] # swith x and y
                local_path[:,1] = -local_path[:,1]
        # if using tartan-vo
        # local_path[:,1:] = -local_path[:,1:]
        # masks: b x h x w x 2
        # patches: b x c x h x w
        local_path = torch.from_numpy(local_path)
        patches, masks = tm.get_crop_batch_and_masks(local_path, crop_params)
        masks = torch.permute(masks, (0, 3, 1, 2)) # b x 2 x h x w

        return patches, masks

def get_coverage_path(map_metadata, stride, crop_size):
    '''
    '''
    origin = map_metadata['origin'] # [-2, -6]
    map_height = map_metadata['height'] # 12
    map_width = map_metadata['width'] # 12

    minx = origin[0] + crop_size[0]/2
    maxx = origin[0] + map_height - crop_size[0]/2
    miny = origin[1] + crop_size[1]/2
    maxy = origin[1] + map_width - crop_size[1]/2

    pathlist = []
    for x in np.arange(minx, maxx+stride, stride):
        for y in np.arange(miny, maxy+stride, stride):
            pathlist.append([x, y, 0.0])
    return np.array(pathlist)

if __name__ == '__main__':
    from .utils import add_text
    import time
    from torch.utils.data import DataLoader
    # framelistfile = '/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/src/rosbag_to_dataset/post_processing/dataloader/tartandrive_train.txt'
    # datarootdir = '/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output'
    # datarootdir = '/cairo/arl_bag_files/SARA/sara_obs_traj'
    # framelistfile = '/home/amigo/workspace/pytorch/ss_costmap/data/sara_obs.txt'

    # datarootdir = '/cairo/arl_bag_files/SARA/2022_06_28_trajs'
    # framelistfile = '/home/amigo/workspace/pytorch/ss_costmap/data/20220628.txt'

    datarootdir = '/cairo/arl_bag_files/SARA/2022_05_31_trajs'
    framelistfile = '/home/amigo/workspace/pytorch/ss_costmap/data/rough_rider_high.txt'

    framelistfile = 'data/rough_rider.txt'
    datarootdir = '/cairo/arl_bag_files/SARA/2022_05_31_trajs'

    framelistfile = 'data/local_test.txt'
    datarootdir = '/cairo/arl_bag_files/SARA/test_loader/2022_traj'

    datatypes = "heightmap,rgbmap,odom,patches,cost,vels" #"img0,img1,imgc,disp0,heightmap,rgbmap,cmd,odom,imu"
    modalitylens = [1,1,10,10,10,10] # [10,10,10,10,10,10,10,10,100]
    datatypes = "heightmap,rgbmap,odom,patches" #"img0,img1,imgc,disp0,heightmap,rgbmap,cmd,odom,imu"
    modalitylens = [1,1,10,10] # [10,10,10,10,10,10,10,10,100]
    batch = 1
    workernum = 0

    map_height = 12.0
    map_width = 12.0
    resolution = 0.02

    crop_width = 4  # in meters
    crop_size = [crop_width, crop_width]
    output_size = [224, 224]


    map_metadata = {
        'height': map_height,
        'width': map_width,
        'resolution': resolution,
        'origin': [-2.0, -6.0],
    }

    crop_params ={
        'crop_size': crop_size,
        'output_size': output_size
    }

    # # test the coverage path
    # get_coverage_path(map_metadata, stride=0.2, crop_size=crop_size)

    stride = 2
    skip = 0
    dataset = TartanCostDataset(framelistfile, \
                            map_metadata=map_metadata,
                            crop_params=crop_params,
                            dataroot= datarootdir, \
                            datatypes = datatypes, \
                            modalitylens = modalitylens, \
                            transform=None, \
                            imu_freq = 10, \
                            frame_skip = skip, frame_stride=stride, new_odom_flag=False, coverage=False)
    print('Dataset length: ',len(dataset))

    # # test the speed
    # for k in range(0, len(dataset), 10):
    #     starttime = time.time()
    #     sample = dataset[k]
    #     print('Sample index: {}, load time {}'.format(k, time.time()-starttime))
    #     for k in sample: 
    #         e = sample[k]
    #         if isinstance(e, list):
    #             print('   {}: len {}, shape {}'.format(k, len(e), e[0].shape))
    #         elif isinstance(e, np.ndarray):
    #             print('   {}: shape {}'.format(k, e.shape))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,1,3)

    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workernum)
    dataiter = iter(dataloader)
    while True:
        starttime = time.time()
        try:
            sample = dataiter.next()
        except StopIteration:
            dataiter = iter(dataloader)
            sample = dataiter.next()

        rgbmap = sample['rgbmap'][0][0].permute(1,2,0).numpy()
        rgbmap = np.clip((rgbmap * std + mean) * 255, 0 ,255).astype(np.uint8)
        patches = sample['patches']
        masks = sample['masks'][0]
        # import ipdb;ipdb.set_trace()
        patches_numpy_list = []
        plotstride = 1

        for w in range(0, masks.shape[0], plotstride):
            # inds = masks[k].view(-1, 2)
            mask = masks[w].permute(1,2,0)
            inds = torch.cat((mask[0,:,:], mask[-1,:,:], mask[:,0,:], mask[:,-1,:]),dim=0)
            rgbmap[inds[:,0],inds[:,1],:] = [255,0,0]

        # cv2.imshow('img',patchvis)
        # cv2.waitKey(1)
        cv2.imshow('map',rgbmap)
        cv2.waitKey(0)

        print('Sample load time: {}'.format(time.time() - starttime))


    # # rgblist = []
    # # rgbind = 9

    # # import pdb;pdb.set_trace()
    # for k in range(40, 160, 1):
    #     sample = dataset[k]
    #     # import ipdb;ipdb.set_trace()

    #     rgbmap = sample['rgbmap'][0] #np.load(join(base_dir, rgbmap_folder, maplist[startframe]))
    #     heightmap = sample['heightmap'][0] #np.load(join(base_dir, heightmap_folder, maplist[startframe]))c

    #     # rgb_map_tensor = torch.from_numpy(rgbmap).permute(2,0,1) # (C,H,W)
    #     # height_map_tensor = torch.from_numpy(heightmap).permute(2,0,1) # (C,H,W)
    #     patches = sample['patches']
    #     masks = sample['masks']
    #     cost = sample['cost']
    #     # cost2 = sample['cost2']
    #     vels = sample['vels']
    #     print(vels)
    #     import ipdb;ipdb.set_trace()

    #     patches_numpy_list = []
    #     plotstride = 1
    #     for ind in range(0, len(patches), plotstride):
    #         ppp = patches[ind][:3].numpy().transpose((1,2,0)).astype(np.uint8)
    #         # costind = np.clip(int(ind),0,len(cost))
    #         # ppp = add_text(ppp.copy(), str(cost[costind])[:5])
    #         # patches_numpy_list.append(ppp) 
    #     # listhalflen = len(patches_numpy_list)//2
    #     # patchvis0 = np.concatenate(patches_numpy_list[:listhalflen], axis=1)
    #     # patchvis1 = np.concatenate(patches_numpy_list[listhalflen:], axis=1)
    #     # patchvis = np.concatenate((patchvis0, patchvis1), axis=0)

    #     for w in range(0, masks.shape[0], plotstride):
    #         # inds = masks[k].view(-1, 2)
    #         mask = masks[w].permute(1,2,0)
    #         inds = torch.cat((mask[0,:,:], mask[-1,:,:], mask[:,0,:], mask[:,-1,:]),dim=0)
    #         rgbmap[inds[:,0],inds[:,1],:] = [255,0,0]

    #     # cv2.imshow('img',patchvis)
    #     # cv2.waitKey(1)
    #     cv2.imshow('map',rgbmap)
    #     cv2.waitKey(0)
    #     # import ipdb;ipdb.set_trace()

    # #     rgblist.append(patches_numpy_list[rgbind])
    # #     rgbind -= 1

    # # patchvis = np.concatenate(rgblist, axis=1)
    # # cv2.imshow("patches", patchvis)
    # # cv2.waitKey(0)
    # import ipdb;ipdb.set_trace()
