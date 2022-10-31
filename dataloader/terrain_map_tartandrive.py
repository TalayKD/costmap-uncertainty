# terrain_map_tartandrive
import torch
import numpy as np

from .utils import quat_to_yaw

from scipy.spatial.transform import Rotation

def get_local_path(odom):
    pos = odom[:,:2]
    quat = odom[:,3:7]
    yaw = quat_to_yaw(quat)

    rel_pos = pos - pos[0,:]
    rel_yaw = yaw - yaw[0]

    theta = -yaw[0]
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # rel_pos = np.matmul(rot, rel_pos.transpose((1,0))).transpose((1,0))
    rel_pos = np.matmul(rot, rel_pos[...,np.newaxis]).squeeze()

    rel_path = np.concatenate((rel_pos, rel_yaw[...,np.newaxis]), axis=1)
    return rel_path
    
class TerrainMap:
    def __init__(self, maps = {}, map_metadata = {}, device='cpu'):
        self.maps = maps
        self.map_metadata = map_metadata
        self.device = device 

        for k,v in self.maps.items():
            if torch.is_tensor(v):
                self.maps[k] = v.to(self.device)
        for k,v in self.map_metadata.items():
            if torch.is_tensor(v):
                self.maps[k] = v.to(self.device)

        self.maps_tensor = torch.cat([self.maps['rgb_map'], self.maps['height_map']], dim=0)

        self.num_channels = self.maps_tensor.shape[0]

        self.resolution = self.map_metadata['resolution']
        self.origin = self.map_metadata['origin']

    def get_crop_batch_and_masks(self, poses, crop_params):
        '''Obtain an NxCxHxW tensor of crops for a given path.

        Procedure:
        1. Get initial meshgrid for crop in metric space centered around the origin
        2. Apply affine transform to all the crop positions (to the entire meshgrid) using batch multiplication:
            - The goal is to obtain a tensor of size [B, H, W, 2], where B is the batch size, H and W are the dimensions fo the image, and 2 corresponds to the actual x,y positions. To do this, we need to rotate and then translate every pair of points in the meshgrid. In batch multiplication, only the last two dimensions matter. That is, usually we need the following dimensions to do matrix multiplication: (m,n) x (n,p) -> (m,p). In batch multiplication, the last two dimensions of each array need to line up as mentioned above, and the earlier dimensions get broadcasted (more details in the torch matmul page). Here, we will reshape rotations to have shape [B,1,1,2,2] where B corresponds to batch size, the two dimensions with size 1 are there so that we can broadcast with the [H,W] dimensions in crop_positions, and the last two dimensions with size 2 reshape the each row in rotations into a rotation matrix that can left multiply a position to transform it. The output of torch.matmul(rotations, crop_positions) will be a [B,H,W,2,1] tensor. We will reshape translations to be a [B,1,1,2,1] vector so that we can add it to this output and obtain a tensor of size [B,H,W,2,1], which we will then squeeze to obtain our final tensor of size [B,H,W,2]
        3. Center around the right origin and rescale to obtain pixel coordinates
        4. Obtain map values at those pixel coordinates and handle invalid coordinates using a mask

        Note: 
        - All tensors will obey the following axis convention: [batch x crop_x x crop_y x transform_x x transform_y]
        - Input is in metric coordinates, so we flip the terrain map axes where necessary to match robot-centric coordinates
        
        Args:
            - path:
                Nx3 tensor of poses, where N is the number of poses to evaluate and each pose is represented as [x, y, yaw].
            - crop_params:
                Dictionary of params for crop:

                {'crop_size': [crop_size_x, crop_size_y],
                 'output_size': [output_size_x, output_size_y]}

                crop_size is in meters, output_size is in pixels

        Returns:
            - crops:
                Tensor of NxCxHxW of crops at poses on the path, where C is the number of channels in self.maps and N is the number of points in the path
        '''
        batch = poses.shape[0]
        ## Create initial crop template in metric space centered around (0,0) to generate all pixel values
        crop_height = crop_params['crop_size'][0] # In meters
        crop_width = crop_params['crop_size'][1] # In meters
        output_height = crop_params['output_size'][0] # In pixels
        output_width = crop_params['output_size'][1] # In pixels

        crop_xs = torch.linspace(-crop_height/2., crop_height/2., output_height).to(self.device)
        crop_ys = torch.linspace(-crop_width/2., crop_width/2., output_width).to(self.device)
        crop_positions = torch.stack(torch.meshgrid(crop_xs, crop_ys, indexing="ij"), dim=-1) # HxWx2 tensor

        ## Obtain translations and rotations for 2D rigid body transformation
        translations = poses[:, :2]  # Nx2 tensor, [x, y] in metric space
        yaws = poses[:,2]
        rotations = torch.stack([torch.cos(yaws), -torch.sin(yaws), torch.sin(yaws), torch.cos(yaws)], dim=-1)  # Nx4 tensor where each row corresponds to [cos(theta), -sin(theta), sin(theta), cos(theta)]

        ## Reshape tensors to perform batch tensor multiplication. 
        rotations = rotations.view(-1, 1, 1, 2, 2).float() #[B x 1 x 1 x 2 x 2]
        crop_positions = crop_positions.view(1, *crop_params['output_size'], 2, 1).float() #[1 x H x W x 2 x 1]
        translations = translations.view(-1, 1, 1, 2, 1).float() #[B x 1 x 1 x 2 x 1]

        # Apply each transform to all crop positions (res = [B x H x W x 2])
        crop_positions_transformed = (torch.matmul(rotations, crop_positions) + translations).squeeze()

        # Obtain actual pixel coordinates
        map_origin = torch.Tensor(self.map_metadata['origin']).view(1, 1, 1, 2).to(self.device)
        resolution = self.map_metadata['resolution']
        pixel_coordinates = ((crop_positions_transformed - map_origin) / resolution).long()  # .long() is needed so that we can use these as indices

        # Obtain maximum and minimum values of map to later filter out pixel locations that are out of bounds
        map_p_low = torch.tensor([0, 0]).to(self.device).view(1, 1, 1, 2)
        map_p_high = torch.tensor(self.maps_tensor.shape[1:]).to(self.device).view(1, 1, 1, 2)
        invalid_mask = (pixel_coordinates < map_p_low).any(dim=-1) | (pixel_coordinates >= map_p_high).any(dim=-1)  # If map is not square we might need to swap these axes as well

        #Indexing method: set all invalid idxs to a valid pixel position (i.e. 0) so that we can perform batch indexing, then mask out the results
        pixel_coordinates[invalid_mask] = 0 # [B, H, W, 2]
        pxlist = pixel_coordinates.view(-1, 2)


        #[B x C x W x H]
        # flipped_maps = self.maps_tensor.swapaxes(-1, -2) # To align with robot-centric coordinates since poses are in robot-centric coords.
        # flipped_maps = self.maps_tensor
        map_values  = self.maps_tensor[:, pxlist[:, 0], pxlist[:, 1]]
        # TODO: compare against grid_sample, accuracy and time

        # Reshape map values so that they go from [C, B*W*H] to [B, C, W, H]
        map_values = map_values.view(self.maps_tensor.shape[0], batch, *crop_params['output_size']).swapaxes(0,1)

        fill_value = 0  # TODO: Could have per-channel fill value as well

        # Convert invalid mask from [B, H, W] to [B, C, H, W] where C=1 so that we can perform batch operations
        invalid_mask = invalid_mask.unsqueeze(1).float()
        patches = (1.-invalid_mask)*map_values + invalid_mask*fill_value

        # # Swap order from [B, C, W, H] to [B, C, H, W]
        # patches = patches.swapaxes(-1, -2) 

        return patches, pixel_coordinates

    def get_crop_batch(self, poses, crop_params):
        patches, masks = self.get_crop_batch_and_masks(poses, crop_params)
        return patches


    # def get_labeled_crops(self, trajectory, crop_params):
        
    #     crop_width = crop_params['crop_size'][0] # Assume crop width and height are equal
    #     # Extract path from trajectory -> use odom to get path
    #     # NOTE: Path coordinates are in robot centric FLU metric coordinates
    #     state = trajectory['observation']['state']
    #     next_state = trajectory['next_observation']['state']

    #     cost = trajectory['observation']['traversability_cost']
    #     local_path = get_local_path_old(state, next_state)
   

    #     # Convert metric coordinates to pixel coordinates to be sampled
    #     crops = self.get_crop_path(local_path, crop_params)


    #     return crops, cost


    def get_rgb_map(self):
        return self.maps['rgb_map']
        

    def get_occupancy_grid(self):
        pass

    default_map_metadata = {
        'height': 10.0,
        'width': 10.0,
        'resolution': 0.05,
        'origin': [0.0, -5.0],
        # 'fields': ['rgb_map_r', 'rgb_map_g', 'rgb_map_r', 'height_map_low', 'height_map_high']
    }

    default_maps = {
        'rgb_map': torch.zeros(3, 200, 200),
        'height_map_low': torch.zeros(3, 200, 200),
        'height_map_high': torch.zeros(3, 200, 200)
    }

if __name__ == "__main__":
    from os.path import join
    from os import listdir
    import time
    import matplotlib.pyplot as plt
    import cv2
    from .utils import add_text
    
    # base_dir = '/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output/20210903_298'
    # base_dir = '/cairo/arl_bag_files/SARA/sara_obs_traj/sara_obs_0'
    base_dir = '/cairo/arl_bag_files/SARA/2022_06_28_trajs/20220628_5'
    odom_folder = 'odom' #'tartanvo_odom'
    rgbmap_folder = 'rgb_map'
    heightmap_folder ='height_map'

    # base_dir = '/cairo/arl_bag_files/SARA/sara_obs_traj_old/sara_obs_0'

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

    odom = np.load(join(base_dir, odom_folder, 'odometry.npy'))
    maplist = listdir(join(base_dir, rgbmap_folder))
    maplist = [mm for mm in maplist if mm.endswith('.npy')]
    maplist.sort()

    cost = np.load(join(base_dir, 'cost/cost.npy'))
    cost_shift = 0

    startframe = 0
    rgblist = []
    rgbind = 25
    # for startframe in range(startframe, len(maplist), 1):
    for startframe in range(startframe, len(maplist), 1):
        print(startframe)
        rgbmap = np.load(join(base_dir, rgbmap_folder, maplist[startframe]))
        heightmap = np.load(join(base_dir, heightmap_folder, maplist[startframe]))

        rgb_map_tensor = torch.from_numpy(rgbmap).permute(2,0,1) # (C,H,W)
        height_map_tensor = torch.from_numpy(heightmap).permute(2,0,1) # (C,H,W)

        maps = {
            'rgb_map':rgb_map_tensor,
            'height_map':height_map_tensor
        }

        tm = TerrainMap(maps=maps, map_metadata=map_metadata, device="cpu")

        local_path = get_local_path(odom[startframe:startframe+26,:])
        # print(local_path)
        # import ipdb;ipdb.set_trace()

        # if using GPS odom
        # local_path = local_path[:, [1,0,2]] # swith x and y
        # local_path[:,1] = -local_path[:,1]
        # if using tartan-vo
        # local_path[:,1:] = -local_path[:,1:] # change y, z, yaw
        local_path = torch.from_numpy(local_path)
        patches, masks = tm.get_crop_batch_and_masks(local_path, crop_params)
        # import ipdb;ipdb.set_trace()

        patches_numpy_list = []
        stride = 1
        for ind in range(0, len(patches), stride):
            ppp = patches[ind][:3].numpy().transpose((1,2,0)).astype(np.uint8)
            costind = np.clip(int(startframe+ind+cost_shift),0,len(cost))
            ppp = add_text(ppp.copy(), str(cost[costind])[:5])
            patches_numpy_list.append(ppp) 
        listhalflen = len(patches_numpy_list)//2
        patchvis0 = np.concatenate(patches_numpy_list[:listhalflen], axis=1)
        patchvis1 = np.concatenate(patches_numpy_list[listhalflen:], axis=1)
        patchvis = np.concatenate((patchvis0, patchvis1), axis=0)

        for k in range(0, masks.shape[0], stride):
            # inds = masks[k].view(-1, 2)
            inds = torch.cat((masks[k,0,:,:], masks[k,-1,:,:], masks[k,:,0,:], masks[k,:,-1,:]),dim=0)
            rgbmap[inds[:,0],inds[:,1],:] = [255,0,0]

        cv2.imshow('img',patchvis)
        cv2.waitKey(1)
        cv2.imshow('map',rgbmap)
        cv2.waitKey(1)
        # import ipdb;ipdb.set_trace()

    #     rgblist.append(patches_numpy_list[rgbind])
    #     rgbind -= 1

    # patchvis = np.concatenate(rgblist, axis=1)
    # cv2.imshow("img", patchvis)
    # cv2.waitKey(0)
    import ipdb;ipdb.set_trace()

