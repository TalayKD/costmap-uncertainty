import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from math import pi

def quat2matrix(pose):
    '''
    N x 7 pose
    Return N x 4 x 4 transform matrix
    '''
    num = len(pose)
    res = np.tile(np.eye(4), (num,1,1))
    res[:,:3,3] = pose[:,:3] # translation
    ori_mat = Rotation.from_quat(pose[:,3:]).as_matrix()
    res[:,:3,:3] = ori_mat
    return res

def euler2matrix(pose):
    '''
    N x 7 pose
    Return N x 4 x 4 transform matrix
    '''
    num = len(pose)
    res = np.tile(np.eye(4), (num,1,1))
    res[:,:3,3] = pose[:,:3] # translation
    ori_mat = Rotation.from_euler(pose[:,3:],"ZYX",degrees=False).as_matrix()
    res[:,:3,:3] = ori_mat
    return res


np.set_printoptions(precision=2, suppress=True, threshold=100000)
odomfile = '/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output/20210903_42/odom/odometry.npy'
# odomfile = '/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output/20210903_298/odom/odometry.npy'
# odomfile = '/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output/20210903_10/odom/odometry.npy'
# odomfile = '/cairo/arl_bag_files/SARA/2022_06_28_trajs/20220628_3/odom/odometry.npy'
# odomfile = '/cairo/arl_bag_files/SARA/2022_06_28_trajs/20220628_5/odom/odometry.npy'
odom = np.load(odomfile)

print(odom)
# odom = odom[0:500]
# odom = odom[100:]
pos = odom[:,:3] - odom[0,:3]
vel = odom[:,7:10]

# vel2 = pos[1:,:] - pos[:-1,:]
# plt.subplot(311)
# plt.plot(vel, '.-')
# plt.subplot(312)
# plt.plot(vel2, '.-')
# plt.subplot(313)
# plt.plot(pos, '.-')
# plt.show()

ori_quat = odom[:,3:7]
ori_euler = Rotation.from_quat(ori_quat).as_euler("XYZ", degrees=False)
w_euler = odom[:,10:]
w_euler2 = ori_euler[1:,:] - ori_euler[:-1,:]
w_euler2[w_euler2<-pi] = w_euler2[w_euler2<-pi] + pi*2
w_euler2[w_euler2>pi] = w_euler2[w_euler2>pi] - pi*2

# plt.subplot(311)
# plt.plot(w_euler, '.-')
# plt.subplot(312)
# plt.plot(w_euler2, '.-')
# plt.subplot(313)
# plt.plot(ori_euler, '.-')
# plt.show()

# convert the vel to the local frame
ori_T = quat2matrix(odom[:,:7])
ori_T_inv = np.linalg.inv(ori_T)
ori_diff = np.matmul(ori_T_inv[:-1,:,:], ori_T[1:,:,:])
vel_trans = ori_diff[:,:3,3]
vel_rot = Rotation.from_matrix(ori_diff[:,:3,:3]).as_euler("XYZ",degrees=False)

# import ipdb;ipdb.set_trace()
vel_trans0 = np.matmul(ori_T_inv[:,:3,:3], vel[...,np.newaxis]).squeeze()
vel_rot0 = np.matmul(ori_T_inv[:,:3,:3], odom[:,10:,np.newaxis]).squeeze()
plt.subplot(411)
plt.plot(vel_trans0, '.-')
plt.subplot(412)
plt.plot(vel_trans, '.-')
plt.subplot(413)
plt.plot(vel_rot0, '.-')
plt.subplot(414)
plt.plot(vel_rot, '.-')
plt.show()

# # visualize tartanvo 
# tartanvofile = '/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output/20210903_42/tartanvo_odom/motions.npy'
# motion = np.load(tartanvofile)
# rot_euler = Rotation.from_quat(motion[:,3:]).as_euler("XYZ",degrees=False)
# plt.subplot(211)
# plt.plot(motion[:,:3], '.-')
# plt.subplot(212)
# plt.plot(rot_euler, '.-')
# plt.show()
