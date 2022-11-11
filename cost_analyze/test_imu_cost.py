# balance the distribution of the data
# 1. balance the cost distribution
# TODO: balance the speed distribution
# TODO: filter out the bad trajectories in terms of mapping

import numpy as np
from os import listdir, system, mkdir, rmdir
from os.path import isdir, join
import matplotlib.pyplot as plt
from shutil import rmtree
np.set_printoptions(threshold=1000000, precision=2, suppress=True)

def load_costs(costfiledir, costfilename='float.npy', costfolder = 'cost'):
    costfile = join(costfiledir, costfolder, costfilename)
    costs = np.load(costfile)
    # remove the last 20 frames because of the intervension actions
    costs = costs[:-20] 
    return costs

def copy_data(targetfolder, rootdir, select_traj, select_costsind, costlist):
    # copy those frames to a high/low cost folder
    # prepare the folders
    target_img_folder = targetfolder + '/image_left_color/' 
    target_height_folder = targetfolder + '/height_map/' 
    target_rgb_folder = targetfolder + '/rgb_map/' 
    if isdir(target_img_folder):
        rmtree(target_img_folder) 
    if isdir(target_height_folder):
        rmtree(target_height_folder) 
    if isdir(target_rgb_folder):
        rmtree(target_rgb_folder) 
    mkdir(target_img_folder)
    mkdir(target_height_folder)
    mkdir(target_rgb_folder)
    cmdlist = []
    odomlist = []
    vellist = []
    # copy the image_left_color, height_map, rgb_map, odom, cost, cmd
    # image_left_color, height_map, rgb_map, odom, cmd are in separate frames
    framecount = 0
    for k,traj in enumerate(select_traj): 
        trajdir = rootdir + '/' + traj
        cmds = np.load(trajdir + '/cmd/twist.npy')
        motions = np.load(trajdir + '/tartanvo_odom/motions.npy')
        odoms = np.load(trajdir + '/odom/odometry.npy')
        for frameind in select_costsind[k]:
            # imgcolorfile = trajdir + '/image_left_color/' + str(frameind).zfill(6) + '.png'
            # heightmapfile = trajdir + '/height_map/' + str(frameind).zfill(6) + '.npy'
            # rgbmapfile = trajdir + '/rgb_map/' + str(frameind).zfill(6) + '.npy'

            imgcolorfile = trajdir + '/image_left/' + str(frameind).zfill(6) + '.png'
            heightmapfile = trajdir + '/height_map_vo/' + str(frameind).zfill(6) + '.npy'
            rgbmapfile = trajdir + '/rgb_map_vo/' + str(frameind).zfill(6) + '.npy'

            if frameind>=len(cmds) or frameind>=len(motions) or frameind>=len(odoms):
                import ipdb;ipdb.set_trace()
            cmd = cmds[frameind] 
            motion = motions[frameind]
            odom = odoms[frameind]
            velx = motion[0] / 0.1 # dt = 0.1
            cmdlist.append(cmd)
            odomlist.append(odom)
            vellist.append(velx)

            cmd = 'ln -s ' + imgcolorfile + ' ' + target_img_folder+str(framecount).zfill(6) + '.png'
            system(cmd)
            cmd = 'ln -s ' + heightmapfile + ' ' + target_height_folder+str(framecount).zfill(6) + '.npy'
            system(cmd)
            cmd = 'ln -s ' + rgbmapfile + ' ' + target_rgb_folder+str(framecount).zfill(6) + '.npy'
            system(cmd)
            cmd = 'ln -s ' + heightmapfile.replace('npy', 'png') + ' ' + target_height_folder+str(framecount).zfill(6) + '.png'
            system(cmd)
            cmd = 'ln -s ' + rgbmapfile.replace('npy', 'png') + ' ' + target_rgb_folder+str(framecount).zfill(6) + '.png'
            system(cmd)
            framecount += 1
    np.save(targetfolder + '/odoms.npy', odomlist)
    np.save(targetfolder + '/cmds.npy', cmdlist)
    np.save(targetfolder + '/vels.npy', vellist)
    costlist = np.concatenate(costlist).reshape(-1)
    np.save(targetfolder + '/costs.npy', costlist)

def find_highcost_frames(sort1, rootdir, trajdirs, target_frame_number = 5000, avoid_consecutive_num = 5, cost_thresh=0.05):
    # find the high cost frames in those 50 trajectories
    select_costslist = []
    select_costsind = []
    select_traj = []
    select_vellist = []
    select_count = 0
    for ind in sort1: # enumerate trajectories from high cost to low cost
        traj_costslist = []
        traj_costsind = []
        vel_list = []
        costs = load_costs(rootdir + '/' + trajdirs[ind])
        if len(costs) < avoid_consecutive_num:
            continue
        select_traj.append(trajdirs[ind])
        sortind = np.argsort(costs)[::-1]
        motions = np.load(rootdir + '/' + trajdirs[ind] + '/tartanvo_odom/motions.npy')
        select_flag = np.zeros(len(costs)) # avoid select consecutive frames to increase diversity
        for k in sortind:
            velx = motions[k][0] / 0.1 # dt = 0.1
            if select_flag[k]==0: 
                traj_costslist.append(costs[k])
                traj_costsind.append(k)
                vel_list.append(velx)
                select_flag[max(0,k-avoid_consecutive_num):min(k+avoid_consecutive_num, len(costs))] = 1
                select_count += 1
            if costs[k]<cost_thresh or select_count==target_frame_number:
                break
        # select_trajlen.append(k+1)
        print("{}: Find {}/{} frames: {}".format(trajdirs[ind], len(traj_costsind), len(costs), traj_costsind))
        print("Costs: {}\n".format(np.array(traj_costslist)))
        select_costsind.append(traj_costsind)
        select_costslist.append(traj_costslist)
        select_vellist.extend(vel_list)
        if select_count==target_frame_number:
            break

    high_costlist = np.concatenate(select_costslist).reshape(-1)
    print("Total number of frames selected {} from {} trajs".format(len(high_costlist), len(select_traj)))
    plt.hist(high_costlist, bins=100)
    plt.savefig('select_hightcost_12k.png')
    plt.close()
    plt.hist(select_vellist, bins=100)
    plt.savefig('select_hightcost_12k_vel.png')

    return select_traj, select_costsind, select_costslist

def bias_high_speed(speed):
    '''
    suppose speed: 0-10 m/s
    '''
    thresh = np.clip(speed, 0, 5)/5.0
    if np.random.rand() < thresh:
        return True
    return False

def find_lowcost_frames(sort1, rootdir, trajdirs, target_frame_number = 1000, avoid_consecutive_num = 20, cost_thresh=0.05):
    select_costslist = []
    select_costsind = []
    select_traj = []
    select_vellist = []
    select_count = 0
    for ind in sort1: # enumerate trajectories in random order
        traj_costslist = []
        traj_costsind = []
        vel_list = []
        costs = load_costs(rootdir + '/' + trajdirs[ind])
        if len(costs) < avoid_consecutive_num: # this trajectory is too short
            continue
        select_traj.append(trajdirs[ind])
        sortind = np.argsort(costs)
        motions = np.load(rootdir + '/' + trajdirs[ind] + '/tartanvo_odom/motions.npy')
        select_flag = np.zeros(len(costs)) # avoid select consecutive frames to increase diversity
        for k in sortind:
            velx = motions[k][0] / 0.1 # dt = 0.1
            if select_flag[k]==0 and bias_high_speed(velx): 
                traj_costslist.append(costs[k])
                traj_costsind.append(k)
                vel_list.append(velx)
                select_flag[max(0,k-avoid_consecutive_num):min(k+avoid_consecutive_num, len(costs))] = 1
                select_count += 1
            if costs[k]>cost_thresh or select_count==target_frame_number:
                break
        # select_trajlen.append(k+1)
        print("{}: Find {}/{} frames: {}".format(trajdirs[ind], len(traj_costsind), len(costs), traj_costsind))
        print("Vels: {}".format(np.array(vel_list)))
        print("Costs: {}\n".format(np.array(traj_costslist)))
        select_costsind.append(traj_costsind)
        select_costslist.append(traj_costslist)
        select_vellist.extend(vel_list)
        if select_count==target_frame_number:
            break

    high_costlist = np.concatenate(select_costslist).reshape(-1)
    print("Total number of frames selected {} from {} trajs".format(len(high_costlist), len(select_traj)))
    plt.hist(high_costlist, bins=100)
    plt.savefig('select_lowcost_6k.png')
    plt.close()
    plt.hist(select_vellist, bins=100)
    plt.savefig('select_lowcost_6k_vel.png')
    assert len(select_traj)==len(select_costsind)
    return select_traj, select_costsind, select_costslist

def split_train_val(select_traj, select_costsind, costlist, trainnum):
    totalnum = sum([len(costs) for costs in costlist])
    assert trainnum <= totalnum
    trajnum = len(select_traj)
    trajrandinds = np.random.permutation(trajnum) # shuffle the trajectories
    train_traj = []
    test_traj = []
    train_costsind = []
    test_costsind = []
    train_costlist = []
    test_costlist = []
    train_count = 0
    for k in range(trajnum):
        trajind = trajrandinds[k]
        trajlen = len(select_costsind[trajind])
        if trajlen + train_count <= trainnum: # add the whole traj to the trainlist
            train_traj.append(select_traj[trajind])
            train_costsind.append(select_costsind[trajind])
            train_costlist.append(costlist[trajind])
            train_count += trajlen
        else: # split the traj for both train and test
            # import ipdb;ipdb.set_trace()
            remainnum = trainnum - train_count
            train_traj.append(select_traj[trajind])
            train_costsind.append(select_costsind[trajind][:remainnum])
            train_costlist.append(costlist[trajind][:remainnum])
            train_count += remainnum

            test_traj.append(select_traj[trajind])
            test_costsind.append(select_costsind[trajind][remainnum:])
            test_costlist.append(costlist[trajind][remainnum:])

        if train_count == trainnum:
            break

    for w in range(k+1, trajnum):
        trajind = trajrandinds[w]
        test_traj.append(select_traj[trajind])
        test_costsind.append(select_costsind[trajind])
        test_costlist.append(costlist[trajind])

    trainnum = sum([len(costs) for costs in train_costlist])
    testnum = sum([len(costs) for costs in test_costlist])
    print("Split {} train, {} test".format(trainnum, testnum))

    return train_traj, test_traj, train_costsind, test_costsind, train_costlist, test_costlist

def main_tartandrive():
    HighCostSampleNum = 12000
    LowCostSampleNum = 6000
    HighCostTrainNum = 10000
    LowCostTrainNum = 5000

    rootdir = '/project/learningphysics/tartandrive_trajs'
    trajdirs = listdir(rootdir)

    trajdirs.sort()
    trajnum = len(trajdirs)
    costsmeanlist = []
    costsmaxlist = []
    for traj in trajdirs:
        costs = load_costs(rootdir + '/' + traj)
        costsmeanlist.append(costs.mean())
        costsmaxlist.append(costs.max())
    sort1 = np.argsort(costsmeanlist)[::-1] # sort the trajectory wrt the mean cost
    sort2 = np.argsort(costsmaxlist)[::-1] # sort the trajectory wrt the max cost

    targetfolder = '/project/learningphysics/tartancost_data/'
    select_traj, select_costsind, high_costlist = find_highcost_frames(sort1, rootdir, trajdirs, 
                                                    target_frame_number = HighCostSampleNum, 
                                                    avoid_consecutive_num = 5, cost_thresh=0.1)
    train_traj, test_traj, train_costsind, test_costsind, train_costlist, test_costlist = split_train_val(select_traj, select_costsind, high_costlist, HighCostTrainNum)
    copy_data(targetfolder+'highcost_10k', rootdir, train_traj, train_costsind, train_costlist)
    copy_data(targetfolder+'highcost_val_2k', rootdir, test_traj, test_costsind, test_costlist)

    # randomly pick the low-cost frames
    sort1 = np.random.permutation(trajnum)
    select_traj, select_costsind, low_costlist = find_lowcost_frames(sort1, rootdir, trajdirs, 
                                                    target_frame_number = LowCostSampleNum, 
                                                    avoid_consecutive_num = 20, cost_thresh=0.05)
    train_traj, test_traj, train_costsind, test_costsind, train_costlist, test_costlist = split_train_val(select_traj, select_costsind, low_costlist, LowCostTrainNum)
    copy_data(targetfolder+'lowcost_5k', rootdir, train_traj, train_costsind, train_costlist)
    copy_data(targetfolder+'lowcost_val_1k', rootdir, test_traj, test_costsind, test_costlist)

def cost_statistics():
    rootdir = '/project/learningphysics/tartandrive_trajs'
    trajdirs = listdir(rootdir)

    trajdirs.sort()
    trajnum = len(trajdirs)
    costlist = []
    for traj in trajdirs:
        costs = load_costs(rootdir + '/' + traj)
        costlist.extend(costs.tolist())
    print("Total number of frames {}".format(len(costlist)))
    plt.hist(costlist, bins=100)

# maxinds = np.concatenate((sort1[-20:],sort2[-10:]),axis=0)
# maxinds = sort1[-50:] # look at the 50 trajectoris w/ the highest mean cost

# costslist = []
# for ind in maxinds:
#     costfile = rootdir + '/' + trajdirs[ind] + '/cost/cost.npy'
#     costs = load_costs(costfile)
#     print(trajdirs[ind], len(costs))
#     costslist.extend(costs.tolist())
# print("Total number of frames {}".format(len(costslist)))
# # plt.hist(costslist, bins=100)
# # plt.savefig('costs_top_mean_50.png')

if __name__ == '__main__':

    HighCostSampleNum = 600
    LowCostSampleNum = 300
    HighCostTrainNum = 500
    LowCostTrainNum = 200

    rootdir = '/project/learningphysics/tartancost_wanda_traj'
    trajdirs = listdir(rootdir)

    trajdirs.sort()
    trajnum = len(trajdirs)
    costsmeanlist = []
    costsmaxlist = []
    for traj in trajdirs:
        costs = load_costs(rootdir + '/' + traj)
        costsmeanlist.append(costs.mean())
        costsmaxlist.append(costs.max())
    sort1 = np.argsort(costsmeanlist)[::-1] # sort the trajectory wrt the mean cost
    sort2 = np.argsort(costsmaxlist)[::-1] # sort the trajectory wrt the max cost

    targetfolder = '/project/learningphysics/tartancost_wanda/'
    select_traj, select_costsind, high_costlist = find_highcost_frames(sort1, rootdir, trajdirs, 
                                                    target_frame_number = HighCostSampleNum, 
                                                    avoid_consecutive_num = 3, cost_thresh=0.1)
    train_traj, test_traj, train_costsind, test_costsind, train_costlist, test_costlist = split_train_val(select_traj, select_costsind, high_costlist, HighCostTrainNum)
    copy_data(targetfolder+'highcost_500', rootdir, train_traj, train_costsind, train_costlist)
    copy_data(targetfolder+'highcost_val_100', rootdir, test_traj, test_costsind, test_costlist)

    # randomly pick the low-cost frames
    sort1 = np.random.permutation(trajnum)
    select_traj, select_costsind, low_costlist = find_lowcost_frames(sort1, rootdir, trajdirs, 
                                                    target_frame_number = LowCostSampleNum, 
                                                    avoid_consecutive_num = 20, cost_thresh=0.05)
    train_traj, test_traj, train_costsind, test_costsind, train_costlist, test_costlist = split_train_val(select_traj, select_costsind, low_costlist, LowCostTrainNum)
    copy_data(targetfolder+'lowcost_200', rootdir, train_traj, train_costsind, train_costlist)
    copy_data(targetfolder+'lowcost_val_100', rootdir, test_traj, test_costsind, test_costlist)

    import ipdb;ipdb.set_trace()


