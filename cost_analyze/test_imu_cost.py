# balance the distribution of the data
# 1. balance the cost distribution
# TODO: balance the speed distribution
# TODO: filter out the bad trajectories in terms of mapping

# from inspect import FrameInfo
import numpy as np
from os import listdir, system, mkdir, rmdir
from os.path import isdir, join
import matplotlib.pyplot as plt
from shutil import rmtree
np.set_printoptions(threshold=1000000, precision=2, suppress=True)

def load_costs(trajfolderdir, costfilename='cost.npy', costfolder = 'cost2', removelast=0):
    costfile = join(trajfolderdir, costfolder, costfilename)
    costs = np.load(costfile)
    # remove the last 20 frames because of the intervension actions
    if removelast>0:
        costs = costs[:-removelast] 
    return costs

def load_odom(trajfolderdir, odomfilename='odometry.npy', odomfolder='odom'):
    odomfile = join(trajfolderdir, odomfolder, odomfilename)
    odom = np.load(odomfile)
    return odom

def copy_data(targetfolder, rootdir, select_traj, select_costsind, costlist, odom_len):
    # copy those frames to a high/low cost folder
    # prepare the folders
    if isdir(targetfolder):
        rmtree(targetfolder) 
    mkdir(targetfolder)

    target_img_folder = targetfolder + '/image_left_color/' 
    target_height_folder = targetfolder + '/height_map/' 
    target_rgb_folder = targetfolder + '/rgb_map/' 
    target_odom_folder = targetfolder + '/odom/'
    target_cost_folder = targetfolder + '/cost/'

    if isdir(target_img_folder):
        rmtree(target_img_folder) 
    if isdir(target_height_folder):
        rmtree(target_height_folder) 
    if isdir(target_rgb_folder):
        rmtree(target_rgb_folder) 
    if isdir(target_odom_folder):
        rmtree(target_odom_folder) 
    if isdir(target_cost_folder):
        rmtree(target_cost_folder) 
    mkdir(target_img_folder)
    mkdir(target_height_folder)
    mkdir(target_rgb_folder)
    mkdir(target_odom_folder)
    mkdir(target_cost_folder)
    cmdlist = []
    # odomlist = []
    vellist = []
    # copy the image_left_color, height_map, rgb_map, odom, cost, cmd
    # image_left_color, height_map, rgb_map, odom, cmd are in separate frames
    framecount = 0
    for k,traj in enumerate(select_traj): 
        trajdir = rootdir + '/' + traj
        cmds = np.load(trajdir + '/cmd/twist.npy')
        motions = np.load(trajdir + '/tartanvo_odom/motions.npy')
        odoms = np.load(trajdir + '/odom/odometry.npy')
        costs = np.load(trajdir+'/cost2/cost.npy')
        for frameind in select_costsind[k]:
            imgcolorfile = trajdir + '/image_left_color/' + str(frameind).zfill(6) + '.png'
            heightmapfile = trajdir + '/height_map/' + str(frameind).zfill(6) + '.npy'
            rgbmapfile = trajdir + '/rgb_map/' + str(frameind).zfill(6) + '.npy'

            # imgcolorfile = trajdir + '/image_left/' + str(frameind).zfill(6) + '.png'
            # heightmapfile = trajdir + '/height_map_vo/' + str(frameind).zfill(6) + '.npy'
            # rgbmapfile = trajdir + '/rgb_map_vo/' + str(frameind).zfill(6) + '.npy'

            if frameind>=len(cmds) or frameind>=len(motions) or frameind+odom_len>=len(odoms):
                import ipdb;ipdb.set_trace()
            cmd = cmds[frameind] 
            motion = motions[frameind]
            odom = odoms[frameind:frameind+odom_len]
            cost = costs[frameind:frameind+odom_len]
            velx = motion[0] / 0.1 # dt = 0.1
            cmdlist.append(cmd)
            # odomlist.append(odom)
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
            np.save(target_odom_folder + str(framecount).zfill(6) + '.npy', odom)
            np.save(target_cost_folder + str(framecount).zfill(6) + '.npy', cost)
            framecount += 1

            if framecount%1000==0:
                print("  Copying {} frames".format(framecount))

    # np.save(targetfolder + '/odoms.npy', odomlist)
    np.save(targetfolder + '/cmds.npy', cmdlist)
    np.save(targetfolder + '/vels.npy', vellist)
    costlist = np.concatenate(costlist).reshape(-1)
    np.save(targetfolder + '/costs.npy', costlist)

def find_highcost_frames(sort1, rootdir, trajdirs, 
                        target_frame_number = 5000, 
                        avoid_consecutive_num = 5, 
                        cost_thresh=0.05, odom_len = 20,
                        savefilename = "select_highcost"):
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
        costs = load_costs(rootdir + '/' + trajdirs[ind], costfolder='cost2', removelast=odom_len)
        odoms = load_odom(rootdir + '/' + trajdirs[ind])
        if len(costs) < avoid_consecutive_num:
            continue
        select_traj.append(trajdirs[ind])
        sortind = np.argsort(costs)[::-1]
        motions = np.load(rootdir + '/' + trajdirs[ind] + '/tartanvo_odom/motions.npy')
        select_flag = np.zeros(len(costs)) # avoid select consecutive frames to increase diversity
        for k in sortind:
            # filter out frame w/o odom
            if abs(odoms[k][0]) < 1e-3 or abs(odoms[k][1]) < 1e-3 or abs(odoms[k][2]) < 1e-3: # odometry is missing
                continue
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
    plt.savefig(savefilename+'_'+str(target_frame_number)+'.png')
    plt.close()
    plt.hist(select_vellist, bins=100)
    plt.savefig(savefilename+'_'+str(target_frame_number)+'_vel.png')
    plt.close()    

    return select_traj, select_costsind, select_costslist

def bias_high_speed(speed):
    '''
    suppose speed: 0-10 m/s
    '''
    thresh = np.clip(speed, 0, 5)/5.0
    if np.random.rand() < thresh:
        return True
    return False

def find_lowcost_frames(sort1, rootdir, trajdirs, 
                        target_frame_number = 1000, 
                        avoid_consecutive_num = 20, 
                        cost_thresh=0.05, odom_len = 20,
                        savefilename = "select_lowcost"):
    select_costslist = []
    select_costsind = []
    select_traj = []
    select_vellist = []
    select_count = 0
    for ind in sort1: # enumerate trajectories in random order
        traj_costslist = []
        traj_costsind = []
        vel_list = []
        costs = load_costs(rootdir + '/' + trajdirs[ind], costfolder='cost2', removelast=odom_len)
        odoms = load_odom(rootdir + '/' + trajdirs[ind])
        if len(costs) < avoid_consecutive_num: # this trajectory is too short
            continue
        select_traj.append(trajdirs[ind])
        sortind = np.argsort(costs)
        motions = np.load(rootdir + '/' + trajdirs[ind] + '/tartanvo_odom/motions.npy')
        select_flag = np.zeros(len(costs)) # avoid select consecutive frames to increase diversity
        for k in sortind:
            # filter out frame w/o odom
            if abs(odoms[k][0]) < 1e-3 or abs(odoms[k][1]) < 1e-3 or abs(odoms[k][2]) < 1e-3: # odometry is missing
                continue
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
    plt.savefig(savefilename+'_'+str(target_frame_number)+'.png')
    plt.close()
    plt.hist(select_vellist, bins=100)
    plt.savefig(savefilename+'_'+str(target_frame_number)+'_vel.png')
    plt.close()
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

def test_odom(rootdir, trajfolder, odomfolder = 'odom', tartanvofolder = 'tartanvo_odom', framenum = -1):
    from .evaluator_base import ATEEvaluator
    trajdir = join(rootdir, trajfolder)
    odom = np.load(join(trajdir, odomfolder, 'odometry.npy'))
    odom[:,:3] = odom[:,:3] - odom[0,:3]

    tartanvo_motion = np.load(join(trajdir, tartanvofolder, 'motions.npy'))
    tartanvo_pose = np.load(join(trajdir, tartanvofolder, 'poses.npy'))

    odom = odom[:framenum]
    tartanvo_pose = tartanvo_pose[:framenum]

    atveval = ATEEvaluator()
    error, gt_traj, est_traj_aligned = atveval.evaluate(odom[:,:7],tartanvo_pose[:,:7], scale=False)
    est_traj_aligned = np.array(est_traj_aligned)
    print('Error:',error)

    # print(odom.shape, tartanvo_motion.shape, tartanvo_pose.shape)
    plt.subplot(221)
    plt.plot(gt_traj[:,0])
    plt.plot(est_traj_aligned[:,0],'--')
    plt.subplot(222)
    plt.plot(gt_traj[:,1])
    plt.plot(est_traj_aligned[:,1],'--')
    plt.subplot(223)
    plt.plot(gt_traj[:,2])
    plt.plot(est_traj_aligned[:,2],'--')
    plt.subplot(224)
    plt.plot(gt_traj[:,0], gt_traj[:,1])
    plt.plot(est_traj_aligned[:,0], est_traj_aligned[:,1],'--')
    plt.show()

def bad_odom(rootdir, odomfolder='odom'):
    '''
    see how many frames are bad
    '''
    trajdirs = listdir(rootdir)
    trajdirs = [join(rootdir, tt) for tt in trajdirs if isdir(join(rootdir, tt))]
    trajdirs.sort()

    badframenumlist = []
    framenumlist = []
    for traj in trajdirs:
        badframenum = 0
        saveprint = False
        odomfile = join(traj, 'odom/odometry.npy')
        odom = np.load(odomfile)
        print(traj, odom.shape)
        pos = odom[:,:3]
        delta_pos = pos[1:] - pos[:-1]
        for k in range(len(pos)-1):
            if delta_pos[k].max() > 10:
                print('  error frame {} - {}, {} - {}'.format(k, k+1, pos[k], pos[k+1]))
                badframenum += 1
            elif abs(pos[k][0]) < 1e-3 or abs(pos[k][1]) < 1e-3 or abs(pos[k][2]) < 1e-3:
                if not saveprint:
                    print('  zero frame {} , {} '.format(k, pos[k]))
                    saveprint = True
                badframenum += 1
            else:
                saveprint = False
        if abs(pos[-1][0]) < 1e-3 or abs(pos[-1][1]) < 1e-3 or abs(pos[-1][2]) < 1e-3:
            badframenum += 1
        badframenumlist.append(badframenum)
        framenum = len(pos)
        framenumlist.append(framenum)
        print('  odom missing for {}/{}'.format(badframenum,framenum))
    print('*** Total odom missing for {}/{} ***'.format(np.sum(np.array(badframenumlist)),np.sum(np.array(framenumlist))))

def test_main():
    # rootdir = '/project/learningphysics/tartandrive_trajs'
    rootdir = '/project/learningphysics/2022_traj'
    # trajfolder = '20220505_3'
    # trajfolder = '20221109_aggressive_sam6'
    # test_odom(rootdir, trajfolder,framenum = 1500)
    bad_odom(rootdir)

def main_tartandrive():
    HighCostSampleNum = 17000
    HighCostTrainNum = 15000
    LowCostSampleNum = 6000
    LowCostTrainNum = 5000

    highoutputfolder = 'highcost_' + str(HighCostTrainNum)
    highoutputfolder_val = 'highcost_val_' + str(HighCostSampleNum - HighCostTrainNum)
    lowoutputfolder = 'lowcost_' + str(LowCostTrainNum)
    lowoutputfolder_val = 'lowcost_val_' + str(LowCostSampleNum - LowCostTrainNum)
    OdomLen = 20

    rootdir = '/project/learningphysics/tartandrive_trajs'
    trajdirs = listdir(rootdir)

    trajdirs.sort()
    trajnum = len(trajdirs)
    costsmeanlist = []
    costsmaxlist = []
    for traj in trajdirs:
        costs = load_costs(rootdir + '/' + traj, costfolder='cost2')
        costsmeanlist.append(costs.mean())
        costsmaxlist.append(costs.max())
    sort1 = np.argsort(costsmeanlist)[::-1] # sort the trajectory wrt the mean cost
    sort2 = np.argsort(costsmaxlist)[::-1] # sort the trajectory wrt the max cost

    targetfolder = '/project/learningphysics/tartancost_data_2022/'
    select_traj, select_costsind, high_costlist = find_highcost_frames(sort1, rootdir, trajdirs, 
                                                    target_frame_number = HighCostSampleNum, 
                                                    avoid_consecutive_num = 3, cost_thresh=0.1, 
                                                    odom_len=OdomLen, savefilename = "select_highcost")
    train_traj, test_traj, train_costsind, test_costsind, train_costlist, test_costlist = split_train_val(select_traj, select_costsind, high_costlist, HighCostTrainNum)
    copy_data(targetfolder + highoutputfolder, rootdir, train_traj, train_costsind, train_costlist, odom_len= OdomLen)
    copy_data(targetfolder + highoutputfolder_val, rootdir, test_traj, test_costsind, test_costlist, odom_len= OdomLen)

    # randomly pick the low-cost frames
    sort1 = np.random.permutation(trajnum)
    select_traj, select_costsind, low_costlist = find_lowcost_frames(sort1, rootdir, trajdirs, 
                                                    target_frame_number = LowCostSampleNum, 
                                                    avoid_consecutive_num = 20, cost_thresh=0.1, 
                                                    odom_len=OdomLen, savefilename = "select_lowcost")
    train_traj, test_traj, train_costsind, test_costsind, train_costlist, test_costlist = split_train_val(select_traj, select_costsind, low_costlist, LowCostTrainNum)
    copy_data(targetfolder + lowoutputfolder, rootdir, train_traj, train_costsind, train_costlist, odom_len= OdomLen)
    copy_data(targetfolder + lowoutputfolder_val, rootdir, test_traj, test_costsind, test_costlist, odom_len= OdomLen)

def main_tartandrive_2022():
    HighCostSampleNum = 18000
    HighCostTrainNum = 15000
    LowCostSampleNum = 5000
    LowCostTrainNum = 4000

    highoutputfolder = 'highcost_' + str(HighCostTrainNum)
    highoutputfolder_val = 'highcost_val_' + str(HighCostSampleNum - HighCostTrainNum)
    lowoutputfolder = 'lowcost_' + str(LowCostTrainNum)
    lowoutputfolder_val = 'lowcost_val_' + str(LowCostSampleNum - LowCostTrainNum)
    OdomLen = 20

    rootdir = '/project/learningphysics/2022_traj'
    trajdirs = listdir(rootdir)

    trajdirs.sort()
    trajnum = len(trajdirs)
    costsmeanlist = []
    costsmaxlist = []
    for traj in trajdirs:
        costs = load_costs(rootdir + '/' + traj, costfolder='cost2')
        costsmeanlist.append(costs.mean())
        costsmaxlist.append(costs.max())
    sort1 = np.argsort(costsmeanlist)[::-1] # sort the trajectory wrt the mean cost
    sort2 = np.argsort(costsmaxlist)[::-1] # sort the trajectory wrt the max cost

    targetfolder = '/project/learningphysics/tartancost_data_2022/'
    select_traj, select_costsind, high_costlist = find_highcost_frames(sort1, rootdir, trajdirs, 
                                                    target_frame_number = HighCostSampleNum, 
                                                    avoid_consecutive_num = 3, cost_thresh=0.1, 
                                                    odom_len=OdomLen, savefilename = "select_highcost_2022")
    train_traj, test_traj, train_costsind, test_costsind, train_costlist, test_costlist = split_train_val(select_traj, select_costsind, high_costlist, HighCostTrainNum)
    copy_data(targetfolder+highoutputfolder, rootdir, train_traj, train_costsind, train_costlist, odom_len= OdomLen)
    copy_data(targetfolder+highoutputfolder_val, rootdir, test_traj, test_costsind, test_costlist, odom_len= OdomLen)

    # randomly pick the low-cost frames
    sort1 = np.random.permutation(trajnum)
    select_traj, select_costsind, low_costlist = find_lowcost_frames(sort1, rootdir, trajdirs, 
                                                    target_frame_number = LowCostSampleNum, 
                                                    avoid_consecutive_num = 10, cost_thresh=0.1, 
                                                    odom_len=OdomLen, savefilename = "select_lowcost_2022")
    train_traj, test_traj, train_costsind, test_costsind, train_costlist, test_costlist = split_train_val(select_traj, select_costsind, low_costlist, LowCostTrainNum)
    copy_data(targetfolder+lowoutputfolder, rootdir, train_traj, train_costsind, train_costlist, odom_len= OdomLen)
    copy_data(targetfolder+lowoutputfolder_val, rootdir, test_traj, test_costsind, test_costlist, odom_len= OdomLen)

def main_wanda():

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
        costs = load_costs(rootdir + '/' + traj, costfolder='cost2')
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


def cost_statistics(rootdir, savefile):
    trajdirs = listdir(rootdir)
    trajdirs = [tt for tt in trajdirs if isdir(join(rootdir, tt))]
    trajdirs.sort()
    trajnum = len(trajdirs)
    costlist = []
    for traj in trajdirs:
        costs = load_costs(rootdir + '/' + traj, costfilename='cost.npy', costfolder='cost2')
        costlist.extend(costs.tolist())
    print("Total number of frames {}".format(len(costlist)))
    print("Cost=0.0 {}, {}".format(np.sum(np.array(costlist)<1e-2), np.sum(np.array(costlist)<1e-3)/len(costlist)))
    print("Cost<0.1 {}, {}".format(np.sum(np.array(costlist)<0.1), np.sum(np.array(costlist)<0.1)/len(costlist)))
    print("Cost<0.3 {}, {}".format(np.sum(np.array(costlist)<0.3), np.sum(np.array(costlist)<0.3)/len(costlist)))
    print("Cost<0.5 {}, {}".format(np.sum(np.array(costlist)<0.5), np.sum(np.array(costlist)<0.5)/len(costlist)))
    print("Cost<0.99 {}, {}".format(np.sum(np.array(costlist)<0.99), np.sum(np.array(costlist)<0.99)/len(costlist)))

    plt.hist(costlist, bins=100, log=True)
    plt.savefig(savefile)

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

    # rootdir = '/project/learningphysics/tartandrive_trajs'
    # savefile = 'costs2_tartandrive.png'
    # rootdir = '/project/learningphysics/2022_traj'
    # savefile = 'costs2_tartandrive2.png'
    # cost_statistics(rootdir, savefile)
    # import ipdb;ipdb.set_trace()
    # test_main()
    main_tartandrive_2022()