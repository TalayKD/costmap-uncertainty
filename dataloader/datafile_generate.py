from os.path import isfile, join, isdir
from os import listdir
import os
import numpy as np

# == Generate txt file for tartan dataset ===

def process_traj(trajdir, folderstr = 'image_left'):
    imglist = listdir(join(trajdir, folderstr))
    imglist = [ff for ff in imglist if ff[-3:]=='png']
    imglist.sort()
    imgnum = len(imglist)

    lastfileind = -1
    outlist = []
    framelist = []
    for k in range(imgnum):
        filename = imglist[k]
        framestr = filename.split('_')[0].split('.')[0]
        frameind = int(framestr)

        if frameind==lastfileind+1: # assume the index are continuous
            framelist.append(framestr)
        else:
            if len(framelist) > 0:
                outlist.append(framelist)
                framelist = []
        lastfileind = frameind

    if len(framelist) > 0:
        outlist.append(framelist)
        framelist = []
    print('Find {} trajs, traj len {}'.format(len(outlist), [len(ll) for ll in outlist]))

    return outlist 


def enumerate_trajs(data_root_dir):
    trajfolders = listdir(data_root_dir)    
    trajfolders = [ee for ee in trajfolders if isdir(data_root_dir+'/'+ee)]
    trajfolders.sort()
    print('Detected {} trajs'.format(len(trajfolders)))
    return trajfolders


def write_datafile(trajdir, trajindlist, outputfile): 
    for trajinds in trajindlist:
        outputfile.write(trajdir)
        outputfile.write(' ')
        outputfile.write(str(len(trajinds)))
        outputfile.write('\n')
        for ind in trajinds:
            outputfile.write(ind)
            outputfile.write('\n')

def sara_obs():
    dataset_dir = '/cairo/arl_bag_files/SARA/sara_obs_traj'
    datafile = '/home/amigo/workspace/pytorch/ss_costmap/data/sara_obs.txt'
    trajlist = enumerate_trajs(dataset_dir)
    f = open(datafile, 'w')
    for trajdir in trajlist:
        trajindlist = process_traj(join(dataset_dir, trajdir))
        write_datafile(trajdir, trajindlist, f)
    f.close()

def rough_rider():
    dataset_dir = '/cairo/arl_bag_files/SARA/2022_05_31_trajs'
    datafile = '/home/amigo/workspace/pytorch/ss_costmap/data/rough_rider.txt'
    trajlist = enumerate_trajs(dataset_dir)
    f = open(datafile, 'w')
    for trajdir in trajlist:
        trajindlist = process_traj(join(dataset_dir, trajdir))
        write_datafile(trajdir, trajindlist, f)
    f.close()

def bag20210910_23():
    dataset_dir = '/cairo/arl_bag_files/SARA/20210910_23'
    datafile = '/home/amigo/workspace/pytorch/ss_costmap/data/bag20210910_23.txt'

    f = open(datafile, 'w')
    trajindlist = process_traj(dataset_dir)
    write_datafile('20210910_23', trajindlist, f)
    f.close()

def arl_local():
    dataset_dir = '/cairo/arl_bag_files/sara_traj'
    datafile = '/home/amigo/workspace/pytorch/ss_costmap/data/arl_local.txt'
    # trajlist = enumerate_trajs(dataset_dir)
    trajlist = ['uniform_gravel_low_20220922']
    f = open(datafile, 'w')
    for trajdir in trajlist:
        trajindlist = process_traj(join(dataset_dir, trajdir))
        write_datafile(trajdir, trajindlist, f)
    f.close()

def arl():
    dataset_dir = '/project/learningphysics/arl_20220922_traj'
    datafile = '../data/arl_20220922.txt'
    # trajlist = enumerate_trajs(dataset_dir)
    trajlist = [
        'flat_loop_long_run',
        'smooth_dirt_high',
        'smooth_dirt_low',
        'smooth_dirt_mid',
        'uniform_gravel_high',
        'uniform_gravel_low_0',
        'uniform_gravel_low_1',
        'vegetation_high',
        'vegetation_low',
        'vegetation_mid_1',
        'woods_hill_loop_high',
        'woods_hill_loop_low_2',
        'woods_hill_loop_mid',
        'woods_logs_slopes_long',
        'woods_loop_low',
    ]
    f = open(datafile, 'w')
    for trajdir in trajlist:
        trajindlist = process_traj(join(dataset_dir, trajdir))
        write_datafile(trajdir, trajindlist, f)
    f.close()

if __name__=="__main__":
    arl()
