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

if __name__=="__main__":
    rough_rider()