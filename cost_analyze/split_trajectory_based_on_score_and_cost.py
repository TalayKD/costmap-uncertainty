import numpy as np
from os import listdir
from os.path import isdir, isfile, join

def extract_good_seqs(good_mask):
    seqs = []
    startind, endind = 0, 0 # split the traj into good sequences
    k = 0
    while k < len(good_mask):
        if good_mask[k]:
            startind = k
            while k < len(good_mask) and good_mask[k]:
                k += 1
            endind = k
            seqs.append([startind, endind])
        k += 1
    return seqs

def split_trajectory_based_on_score_and_cost():
    # rootdir = '/project/learningphysics/tartandrive_trajs'
    rootdir = '/project/learningphysics/2022_traj'
    # rootdir = '/cairo/arl_bag_files/SARA/2022_05_31_trajs'

    rootfolder = rootdir.split('/')[-1]
    trajlist = listdir(rootdir)
    trajlist = [tt for tt in trajlist if isdir(join(rootdir, tt))]
    trajlist.sort()
    print("Find {} trajectories".format(len(trajlist)))

    outputfilelist = [
        'cost_high', 
        'cost_mid',
        'cost_low',
        'cost_zero'
    ]
    costrangelist = [
        [0.5, 1.0],
        [0.3, 0.5],
        [0.1, 0.3],
        [0.0, 0.1],
    ]

    costfolder = 'cost2'
    tartanvofolder = 'tartanvo_odom'
    ate_thresh = 3.0
    cropnum = 10 # write this number of frame more at the end of each seq

    total_good_frames = 0
    for outputfile, costrange in zip(outputfilelist, costrangelist):
        # import ipdb;ipdb.set_trace()
        print("==> Filter cost range {}-{}".format(costrange[0], costrange[1]))
        outputfile = outputfile + '_' + rootdir.split('/')[-1] + '_crop'+str(cropnum)+ '_score'+str(ate_thresh)+'.txt' 
        f = open(join('datafile', outputfile), 'w')
        for traj in trajlist:
            print("Working on traj {}".format(traj))
            trajdir = join(rootdir, traj)

            # filter the score
            scores = np.load(join(trajdir, tartanvofolder, 'scores.npy'))
            good_frames = scores < ate_thresh
            trajlen = len(scores)

            # filter the starting zero vel
            motions = np.load(join(trajdir, tartanvofolder, 'motions.npy'))
            motions = np.concatenate((motions, motions[-1:,:])) # add one more frame  
            velx = motions[:,0] / 0.1 # hard coded
            ind = 0 
            while ind < trajlen:
                if velx[ind] < 1:
                    good_frames[ind] = False
                else:
                    break
                ind += 1

            # filter the ending zero vel and intervention
            ind = trajlen-1
            while ind >= 0:
                if velx[ind] < 1:
                    good_frames[ind] = False
                else:
                    # delete a few more for intervention
                    good_frames[max(ind-19,0):ind+1] = False # hard coded
                    break
                ind -= 1

            # filter the cost range
            costs = np.load(join(trajdir, costfolder, 'cost.npy'))
            good_frames[costs<costrange[0]] = False
            good_frames[costs>costrange[1]] = False

            # write sub seqs to file
            seqs = extract_good_seqs(good_frames)
            if len(seqs) > 0:
                print("  Split into {} seqs {}".format(len(seqs), seqs))
                for seq in seqs:
                    seqlen = seq[1]-seq[0]
                    # make sure the seq has 10 frames more after the last frame
                    assert seq[1]+20 <=trajlen, 'seq {}-{} is too close to the end {}!'.format(seq[0], seq[1], trajlen)
                    f.write(rootfolder +'/'+ traj + ' ' + str(seqlen+cropnum-1) + '\n')
                    for frames in range(seq[0], seq[1]+cropnum-1): # write
                        f.write(str(frames).zfill(6))
                        f.write('\n')
                    total_good_frames += seqlen
            else:
                print("  No frame available! ")
        f.close()
    print("*** Processed {} good frames".format(total_good_frames))

if __name__=="__main__":
    split_trajectory_based_on_score_and_cost()