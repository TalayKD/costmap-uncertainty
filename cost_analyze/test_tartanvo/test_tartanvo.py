import numpy as np
from os import listdir, system, mkdir, rmdir
from os.path import isdir, join, isfile, isdir
import matplotlib.pyplot as plt
from shutil import rmtree
import time
np.set_printoptions(threshold=1000000, precision=2, suppress=True)


class FileLogger():
    def __init__(self, filename, overwrite=False):
        if isfile(filename):
            if overwrite:
                print('Overwrite existing file {}'.format(filename))
            else:
                timestr = time.strftime('%m%d_%H%M%S',time.localtime())
                filename = filename+'_'+timestr
        self.f = open(filename, 'w')

    def log(self, logstr):
        print(logstr)
        self.f.write(logstr)

    def logline(self, logstr):
        print(logstr)
        self.f.write(logstr+'\n')

    def close(self,):
        self.f.close()

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

def extract_good_seqs(bad_mask):
    seqs = []
    startind, endind = 0, 0 # split the traj into good sequences
    k = 0
    while k < len(bad_mask):
        if not bad_mask[k]:
            startind = k
            while k < len(bad_mask) and not bad_mask[k]:
                k += 1
            endind = k
            seqs.append([startind, endind])
        k += 1
    return seqs

def filter_odom(traj_odom):
    # in tartandrive y and z is zero
    # in data 2022 x and z is zero
    bad_mask = np.logical_and(np.logical_or(abs(traj_odom[:,0]) < 1e-3, abs(traj_odom[:,1]) < 1e-3), abs(traj_odom[:,2]) < 1e-3)
    seqs = extract_good_seqs(bad_mask)
    return seqs, bad_mask

def load_odom_filter(rootdir, trajfolder, odomfolder = 'odom', tartanvofolder = 'tartanvo_odom', filter_odom_flag=True):
    trajdir = join(rootdir, trajfolder)
    odom = np.load(join(trajdir, odomfolder, 'odometry.npy'))
    pose = odom[:,:7]

    # tartanvo_motion = np.load(join(trajdir, tartanvofolder, 'motions.npy'))
    tartanvo_pose = np.load(join(trajdir, tartanvofolder, 'poses.npy'))
    if filter_odom_flag:
        seqs, bad_mask = filter_odom(pose) 
        if len(seqs) > 0:
            seq = seqs[0]
            pose = pose[seq[0]:seq[1]]
            tartanvo_pose = tartanvo_pose[seq[0]:seq[1]]
    return pose, tartanvo_pose

def plot_traj(traj1, traj2):
    plt.plot(traj1[:,0], traj1[:,1],'--', linewidth=3)
    plt.plot(traj2[:,0], traj2[:,1], linewidth=3)
    plt.show()

def evaluate_tartanvo_odom(pose, tartanvo_pose, framenum = 50, stride = 10):
    '''
    evaluate the tartanvo odom by comparing it to the GPS odom
    return score N x 1 array, lower the better, if the the frames that odom is missing, the score is a large value
    '''
    from .evaluator_base import ATEEvaluator
    atveval = ATEEvaluator()
    trajlen = len(pose)
    assert trajlen >= framenum, 'Trajectory too short {}'.format(trajlen)

    scores = np.zeros(trajlen, dtype=np.float32)
    scores_count = np.zeros(trajlen, dtype=np.int32)
    # a lot of trajectories have missing the odom topic
    # filter the odom 

    for k in range(0, trajlen-framenum, stride):
        sub_odom = pose[k:k+framenum]
        sub_tartanvo = tartanvo_pose[k:k+framenum]

        error, gt_traj, est_traj_aligned = atveval.evaluate(sub_odom, sub_tartanvo, scale=False)
        scores[k:k+framenum] = scores[k:k+framenum] + error
        scores_count[k:k+framenum] = scores_count[k:k+framenum] + 1

        # if error > 0.1: #error > 5:
        #     print(k, error)
        #     plot_traj(gt_traj, est_traj_aligned)

    # calclulate sore for the last section
    if trajlen%stride != 0:
        last_odom = pose[-framenum:]
        last_tartanvo = tartanvo_pose[-framenum:]

        error, gt_traj, est_traj_aligned = atveval.evaluate(last_odom, last_tartanvo, scale=False)
        scores[-framenum:] = scores[-framenum:] + error
        scores_count[-framenum:] = scores_count[-framenum:] + 1

    scores = scores / (scores_count + 1e-5)
    return scores

def get_color(cm, value, minv, maxv, inverse=False):
    value = min(max(value, minv), maxv)
    value = (value-minv)/(maxv-minv)
    if inverse:
        value = 1-value
    return cm(value)

def plot_trajs_with_score(scores, traj1, traj2):
    from .evaluator_base import ATEEvaluator
    atveval = ATEEvaluator()

    error, gt_traj, est_traj_aligned = atveval.evaluate(traj1,traj2, scale=False)
    est_traj_aligned = np.array(est_traj_aligned)
    cm = plt.cm.get_cmap('Spectral')

    plt.plot(traj1[:,0], traj1[:,1],'--', linewidth=3)
    for k in range(0, len(traj1), 10):
        c = get_color(cm, scores[k], 0, 5, inverse=True)
        plt.plot(est_traj_aligned[k:k+10,0], est_traj_aligned[k:k+10,1],color=c, linewidth=3)
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
        odomfile = join(traj, odomfolder, 'odometry.npy')
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
    rootdir = '/home/wenshan/tmp/arl_data'
    trajfolder = '20221109_aggressive_sam' #'20221109_aggressive_sam6'#'20220628_aggresive_13' #'20221109_aggressive_sam'#'20220505_6'#'20220505_3'
    # trajfolder = '20221109_aggressive_sam6'
    # 20220505_3
    # 20220505_6
    # 20220628_aggresive_13
    # 20221109_aggressive_sam
    # 20221109_aggressive_sam6
    # test_odom(rootdir, trajfolder,framenum = 1500)
    # bad_odom(rootdir)
    pose, tartanvo_pose = load_odom_filter(rootdir, trajfolder, filter_odom_flag=True)
    # odom = np.load(join(rootdir, trajfolder, 'odom/odometry.npy'))
    # mask, seqs = filter_odom(odom)
    scores= evaluate_tartanvo_odom(pose, tartanvo_pose)
    print (scores)
    import ipdb;ipdb.set_trace()
    # ind1,ind2 = 400,1290
    plot_trajs_with_score(scores[:], pose[:], tartanvo_pose[:])

    # print(seqs)
    # import ipdb;ipdb.set_trace()

def gen_scores():
    # rootdir = '/project/learningphysics/2022_traj'
    # rootdir = '/project/learningphysics/tartandrive_trajs'
    rootdir = '/project/learningphysics/arl_20220922_traj'
    odomfolder = 'odom'
    tartanvofolder = 'tartanvo_odom'
    ate_thresh = 0.5
    bad_score = 1000

    logger = FileLogger(join(rootdir, 'tartanvo_score_'+str(ate_thresh)+'.log'))
    trajlist = listdir(rootdir)
    trajlist = [tt for tt in trajlist if isdir(join(rootdir, tt))]
    trajlist.sort()
    logger.logline("Find {} trajectories".format(len(trajlist)))

    goodnumlist = []
    validnumlist = []
    totoalframenum = 0
    for traj in trajlist:
        logger.logline("Working on traj {}".format(traj))
        trajdir = join(rootdir, traj)
        # load the odoms
        odom = np.load(join(trajdir, odomfolder, 'odometry.npy'))
        pose = odom[:,:7]
        tartanvo_pose = np.load(join(trajdir, tartanvofolder, 'poses.npy'))

        trajlen = len(pose)
        totoalframenum += trajlen
        scores = np.zeros(trajlen, dtype=np.float32)
        seqs, bad_mask = filter_odom(pose) # filter the frames without odom
        scores[bad_mask] = bad_score # assign bad score to the missing frames
        validframenum = 0
        goodframenum = 0
        if len(seqs) == 0:
            logger.logline("  Odom missing!")
        else:
            # logger.logline("  Total frame {}, Odom available for {}".format(trajlen, seqs))
            for seq in seqs:
                sub_pose = pose[seq[0]:seq[1]]
                sub_tartanvo_pose = tartanvo_pose[seq[0]:seq[1]]
                subseqlen = (seq[1]-seq[0])
                if subseqlen < 50: # hard code
                    logger.logline("  Subseq too short {}!".format(subseqlen))
                    scores[seq[0]:seq[1]] = bad_score
                    continue
                sub_scores= evaluate_tartanvo_odom(sub_pose, sub_tartanvo_pose)
                scores[seq[0]:seq[1]] = np.array(sub_scores)
                validframenum += subseqlen

                goodframes = sub_scores < ate_thresh
                goodframenum += np.sum(goodframes)

        np.save(join(trajdir, tartanvofolder, 'scores.npy'), scores)
        np.savetxt(join(trajdir, tartanvofolder, 'scores.txt'), scores)
        goodnumlist.append(goodframenum)
        validnumlist.append(validframenum)
        logger.logline('  -- Valid frames {}/{}'.format(validframenum, trajlen))
        logger.logline('  -- Good frames {}/{}'.format(goodframenum, trajlen))

    totalgoodnum = np.sum(np.array(goodnumlist))
    totalvalidnum = np.sum(np.array(validnumlist))
    logger.logline('*** Total valid frames {}/{}'.format(totalvalidnum, totoalframenum))
    logger.logline('*** Total Good frames {}/{}'.format(totalgoodnum, totoalframenum))

if __name__ == '__main__':

    # rootdir = '/project/learningphysics/tartandrive_trajs'
    # savefile = 'costs2_tartandrive.png'
    # rootdir = '/project/learningphysics/2022_traj'
    # savefile = 'costs2_tartandrive2.png'
    # cost_statistics(rootdir, savefile)
    # import ipdb;ipdb.set_trace()
    gen_scores()