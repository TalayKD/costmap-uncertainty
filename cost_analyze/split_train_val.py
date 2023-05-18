test_traj_list = [
    '20220531_test_0',
    '20220531_test_1',
    '20220531_test_2',
    '20220531_test_3',
    '20220505_8',
    '20220505_9',
    '20220630_18',
    '20220630_19',
    '20210910_8',
    '20210910_7',
    '20210903_146',
    '20210903_145',
    '20210903_144',
    '20210903_143',
    '20210903_99',
    '20210903_98',
    '20210903_97',
    '20210903_96',
    '20210902_99',
    '20210902_98',
    '20210902_97',
    '20210902_96',
    '20210902_95',
    '20210902_130',
    '20210902_131',
    '20210902_132',
    '20210902_133',
    '20210812_trial9',
    '20210812_trial8',
    '20210812_trial7'
]

file_balance = {
    'datafile/cost_high_2022_crop1_score3.0.txt': 5,
    'datafile/cost_high_td_crop1_score2.0.txt': 5,
    'datafile/cost_low_2022_crop1_score3.0.txt': 2,
    'datafile/cost_low_td_crop1_score2.0.txt': 2,
    'datafile/cost_mid_2022_crop1_score3.0.txt': 5,
    'datafile/cost_mid_td_crop1_score2.0.txt': 5,
    'datafile/cost_zero_2022_crop1_score3.0.txt': 1,
    'datafile/cost_zero_td_crop1_score2.0.txt': 1
}

outputfile = 'datafile/combine_train_crop1.txt'
outputfile_val = 'datafile/combine_test_crop1.txt'

# warthog5 arl_20220922
test_traj_list = [
    'smooth_dirt_low',
    'uniform_gravel_low_0',
]

file_balance = {
    'datafile/arl_cost_high_arl_20220922_traj_crop20_score0.5.txt': 8,
    'datafile/arl_cost_mid_arl_20220922_traj_crop20_score0.5.txt': 8,
    'datafile/arl_cost_low_arl_20220922_traj_crop20_score0.5.txt': 4,
    'datafile/arl_cost_zero_arl_20220922_traj_crop20_score0.5.txt': 1,
}

outputfile = 'datafile/arl_combine_train_crop20.txt'
outputfile_val = 'datafile/arl_combine_test_crop20.txt'

def parse_inputfile(inputfile):
    '''
    trajlist: [TRAJ0, TRAJ1, ...]
    trajlenlist: [TRAJLEN0, TRAJLEN1, ...]
    framelist: [FRAMESTR0, FRAMESTR1, ...]
    '''
    with open(inputfile,'r') as f:
        lines = f.readlines()
    trajlist, trajlenlist, framelist = [], [], []
    ind = 0
    while ind<len(lines):
        line = lines[ind].strip()
        traj, trajlen = line.split(' ')
        trajlen = int(trajlen)
        trajlist.append(traj)
        trajlenlist.append(trajlen)
        ind += 1
        frames = []
        for k in range(trajlen):
            if ind>=len(lines):
                print("Datafile Error: {}, line {}...".format(inputfile, ind))
                raise Exception("Datafile Error: {}, line {}...".format(inputfile, ind))
            line = lines[ind].strip()
            frames.append(line)
            ind += 1
        framelist.append(frames)
    print('Read {} trajectories, including {} frames'.format(len(trajlist), len(framelist)))
    return trajlist, trajlenlist, framelist

def write_traj(file, trajstr, trajlen, frames):
    file.write(trajstr + ' ' + str(trajlen) + '\n')
    for frame in frames:
        file.write(frame + '\n')

ftrain = open(outputfile, 'w')
ftest = open(outputfile_val, 'w')
traincount = 0
testcount = 0
for filename in file_balance:
    repeatnum = file_balance[filename]
    trajlist, trajlenlist, framelist = parse_inputfile(filename)    
    for k in range(repeatnum):
        for trajstr, trajlen, frames in zip(trajlist, trajlenlist, framelist): 
            # import ipdb;ipdb.set_trace()
            trajfolder = trajstr.split('/')[-1]
            if trajfolder in test_traj_list: # save traj to the test file
                write_traj(ftest, trajstr, trajlen, frames)
                testcount += trajlen
            else:
                write_traj(ftrain, trajstr, trajlen, frames)
                traincount += trajlen
            
ftrain.close()
ftest.close()

print("Split {} train, {} test".format(traincount, testcount))