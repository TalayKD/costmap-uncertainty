import numpy as np
import cv2
from os import listdir
from os.path import isdir, isfile
from scipy.spatial.transform import Rotation

# The visualization view looks like this: 
# - vel_body
# - slope angle
#          RGB   
# -----------------------
# Heightmap | RGBmap


if __name__ == '__main__':

    imgvissize = (512, 256)
    mapvissize = (256, 256)
    dt = 0.5
    root = '/project/learningphysics/tartancost_wanda'
    outfolderlist = ['highcost_500',
                    'highcost_val_100', 
                    'lowcost_200',
                    'lowcost_val_100']
    save_to = root + '/data_preview'

    for outfolder in outfolderlist:
        print('--- {} ---'.format(outfolder))
        trajdir = root + '/' + outfolder

        if not isdir(trajdir):
            print('!!! Trajectory Not Found {}'.format(trajdir))
            continue

        if not isdir(trajdir + '/image_left_color') or \
             not isdir(trajdir + '/rgb_map') or \
             not isdir(trajdir + '/height_map'):
             print('!!! Missing data folders')

        outvidfile = save_to + '/wanda_' + outfolder + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fout=cv2.VideoWriter(outvidfile, fourcc, 1.0/dt, (512, 512))

        imglist = listdir(trajdir + '/image_left_color')
        imglist = [trajdir + '/image_left_color/' + img for img in imglist if img.endswith('.png')]
        imglist.sort()

        rgbmaplist = listdir(trajdir + '/rgb_map')
        rgbmaplist = [trajdir + '/rgb_map/' + img for img in rgbmaplist if img.endswith('.npy')]
        rgbmaplist.sort()

        heightmaplist = listdir(trajdir + '/height_map')
        heightmaplist = [trajdir + '/height_map/' + img for img in heightmaplist if img.endswith('.npy')]
        heightmaplist.sort()

        cmds = np.load(trajdir + '/cmds.npy')
        vels = np.load(trajdir + '/vels.npy')
        costs = np.load(trajdir + '/costs.npy')

        datanum = len(imglist)
        for k in range(datanum):
            # import ipdb;ipdb.set_trace()
            disp1 = cv2.imread(imglist[k])
            disp1 = disp1[32:-32, 64:-64, :] # crop and resize the image in the same way with the stereo matching code
            disp1 = cv2.resize(disp1, imgvissize)

            disp3 = np.load(rgbmaplist[k])
            disp3 = cv2.resize(disp3, mapvissize)
            disp3 = cv2.flip(disp3, -1)

            disp4 = np.load(heightmaplist[k])
            mask = disp4[:,:,0]>10000
            disp4 = disp4[:,:,2] # mean channel
            disp4 = np.clip((disp4 - (-1.5))*73, 0, 255).astype(np.uint8) # convert height to 0-255
            disp4[mask] = 0
            disp4 = cv2.resize(disp4, mapvissize)
            disp4 = cv2.flip(disp4, -1)
            disp4 = cv2.applyColorMap(disp4, cv2.COLORMAP_JET)

            disp = np.concatenate((disp1, np.concatenate((disp3, disp4), axis=1)), axis=0)

            cmd = cmds[k] 
            cost = costs[k]
            velx = vels[k]

            text1 = "{} Throttle: {:.2f},    Steering: {:.2f}".format(str(k).zfill(4), cmd[0], cmd[1])
            text2 = "      Velx:    {:.2f} m/s".format(velx)
            text3 = "      Cost:    {:.2f}".format(cost)

            # pts = np.array([[0,0],[320,0],[320,20],[0,20]],np.int32)
            # put a bg rect
            disp[10:75, 0:270, :] = disp[10:75, 0:270, :]/5 * 2 + np.array([70, 40, 10],dtype=np.uint8)/5 * 3
            # cv2.fillConvexPoly(disp, pts, (70,30,10))
            cv2.putText(disp,text1, (15,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,210,245),thickness=1)
            cv2.putText(disp,text2, (15,45),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,210,245),thickness=1)
            cv2.putText(disp,text3, (15,65),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,210,245),thickness=1)

            # print('cmd {}, velx {}, yaw {}, slope {}'.format(cmd, velx, yaw, slope))
            # cv2.imshow('img', disp)
            # cv2.waitKey(0)

            fout.write(disp)
        fout.release()