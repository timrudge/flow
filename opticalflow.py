import cv2
import numpy as np

def div(F):
    gfx = np.gradient(F[:,:,0])
    gfy = np.gradient(F[:,:,1])
    return gfx[0]+gfy[1]
    

filestr = '/scratch/cellmodeller/data/ConstFluo-15-11-12-15-02/step-%05d.jpg'
frame1 = cv2.imread(filestr%0)
#frame1 = cv2.imread('/Users/timrudge/cellmodeller/data/ex1a_simpleGrowth2D-15-08-05-20-50/step-00116.jpg')
#frame1 = cv2.imread('/Users/timrudge/Pictures/TimRudge(PS)/seq/MAX_TimRudge_594AraE7_s1_t10000.tif')
#frame1 = cv2.imread('/Users/timrudge/Pictures/TimRudge(PS)/594/seq_avg_smooth/AVG_TimRudge_594AraE7_s1_t1-10000.tif')
#frame1 = cv2.imread('/Users/timrudge/Code/mim1.png')

#for _ in range(2):
#    frame1 = cv2.pyrDown(frame1)

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)


#outfilename = '/Users/timrudge/cellmodeller/data/ex1a_simpleGrowth2D-15-08-05-20-50/flow.csv'
outfilename = '/scratch/OpticalFlow/mflow.csv'

n_frames = 66 

# Make an array to store all flow velocity fields
flow = np.zeros(frame1.shape[0:2]+(2,n_frames))

for i in range(n_frames):
    print 'Processing frame %d'%i

    frame2 = cv2.imread(filestr%((i+1)*10))
    #frame2 = cv2.imread('/Users/timrudge/cellmodeller/data/ex1a_simpleGrowth2D-15-08-05-20-50/step-%05d.jpg'%((i+1)*10+116))
    #frame2 = cv2.imread('/Users/timrudge/Pictures/TimRudge(PS)/seq/MAX_TimRudge_594AraE7_s1_t1%04d.tif'%i)
    #frame2 = cv2.imread('/Users/timrudge/Pictures/TimRudge(PS)/594/seq_avg_smooth/AVG_TimRudge_594AraE7_s1_t1-1%04d.tif'%i)
    #frame2 = cv2.imread('/Users/timrudge/Code/mim2.png')
    
#    for _ in range(2):
#        frame2 = cv2.pyrDown(frame2)
    
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    #flow[:,:,:,i] = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow[:,:,:,i] = cv2.calcOpticalFlowFarneback(prvs,next,0.5, \
                                                         3, \
                                                         3, \
                                                         7, \
                                                         3, \
                                                         3, \
                                                         1)

    #strain = div(flow[:,:,:,i])
    #strainlow = cv2.pyrDown(strain)
    #strain.tofile('strain.csv', sep=',', format='%e')

    prvs = next

# Display shits
import matplotlib
from matplotlib import pyplot
from numpy import matlib

(w,h,_) = frame1.shape
#2362,2362
hx = np.matlib.repmat(np.arange(h),w,1).astype(np.float64)
hy = np.matlib.repmat(np.arange(w),h,1).astype(np.float64)
hy = np.transpose(hy)

pyplot.ion()
def_frame = frame1
#def_frame = cv2.imread('/Users/timrudge/Pictures/TimRudge(PS)/MaxProj/halves2.png')
#def_frame = cv2.imread('/Users/timrudge/Code/cf1test.png')
for f in range(n_frames):
    #frame = n_frames-1-f
    frame = f
    hhx = (hx - flow[:,:,0,frame]).astype('float32')
    hhy = (hy - flow[:,:,1,frame]).astype('float32')
    (chhx,chhy) = cv2.convertMaps(hhx,hhy,cv2.CV_32FC1)
    def_frame = cv2.remap(def_frame, hhx, hhy, cv2.INTER_CUBIC)
    pyplot.imshow(def_frame)
    cv2.imwrite('mim_def_frame_fwd_%05d.png'%(frame), def_frame)
    strain = div(flow[:,:,:,f])*255;
    cv2.imwrite('mim_def_frame_fwd_strain_%05d.png'%(frame), strain)



# Write to file
flow.tofile(outfilename,sep=',',format='%10.5f')
