import numpy as np
from robot1 import robot
from pprint import pprint as prnt
import random
import pdb
#import pdb
import copy
from time import clock
from time import sleep

ini_pose = dict(x=15,y=-20,orientation=0)
ini_noise = dict(new_f_noise=0.5,
                 new_t_noise=2*np.pi/180,
                 new_d_noise=0.67,
                 new_a_noise=0.024) # forward,turn,distance,angle
# World param
landmarks = [[random.randint(-50,60),random.randint(-45,55)] for x in range(60)]
# landmarks  = [[40.0, -20.0],[30,20],[50,15],[40.0, 40.0],[10,45],\
#               [-20.0, 40.0],[-30,10],[-20.0, -20.0],\
#               [10,-30],[-40,-40]]
num_landmarks = len(landmarks)
seen = [False for row in range(num_landmarks)]
world = dict(world_size = 100.0,
             measurement_range = 40.0,
             num_landmarks = num_landmarks,
             angle_vision = np.pi/4,
             world_landmarks = landmarks)

Vxy = 5.0 # velocity
Wxy = 0.2 # theta

data = []

# Simulation Param
N = 10*1 # number of particles
Steps = 32 # number of steps to run


# init robot
robotCar = robot()
robotCar.set(**ini_pose)
robotCar.set_world(**world)
robotCar.init_seen(num_landmarks)
robotCar.set_noise(**ini_noise)

#init particles
p = []
for i in range(N):
    p.append(robot())
for i in p:
    i.set(**ini_pose)
    i.set_noise(**ini_noise)
    i.set_world(**world)
    i.init_seen(num_landmarks)
    i.weight = 1.0/N


# run simulation
#f = open('DATA_SLAM.txt','w')
#############################
######## Aux function ######
def new_particle(list_index,p):
    p3 = []

    for index in list_index:
        t = copy.deepcopy(p[index])
        t.set_weight(1.0/N)
        p3.append(t)
    return p3
#run()
#f.close()


from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="Simulation of FastSLAM 1.0")
win.resize(500,500)
win.setWindowTitle('FastSLAM 1.0')
label = pg.LabelItem(justify = "right")
win.addItem(label)
# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

#p1 = win.addPlot(title="Updating plot")

p1 = win.addPlot(row = 1, col = 0)


#p1.setXRange(-50,60)
#p1.setYRange(-45,55)

vb = p1.vb
def getCoord(evt):
    # mousePointX = p1.vb.mapSceneToView(event[0]).x()
    # mousePointY = p1.vb.mapSceneToView(event[1]).y()
    mousePoint = vb.mapSceneToView(evt[0])
    label.setText("<span style='font-size: 14pt; color: white'> x = %0.2f, <span style='color: white'> y = %0.2f</span>"\
     % (mousePoint.x(), mousePoint.y()))

proxy = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=getCoord)

# landmarks view
landmarks_plot = p1.plot(pen=None ,symbol='d')
# l_points  = np.array([[40.0, -20.0],[50,15],[40.0, 40.0],[10,45],\
#               [-20.0, 40.0],[-30,10],[-20.0, -20.0],\
#               [10,-30],[-40,-40]])
l_points = np.array(landmarks)

#data = np.random.normal(size=(10,1000))

landmarks_plot.setData(l_points)

# Real motion view
motion = p1.plot(pen='y')
pen = pg.mkPen(cosmetic=False, width=4.5, color='r')

# particle view
particle = []
for c in range(N):
    particle.append(p1.plot(pen=None,symbol='+',brush=pg.mkBrush(255, 255, 255, 120)))
group_particles = p1.plot(pen=None,symbol='o')

ptr = 1
n= 0
data = []
data_particles = []
p1.enableAutoRange('xy', False)
def update():
    global ptr, p,robotCar,Vxy,Wxy,n
    if n < Steps:
        tnt = clock()
        n+=1
        #robotCar =
        robotCar.move(Wxy,Vxy,False) # move robot
        #print "\n[X: ",robotCar.x,",Y: ",robotCar.y,"]"
        data.append([robotCar.x,robotCar.y])
        motion.setData(np.array(data))
        #f.write(str(robotCar.x)+" "+str(robotCar.y)+"\n")
        Z = robotCar.sense() # sense enviroment


        for i_ in range(len(Z)):
            seen[Z[i_][0]] = True

        #predict Z observation to landmarks
        # Compute
        w = []
        p3 = []
        do_resample = False
        # print "before measurement", [p[i].get_weight() for i in range(len(p))]
        # print "!Z",not(Z)
        for i_p,p_ in enumerate(p):

            p[i_p].move(Wxy,Vxy) # move particles

            if Z:
                p[i_p].update_measurement(Z) # update mean,cov of landmarks
                do_resample = True

            # apply weights to particles according readings
        #pdb.set_trace()
        w = [p[i].get_weight() for i in range(len(p))]
        normalizer = sum(w)

        # print "normalizer, sum(w)", normalizer
        # print w
        wn = map(lambda x: x/normalizer, w) #normalized weights
        for e in range(len(p)):
            p[e].weight = p[e].weight/normalizer
        #     w[e] = w[e]/normalizer
        wv = np.array(wn) # vector form
        # print [p[i].get_weight() for i in range(len(p))]
        neff = 1/sum(wv**2)
        # print "neff",neff
        # print "do_resample", do_resample
        # print "neff < N/2", neff < N/2.0
        if do_resample and (neff < N*.5): #and seen.count(True) != len(landmarks)-1:
            # print "doing resampling"

            # low-variance resample
            index = int(random.random() * N)
            u = 0.0
            list_index = []
            mw = max(w)
            for i in range(N):
                u += random.random() * i * mw
                while u > w[index]:
                    u -= w[index]
                    index = (index + 1) % N
                list_index.append(index)

            p4 = new_particle(list_index,p)

            p = p4

            # print "after resample",[p[i].weight for i in range(len(p))]

        p_data = np.array([[i.x,i.y] for i in p])
        #print "p_data",p_data
        mean_p_data_x = np.mean(p_data.T[0])
        mean_p_data_y = np.mean(p_data.T[1])
        data_particles.append([mean_p_data_x,mean_p_data_y])
        landamarks_per_particle = np.array([])
        for i,y in enumerate(p):
            landamarks_per_particle = np.array([ [ l[1].T[0][0],l[1].T[0][1] ]  for l in y.landmarks])
            particle[i].setData(landamarks_per_particle)
        group_particles.setData(p_data)#np.array(data_particles))

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)



## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
