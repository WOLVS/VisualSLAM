import random
import numpy as np
class robot:
    def __init__(self):

        #### Kalman Filter Param #######
        self.l_mean = np.array([[0.,0.]]) # initial mean landmark
        self.P = np.identity(2)*0.0 # covariance landmark
        self.weight = 0 # default particle weigth

        ##### World #####
        self.world_size = 100.0
        self.measurement_range = 40.0
        self.num_landmarks = 0
        self.world_landmarks = []
        self.angle_vision = np.pi/4
        self.landmarks = []
        self.world_param = {}
        self.seen = []

        #### Robot ####
        self.x = random.uniform(-1,1) * self.world_size
        self.y = random.uniform(-1,1) * self.world_size
        self.orientation = random.random() * 2.0 * np.pi
        self.forward_noise = 0.0;
        self.turn_noise    = 0.0;

        self.distance_noise = 0.0;
        self.angle_measure_noise = 0.0;

    # make random landmarks located in the world
    #
    def set_world(self,world_size,measurement_range,num_landmarks,angle_vision,world_landmarks ):
        self.world_size = world_size
        self.measurement_range = measurement_range
        self.angle_vision = angle_vision
        self.num_landmarks = num_landmarks
        self.world_landmarks = world_landmarks
        self.world_param = dict(world_size=world_size,measurement_range=measurement_range,
                                angle_vision=angle_vision,num_landmarks=num_landmarks
                                ,world_landmarks=world_landmarks)

    def make_landmarks(self, num_landmarks):
        self.landmarks = []
        for i in range(num_landmarks):
            self.landmarks.append([round(random.random() * self.world_size),
                                   round(random.random() * self.world_size)])
        self.num_landmarks = num_landmarks

    def set(self, x, y, orientation):

        # if orientation < 0 or orientation >= 2 * np.pi:
        #     raise ValueError, 'Orientation must be in [0..2pi]'

        self.x = float(x)
        self.y = float(y)
        self.orientation = self.norm_ang(float(orientation))


    def set_noise(self, new_f_noise, new_t_noise,new_d_noise,\
                  new_a_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.forward_noise = float(new_f_noise);
        self.turn_noise    = float(new_t_noise);

        self.distance_noise = float(new_d_noise);
        self.angle_measure_noise = float(new_a_noise);

    def set_landmarks(self,param):
        self.landmarks = param
    def set_weight(self,weight):
        self.weight = weight
    def get_weight(self):
        return self.weight
    def init_seen(self,seen):
        self.seen = [False for row in range(seen)]
    def set_seen(self,seen):
        self.seen = seen

    def sense(self):

        Z = []
        # param = self.landmarks

        for i in range(len(self.world_landmarks)):
            ######define range of measurement (angle, range)
            alpha = (self.orientation - self.angle_vision)
            beta = (self.orientation + self.angle_vision)
            ## Vector to form a triangle for sensor range measurement
            v1 = [self.x+self.measurement_range*np.math.cos(alpha),\
              self.y + self.measurement_range*np.math.sin(alpha)]
            v2 = [self.x+self.measurement_range*np.math.cos(beta),\
                  self.y +self.measurement_range*np.math.sin(beta)]
            v3 = [self.x,self.y]
            #Vector to Landmark
            v_to_l = [(self.world_landmarks[i][0] - self.x),(self.world_landmarks[i][1] - self.y)]

            landmark_angle = np.math.atan2(v_to_l[1],v_to_l[0]) # measure angle
            landmark_angle += random.gauss(0,self.angle_measure_noise)


            landmark_dist = np.linalg.norm(v_to_l) # measure distance
            landmark_dist += random.gauss(0,self.distance_noise)

            #real distance from robot to landmark
            d = np.linalg.norm(v_to_l)
            # check if measurement in range
            is_in = self.check_if_point_in_range([self.world_landmarks[i][0],self.world_landmarks[i][1]],v1,v2,v3)

            if (d <= self.measurement_range) and \
               ( is_in ):
                # #Jacobian Measurement function
                # q = landmark_dist**2 # = q = r**2
                # G = self.Jacobian_h(v_to_l,landmark_dist)

                z_real = np.array([[landmark_dist],[landmark_angle]])

                Z.append([i,z_real])

                # param.append([i,self.l_mean,self.P,self.weight])
                # self.set_landmarks(param)

        return Z


    def update_measurement(self,Z):

        Qt = np.array([[self.distance_noise**2,0],\
                               [0,self.angle_measure_noise**2]])

        param = self.landmarks

        for i,Z_ in enumerate(Z):

            z = Z_[1]
            ### from measurement get mean landmark, u,v
            r = z[0][0]
            theta = z[1][0]
            u = r*np.math.cos(theta)
            v = r*np.math.sin(theta)

            #G = Z_[2]
            #dist = z[0][0] # Z_dist
            #theta = z[1][0] # angle rotation for particle
            dx = self.x + u # landmark_particle_x
            dy = self.y + v # landmark_particle_y
            v_to_l = [(dx-self.x),(dy-self.y)]
            landmark_pos = [dx,dy]

            if not(self.seen[Z_[0]]): # new feature detected
                ## initialize mean and covariance of landmark
                self.l_mean = np.array([[dx],[dy]])
                H = self.Jacobian_h(v_to_l,r) #Jacobian inverse observation model

                self.P = np.linalg.inv(H).dot(Qt).dot(np.linalg.inv(H).T)

                param.append([Z_[0],self.l_mean,self.P,self.weight])
                self.set_landmarks(param)
                self.seen[Z_[0]] = True

            else: # observed feature

                #update mean and cov measurement

                # Get data association
                index = 0
                for idl,l in enumerate(self.landmarks):
                    if l[0] == Z_[0]:
                       index = idl
                #get previous values of landmark
                idx_,l_mean,P_pred,w = self.landmarks[index] # mu_t, simga_t
                v_to_l_pred = [(l_mean[0][0]- self.x), (l_mean[1][0] - self.y)]

                landmark_angle_pred = np.math.atan2(v_to_l_pred[1],v_to_l_pred[0]) # measure angle

                landmark_dist_pred = np.linalg.norm(v_to_l_pred) # measure distance
                z_pred = np.array([[landmark_dist_pred],[landmark_angle_pred]])

                G = self.Jacobian_h(v_to_l_pred,landmark_dist_pred) # Jacobian with respect to landmark
                residual = z - z_pred
                residual[1][0] = self.norm_ang(residual[1][0])
                #Update Step Kalman Filter
                new_mean,new_sigma = self.EKFUpdate( l_mean,\
                                      residual,\
                                        P_pred,Qt,G)
                self.l_mean = new_mean
                self.P = new_sigma
                # compute weight
                self.weight = self.compute_weight(G,P_pred,Qt,residual)

                for idx,n in enumerate([Z_[0],self.l_mean,self.P,self.weight]):
                    self.landmarks[index][idx] = n

    def move(self, turn, forward, particle=False):
        if forward < 0:
            raise ValueError, 'Robot cant move backwards'

        # turn, and add randomness to the turning command
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation = self.norm_ang(orientation)


        # move, and add randomness to the motion command
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + (np.math.cos(orientation) * dist)
        y = self.y + (np.math.sin(orientation) * dist)

        # set particle
        self.set(x,y,orientation)
        # res = robot()
        # res.set(x, y, orientation)
        # res.set_noise(self.forward_noise, self.turn_noise,\
        #               self.distance_noise,self.angle_measure_noise)
        # res.set_weight(self.weight)
        # res.set_seen(self.seen)
        # if self.landmarks:
        #     res.set_landmarks(self.landmarks)
        # if self.world_param:
        #     res.set_world(**self.world_param)
        # return res
    def Jacobian_h(self,v_to_l,r):

        return np.array([ [(v_to_l[0])/r,(v_to_l[1])/r ],\
                                [-(v_to_l[1])/r**2, (v_to_l[0])/r**2]])
    def EKFUpdate(self,landmark_mean,residual,P_pred,Qt,G):

        # residual = z - z_pred

        Zt = G.dot(P_pred).dot(G.T) + Qt

        K = P_pred.dot(G.T).dot(np.linalg.inv(Zt))
        I = np.identity(len(K))
        P = np.dot((I - np.dot(K,G)),P_pred) # innovation Covariance

        landmark_mean = landmark_mean + np.dot(K,residual)

        return landmark_mean, P
    def compute_weight(self,G,P_pred,Qt,residual):

        ### Weights FastSLAM 2.0 ###
        Zt = G.dot(P_pred).dot(G.T)+Qt
        id_exp = ( ((residual.T)).dot(np.linalg.inv(Zt)).dot(residual))
        normalizer = np.math.sqrt(np.linalg.det(2*np.math.pi*Zt))

        if id_exp > 500:
            print "Overflow"

        w = self.weight
        w *= (1.0/normalizer)*np.math.exp(-.5*id_exp)
        return w
    def sign(self,p1,p2,p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    def check_if_point_in_range(self,pt,v1,v2,v3):
        b1 = self.sign(pt, v1, v2) < 0.0;
        b2 = self.sign(pt, v2, v3) < 0.0;
        b3 = self.sign(pt, v3, v1) < 0.0;
        return ((b1 == b2) and (b2 == b3))
    def norm_ang(self,y):
        if (y > np.pi):
            y -= 2 * np.pi
        elif(y < -np.pi):
            y += 2*np.pi
        return y
    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))
