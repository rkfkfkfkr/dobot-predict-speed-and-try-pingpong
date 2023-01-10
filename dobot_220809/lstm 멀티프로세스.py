from numpy import array
from keras.models import Sequential
from keras.layers import Dense,LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import cv2
import numpy as np
import imutils
import threading
import math
import time
import DobotDllType as dType
from numpy.linalg import inv

import cv2.aruco as aruco
import os

from multiprocessing import Process, Pipe, Queue, Value, Array, Lock

import numpy.random as rnd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


new_model = tf.keras.models.load_model('lstm2.h5')
api = dType.load()

def dobot_get_start():
    
    dType.SetQueuedCmdClear(api)

    dType.SetHOMEParams(api, 230, 0, 20, -23, -100) # x, y, z, r 

    dType.SetJOGJointParams(api, 900, 700, 700,480,300, 300, 300, 300, 0)
    dType.SetJOGCoordinateParams(api,900,900,900,480,300,300,300,300,0)
    dType.SetJOGCommonParams(api, 900,350,0)

    dType.SetPTPJointParams(api,320,320,320,480,300,300,300,300,0) # velocity[4], acceleration[4]
    dType.SetPTPCoordinateParams(api,700,700,480,300,0) 
    dType.SetPTPCommonParams(api, 500, 350,0) # velocityRatio(속도율), accelerationRation(가속율)
       
    dType.SetHOMECmd(api, temp = 0, isQueued = 0)

    #dType.SetQueuedCmdStartExec(api)

def dobot_connect():

    CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound", 
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:",CON_STR[state])

    if (state == dType.DobotConnect.DobotConnect_NoError):
        dobot_get_start()

def segmentaition(frame):

    img_ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(img_ycrcb)

    _, cb_th = cv2.threshold(cb, 90, 255, cv2.THRESH_BINARY_INV)
    cb_th = cv2.dilate(cv2.erode(cb_th, None, iterations=2), None, iterations=2)
    #cb_th = cv2.dilate(cb_th, None, iterations=2)

    return cb_th

def get_distance(x, y, imagePoints):
    
    objectPoints = np.array([[33.6,85,0],
                            [33.6,75,0],
                            [23.6,75,0],
                            [23.6,85,0],],dtype = 'float32')


    fx = float(470.5961)
    fy = float(418.18176)
    cx = float(275.7626)
    cy = float(240.41246)
    k1 = float(0.06950)
    k2 = float(-0.07445)
    p1 = float(-0.01089)
    p2 = float(-0.01516)

    #cameraMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype = 'float64')
    #distCoeffs = np.array([k1,k2,p1,p2],dtype = 'float64')

    cameraMatrix = np.array([[470.5961,0,275.7626],[0,418.18176,240.41246],[0,0,1]],dtype = 'float32')
    distCoeffs = np.array([0.06950,-0.07445,-0.01089,-0.01516],dtype = 'float32')
    _,rvec,t = cv2.solvePnP(objectPoints,imagePoints,cameraMatrix,distCoeffs)
    R,_ = cv2.Rodrigues(rvec)
            
    u = (x - cx) / fx
    v = (y - cy) / fy
    Qc = np.array([[u],[v],[1]])
    Cc = np.zeros((3,1))
    Rt = np.transpose(R)
    Qw = Rt.dot((Qc-t))
    Cw = Rt.dot((Cc-t))
    V = Qw - Cw
    k = -Cw[-1,0]/V[-1,0]
    Pw = Cw + k*V
    
    px = Pw[0]
    py = Pw[1]

    #print("px: %f, py: %f" %(px,py))

    return px,py

def find_ball(frame,cb_th,box_points):

    cnts = cv2.findContours(cb_th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    px = None
    py = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
            px,py = get_distance(center[0], center[1],box_points)
            
            #text = " %f , %f" %(px,py)
            #cv2.putText(frame,text,center,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            #print("px: %f, py: %f" %(px,py))

    return px,py

def findArucoMarkers(img, markerSize = 6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)

    if draw:
        cv2.aruco.drawDetectedMarkers(img,bboxs)
        #print(len(bboxs))

    if len(bboxs) > 0:
        return bboxs[0][0]
    else:
        return [0,0]

def move_dobot(of_x,of_y):
    
    offset_x = float(of_x * 10 - 80 - 60)#float((of_x + 6) * 10 - 5)
    offset_y = float(of_y * 10 - 1320 - 20) #float(of_y * 10 - 3)

    offset_x = round(offset_x,2)
    offset_y = round(offset_y,2)
    #last_index = 0
    
    length = math.sqrt(math.pow(offset_x,2) + math.pow(offset_y,2))

    print(length)

    if length > 80 and length < 270:

        print("offset_x: %f, offset_y: %f, length: %f \n" %(offset_x,offset_y,length))
        
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, offset_x, offset_y, -10, 0, isQueued = 1)[0]

def hit(of_x,of_y):

    offset_x = float(of_x * 10 - 80 - 60)#float((of_x + 6) * 10 - 5)
    offset_y = float(of_y * 10 - 1320 - 20) #float(of_y * 10 - 3)

    offset_x = round(offset_x,2)
    offset_y = round(offset_y,2)

    length = math.sqrt(math.pow(offset_x,2) + math.pow(offset_y,2))

    if length > 80 and length < 270:

        #dType.SetQueuedCmdStartExec(api)

        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, offset_x, offset_y, -23, -100, isQueued = 1)

        #dType.SetJOGCmd(api, 1, 1, 1)
        #dType.SetWAITCmd(api, 300, 1)
        #dType.SetJOGCmd(api, 1, 0, 1)

        dType.SetJOGCmd(api, 1, 2, 1)
        dType.SetWAITCmd(api, 300, 1)
        dType.SetJOGCmd(api, 1, 0, 1)

        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 200, 0, -23, -100, isQueued = 1)

        #dType.SetQueuedCmdStopExec(api)


def predict_time(ball_x,ball_y,time_list):

    vel = []
    vel.append(0)
    for i in range(len(ball_x)-1):

        v = math.sqrt((math.pow(ball_x[i+1] - ball_x[i],2) + math.pow(ball_y[i+1] - ball_y[i],2)))/(time_list[i+1] - time_list[i])
        vel.append(v)

    X = np.reshape(ball_y,(-1,1))
    y = np.reshape(vel,(-1,1))

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    lin_reg.intercept_, lin_reg.coef_
    X_new=np.linspace(float(ball_y[0]), float(ball_y[-1]), 100).reshape(100, 1)

    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)

    p_y = 132

    py = [[p_y, p_y**2]]
    pv = lin_reg.predict(py)
    predit_v = float(pv[0][0])

    return pv


def predict_xy(bx,by):

    input_x = []

    for j in range(5):

        x = float(bx[j])
        y = float(by[j])

        input_x.append([x,y])

    x_input = np.array(input_x)

    x_input = x_input.reshape((1,5,2))

    yhat = new_model.predict(x_input)

    predict_x = float(yhat[0][0])
    predict_y = float(yhat[0][1])

    return predict_x,predict_y
        
def predict_control(conn):

    cap = cv2.VideoCapture(0)

    ball_x = []
    ball_y = []

    p_x = []
    p_y = []
    
    hit_count = 0
    time_list = []

    a1 = 0
    a2 = 0

    while(1):    

        _,frame = cap.read()
        box_points = findArucoMarkers(frame)

        print(len(box_points))

        if len(box_points) > 2:
            break
        
    while(1):

        _,frame = cap.read()
        cb_th = segmentaition(frame)
        px,py = find_ball(frame,cb_th,box_points)

        conn.send([0])

        if px != None and py != None:

            ball_x.append(px)
            ball_y.append(py)

            t = time.time()
            time_list.append(t)

            #print("y: %f, t: %f " %(py,t))

            a1 = int(len(ball_x)/5)
            
            if a1 > a2 and ball_y[-1] < 132:

                predict_v = predict_time(ball_x,ball_y,time_list)
                conn.send([predict_v])
                predict_t = (132 - ball_y[-1])/predict_v
                #print("pt: %f sec, pv: %f cm/s" %(predict_t + time_list[-1], predict_v))

                bx = ball_x[((a1-1)*5):(a1*5)]
                by = ball_y[((a1-1)*5):(a1*5)]

                conn.send([bx,by,predict_t])

                a2 = a1

                '''                    
                of_x,of_y = predict_xy(bx,by)

                a2 = a1
                of_x = float(of_x) - 15
                of_y = float(of_y)
                '''
                
            
        elif px == None:

            conn.send([0,0])

            ball_x.clear()
            ball_y.clear()
            time_list.clear()
            p_x.clear()
            p_y.clear()

            a1 = 0
            a2 = 0

            
            
        cv2.imshow('cam',frame)
        
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    p_conn,c_conn = Pipe()
    hit_count = 0

    p1 = Process(target=predict_control, args=(c_conn,))
    p1.start()
    

    while(1):

        q = p_conn.recv()

        if len(q) > 2:

            bx = q[0]
            by = q[1]
            predict_t = float(q[2])

            of_x,of_y = predict_xy(bx,by)

            print(of_y)
            print(predict_t)

            if predict_t < 1.5 and predict_t > 0 and of_y > 60 and hit_count == 0:

                of_y = 132
                print('of_x: %f, of_y: %f' % (of_x,of_y))
                #hit(of_x,of_y)
                hit_count = 1

        if len(q) == 2:
            hit_count = 0



