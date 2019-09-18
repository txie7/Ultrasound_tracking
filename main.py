import random
import numpy as np
from numpy import linalg as LA
import cv2
import csv
import os
from lieGroup import exp_map_SO3,exp_map_SE3, log_map_SO3,vb2vs,skew, rpy2R, R2rpy
from transformation_est import find_target, surrounding_patches, cal_dz, cal_signed_dz, six_DOF_transformation
from velocity_control import vel_control
from model import Model

######################## Simulation #####################################
def load_volume(img_dir, start_num, end_num):
    win_x = 150
    win_y = 250
    win_w = 500
    win_h = 400
    N = end_num-start_num+1
    V = np.zeros([N,400,500])
    for i in range(N):
        img_name = img_dir + str(start_num + i) + '.jpg'
        img = cv2.imread(img_name,0)
        img = img[win_x:win_x+win_h, win_y:win_y+win_w]
        V[i,:,:] = img
        
    return V

def a_position(t, amps, phases, start_phase,r1,r2):
    # Calculate the current tumor position in mm
    PI = np.pi
    a_x = amps[0]/2 - amps[0]*np.cos(PI*t/5 + phases[0] + start_phase)*np.cos(PI*t/5 + phases[0] + start_phase)
    a_y = amps[1]/2 - amps[1]*np.cos(PI*t/5 + phases[1] + start_phase)*np.cos(PI*t/5 + phases[1] + start_phase)
    a_z = amps[2]/2 - amps[2]*np.cos(PI*t/5 + phases[2] + start_phase)*np.cos(PI*t/5 + phases[2] + start_phase)
    
    a_x0 = amps[0]/2 - amps[0]*np.cos(phases[0] + start_phase)*np.cos(phases[0] + start_phase)
    a_y0 = amps[1]/2 - amps[1]*np.cos(phases[1] + start_phase)*np.cos(phases[1] + start_phase)
    a_z0 = amps[2]/2 - amps[2]*np.cos(phases[2] + start_phase)*np.cos(phases[2] + start_phase)
    
    a_x = a_x - a_x0
    a_y = a_y - a_y0
    a_z = a_z - a_z0
    
    r_x = r1*np.sin(PI*t/5)*np.sin(PI*t/5)
    r_y = r2*np.sin(PI*t/5)*np.sin(PI*t/5)
    Rx = np.array([[1,0,0],[0, np.cos(r_x), -np.sin(r_x)],[0, np.sin(r_x), np.cos(r_x)]])
    Ry = np.array([[np.cos(r_y),0,np.sin(r_y)],[0,1,0],[-np.sin(r_y),0,np.cos(r_y)]])
    R = np.matmul(Ry,Rx)
    
    return np.array([a_x, a_y, a_z]), R

def update_display(V, win_info, a_t, a_R, p_t, p_R):
    dt = p_t - a_t
    x = np.around(dt[0]/0.1) + win_info['win_x']
    y = np.around(dt[1]/0.1) + win_info['win_y']
    z = np.around(dt[2]/0.0565) + win_info['img0']
    frame = np.array([x,y,z]).astype(int)
    
    pix_p = np.zeros([3,win_info['win_h']*win_info['win_w']])
    cntr = 0
    for i in range(win_info['win_h']):
        for j in range(win_info['win_w']):
            pix_p[0,cntr] = i
            pix_p[1,cntr] = j
            cntr += 1
    
    dR = np.matmul(np.transpose(a_R),p_R)
    frame = np.resize(frame, (3,1))
    pix_w = frame + np.around(np.matmul(dR,pix_p)).astype(int)
    
    window = np.zeros([win_info['win_h'],win_info['win_w']])
    cntr = 0
    for i in range(win_info['win_h']):
        for j in range(win_info['win_w']):
            p = pix_w[0,cntr]
            q = pix_w[1,cntr]
            r = pix_w[2,cntr]
            window[i,j] = V[r,p,q]
            cntr += 1
    window = window.astype(np.uint8)    
    return window, dt, dR, frame
#######################################################################

def cal_error(R_target, t_target, R, t):
    e_R = np.matmul(np.transpose(R_target),R)
    e_t = np.matmul(np.transpose(R_target),t) -  np.matmul(np.transpose(R_target),t_target)
    return e_R, e_t
    

## Initialization:
delta_time = 0.1
# Generate the tumor motion: call the a_position function to get current position
amps = np.array([4,11,6])
phases = np.array([0.15,0.22,0.05]) 
#phases = np.array([random.random()*0.5,random.random()*0.5,random.random()*0.5])
start_phase = 0 
#start_phase = random.random()*0.5
r_x = 0.174
r_y = 0.105


V = load_volume('./volume/',40,220)
model = Model()


# T = 0
time = 0
probe_pos = np.array([0,0,0])
a_pos,Rx = a_position(time,amps,phases,start_phase,r_x,r_y)                 # Start with 000

angle = []
d = []
a_angle = []

img0 = 58 #The center plane in the volume
# Choose the target frame and ROI:
img = V[img0,:,:]

win_x = 75
win_y = 75
win_w = 350
win_h = 250
window = img[win_x: win_x + win_h, win_y:win_y + win_w]

cv2.imwrite('targetwindow.jpg',window)
win_info = {'img0':img0, 'win_x':win_x,'win_y':win_y,'win_h':win_h, 'win_w':win_w}
#print(win_info)

temp_x = 90
temp_y = 95
temp_w = 120
temp_h = 100
target = window[temp_x:temp_x + temp_h, temp_y:temp_y+temp_w].astype(np.uint8)


## Generate 16 patches
pat_target,pat_target_coor = surrounding_patches(window, temp_x, temp_y)

R_ap_target = np.identity(3)
t_ap_target = - np.array([0.1*pat_target_coor[0,0],0.1*pat_target_coor[0,1],0])
R_fp = np.identity(3)
t_fp = - np.array([0.1*pat_target_coor[0,0],0.1*pat_target_coor[0,1],0])
R_af = np.identity(3)
t_af = np.array([0,0,0])

pts_a_xy = 0.1*(pat_target_coor - pat_target_coor[0])
pts_a = np.concatenate((pts_a_xy,np.zeros([16,1])),axis = 1)
pts_c = pts_a




time = 0.1
p_t = np.array([0,0,0.4])
p_R = np.identity(3)
a_t,a_R = a_position(time,amps,phases,start_phase,r_x,r_y)
window, dt, dR,frame = update_display(V, win_info, a_t, a_R, p_t, p_R)
#print('dt:',dt,'frame:', frame)
d.append(dt)
angle.append(R2rpy(dR))

# Find the target in the 2nd frame
p,top_left = find_target(window, target)
print(top_left)
pat_ref, pat_ref_coor = surrounding_patches(window,top_left[0],top_left[1])
dist_t_r = cal_dz(pat_target,pat_ref,model)

time = 0.2
p_t = p_t
p_R = p_R
a_t,a_R = a_position(time,amps,phases,start_phase,r_x,r_y)
window, dt, dR,frame = update_display(V, win_info, a_t, a_R, p_t, p_R)
#print('dt:',dt,'frame:',frame,'dR:', dR)
d.append(dt)
angle.append(R2rpy(dR))

rs = np.array([0,0,0])
wb = 0
vb = 0
vs_prev = 0
prev_a =0
prev_b= 0
prev_c = 0
while(time < 3):
    patch, top_left = find_target(window, target)
    pat_c, pat_c_coor = surrounding_patches(window,top_left[0],top_left[1])
    
    # frame w.r.t. probe
    R_fp = np.identity(3)
    t_fp = - np.array([0.1*pat_c_coor[0,0],0.1*pat_c_coor[0,1],0])
    
    
    dz_a = cal_signed_dz(pat_c,pat_target, pat_ref,dist_t_r,model)
    pts_c = np.concatenate((pts_a_xy,-dz_a),axis = 1)


    # Calculate the natural error
    R_af, t_af = six_DOF_transformation(pts_c,pts_a) # transformation of g w.r.t g_a
    x = R2rpy(R_af)
    delta_R = np.array([[np.cos(x[0]), -np.sin(x[0]),0],[np.sin(x[0]), np.cos(x[0]),0],[0,0,1]])
    R_af = np.matmul(np.transpose(delta_R),R_af)
    
    R_ap = np.matmul(R_af,R_fp)
    t_ap = np.matmul(R_af,t_fp)+t_af
    e_R, e_t = cal_error(R_ap_target,t_ap_target,R_ap,t_ap)

    Kw = np.array([[1,0,0],[0,1,0],[0,0,1]])
    Kv = np.array([[9,0,0],[0,12,0],[0,0,8]])
    wb,vb = vel_control(Kw, Kv, e_R, e_t,wb,vb)
    
    ws = np.matmul(p_R,wb)
    d_p_R = exp_map_SO3(ws*0.1)
    a,b,c = R2rpy(d_p_R)
    # control r,p y         
    a = 1*a
    b = 10*b
    c = 10*c
    d_p_R = rpy2R(a,b,c)
    dws = log_map_SO3(d_p_R)
    prev_a = a
    prev_b = b
    prev_c = c
    vs = np.matmul(p_R,vb) + np.matmul(np.matmul(skew(p_t),p_R),wb)
    
    rs = rs + dws
    p_R = exp_map_SO3(rs)
    p_t = vs*0.1 + p_t
    vs_prev = vs
    time = time + 0.1
    a_t,a_R = a_position(time,amps,phases,start_phase,r_x,r_y)
    window, dt, dR,frame = update_display(V, win_info, a_t, a_R, p_t, p_R)
    
    d.append(dt)
    print('t:',dt)

    angle.append(R2rpy(dR))
    print(R2rpy(dR))
    

with open('tracking_d1.csv','w',newline='') as f:
    writer = csv.writer(f)
    for i in range(len(d)):
        writer.writerow(d[i])
with open('tracking_a1.csv','w',newline='') as f:
    writer = csv.writer(f)
    for i in range(len(angle)):
        t = angle[i]
        writer.writerow(t)













