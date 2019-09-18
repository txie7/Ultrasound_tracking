import numpy as np
from numpy import linalg as LA

def skew(x):
	# X should be a 1x3 vector
    return np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1], x[0],0]])

def exp_map_SO3(w):
    theta = LA.norm(w)
    if theta == 0:
        r = np.identity(3)
    else:
        n = w/theta
        r = np.identity(3) + np.sin(theta)*skew(n) + (1-np.cos(theta))*np.matmul(skew(n),skew(n))
    return r

def exp_map_SE3(w,v,theta):
    
    if w[0] == 0 and w[1]== 0 and w[2] == 0:
        R = np.identity(3)
        t = np.transpose(v)*theta
    else:
        R = exp_map_SO3(w*theta)
        t = np.matmul(np.matmul((np.identity(3) - R),skew(w)),v) + np.matmul(np.dot(w[:,np.newaxis],w[np.newaxis,:]),v*theta)
    return R,t

def log_map_SO3(R):
    theta = np.arccos((np.trace(R)-1)/2)
    if theta < 0.00000001:
        so3 = np.zeros([3,3])
    else:
        so3 =  theta/(2*np.sin(theta))*(R - np.transpose(R))
    
    wz = so3[1,0]
    wy = so3[0,2]
    wx = so3[2,1]
    w = np.array([wx,wy,wz])
       
    return w
	

def vb2vs(R,p,wb,vb):
    ws = np.matmul(R,wb)
    vs = np.matmul(R,vb) + np.matmul(np.matmul(skew(p),R),wb)
    return ws,vs

def R2rpy(R):
    a = np.arctan2(R[1,0],R[0,0])
    b = np.arctan2(-R[2,0],np.sqrt(R[2,1]*R[2,1]+R[2,2]*R[2,2]))
    c = np.arctan2(R[2,1],R[2,2])
    return a, b, c

def rpy2R(r,p,y):
    Rz = np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rx = np.array([[1,0,0],[0,np.cos(y),-np.sin(y)],[0,np.sin(y),np.cos(y)]])
    R = np.matmul(Rz,np.matmul(Ry,Rx))
    return R


