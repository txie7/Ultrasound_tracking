import numpy as np
from numpy import linalg as LA

PI = 3.1415926536

def skew(x):
	# X should be a 1x3 vector
	return np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1], x[0],0]])

def expcr(x):
	# x should be a 1x3 vector
	theta = LA.norm(x)
	if theta == 0:
		r = np.identity(3)
	else:
		n = x/theta
		r = np.identity(3) + np.sin(theta)*skew(n) + (1-np.cos(theta))*np.matmul(skew(n),skew(n))
	return r

def quaternion2r(q):
	# q should be a unit quaternion
	s = q[0]
	v = q[1:]
	
	# Consider three cases:
	# 1. No rotation:
	if LA.norm(v) == 0:
		R = np.identity(3)
	elif s == 0:
		theta = PI
		n = v/LA.norm(v)
		R = expcr(theta*n)
	else:
		theta = 2*np.arctan2(LA.norm(v),s)
		n = v/LA.norm(v)
		R = expcr(theta*n)
	return R

def ptscloudreg(a,b):
	# a and b are Nx3 matrices
	num = a.shape[0]
	a_avg = np.mean(a, axis = 0)
	b_avg = np.mean(b, axis = 0)
	a_t = a - a_avg
	b_t = b - b_avg

	M = np.zeros([num*4,4])
	for i in range(num):
		m = np.zeros([4,4])
		m[0,1:] = b_t[i,:] - a_t[i,:]
		m[1:,0] = np.transpose(b_t[i,:] - a_t[i,:])
		m[1:,1:] = skew(b_t[i,:] + a_t[i,:])
		M[i*4:i*4+4,:] = m


	U,S,VH = LA.svd(M)
	V = np.transpose(VH)
	q = V[:,3]
	R = quaternion2r(q)
	p = b_avg - np.matmul(R,a_avg) 
	return R, p
