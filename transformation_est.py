import numpy as np
import cv2
from model import Model
from pts_cloud_reg import ptscloudreg

def find_target(window, target):
    # window: the searching window
    # target: target patch
    # patch: the matched patch in the window
    # top_left: the top_left corner of the patch in the window
    ###################################################################

    w,h = target.shape[::-1]
    res = cv2.matchTemplate(window,target,cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # the top_left corner of the center target patch
    top_left = [max_loc[1], max_loc[0]]
    patch = window[top_left[0]:top_left[0]+h, top_left[1]:top_left[1]+w]
            
    return patch,top_left

def surrounding_patches(window, temp_x, temp_y):
    #16 patches
    top_left_x = temp_x - 30 
    top_left_y = temp_y - 50 
    patches = np.zeros([16,60,100])
    patches_coordinate = np.zeros([16,2]) 
    cntr = 0
    for i in range(4):
        for j in range(4):
            x = top_left_x + 30*i
            y = top_left_y + 50*j
            patches[cntr,:,:] = window[x:x+60, y:y+100]
            patches_coordinate[cntr,0] = x + 30
            patches_coordinate[cntr,1] = y + 50
            cntr += 1          
    
    return patches, patches_coordinate

def cal_dz(template,image,model):
    # template and image should be size of [n,60,100]
    dim = len(template.shape)
        
    if dim == 2:
        image_pair = np.zeros([1,60,100,2])
        image_pair[0,:,:,0] = template
        image_pair[0,:,:,1] = image
    else:
        sz = template.shape[0]
        image_pair = np.zeros([sz,60,100,2])
        for i in range(sz):
            image_pair[i,:,:,0] = template[i,:,:]
            image_pair[i,:,:,1] = image[i,:,:]
            
    dz = model.model_pred(image_pair/255.)
    dz = dz
    return dz

def cal_signed_dz(window, target, ref, dist_t_f, model):
    # distance of target frame w.r.t. 
    # w.r.t. anatomy
    dist_t_c = cal_dz(target, window, model)
    dist_r_c = cal_dz(ref, window, model)
    sz = dist_t_c.shape[0]
            
    for i in range(sz):
        d_t_c = dist_t_c[i,0]
        d_r_c = dist_r_c[i,0]
        d_t_f = dist_t_f[i,0]
        if (max(d_t_c,d_t_f,d_r_c) == d_r_c ):
            dist_t_c[i,0] = -dist_t_c[i,0]
        else:
            dist_t_c[i,0] = dist_t_c[i,0]
          
    
    return dist_t_c

def six_DOF_transformation(pts_c,pts_a):
    return ptscloudreg(pts_c,pts_a)
