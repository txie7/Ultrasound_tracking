import numpy as np
from lieGroup import log_map_SO3

def vel_control(Kw, Kv, R, p, wb,vb):
    # R is 3x3, p is 1x3
    # Kw and Kv are 3x3
    # wb and vb are 3x1
    wb = -np.matmul(Kw,np.transpose(log_map_SO3(R)))
    Kv_p = np.matmul(Kv, np.transpose(p))
    vb = -np.matmul(np.transpose(R),Kv_p)
    t = np.arccos((np.trace(R)-1)/2)
    
    return wb,vb

