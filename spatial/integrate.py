#! /usr/bin/env python3

import numpy as np

def integrate_density_cl_cube(cl, X_iso, labels, incr):
    cl_idx = np.where(labels == cl)[0]
    density = np.absolute(X_iso[cl_idx, 3])
    
    dvol = incr[0]*incr[1]*incr[2]
    idx_dens = np.where(density<1000)[0]
    return np.sum(density[idx_dens]*dvol)
