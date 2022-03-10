import numpy as np

def path1():
    cx = np.arange(0, 500, 1) + 50
    cy = [270 for ix in cx]
    cyaw = [0 for ix in cx]
    ccurv = [0 for ix in cx]
    path = np.array([(cx[i],cy[i],cyaw[i],ccurv[i]) for i in range(len(cx))])
    return path

def path2(p1 = 80.0):
    cx = np.arange(0, 500, 1) + 50
    cy = [np.sin(ix / p1) * ix / 4.0 + 270 for ix in cx]
    diff1 = [(np.cos(ix/p1)/p1*ix + np.sin(ix/p1))/4.0 for ix in cx]
    diff2 = [(-np.sin(ix/p1)/(p1**2)*ix + np.cos(ix/p1)/p1 + np.cos(ix/p1)/p1)/4.0 for ix in cx]
    cyaw = [np.rad2deg(np.arctan2(d1,1)) for d1 in diff1]
    ccurv = [np.abs(diff2[i])/np.power((1+diff1[i]**2),3/2) for i in range(len(cx))]
    path = np.array([(cx[i],cy[i],cyaw[i],ccurv[i]) for i in range(len(cx))])
    return path

def search_nearest(path, pos):
    min_dist = 99999999
    min_id = -1
    for i in range(path.shape[0]):
        dist = (pos[0] - path[i,0])**2 + (pos[1] - path[i,1])**2
        if dist < min_dist:
            min_dist = dist
            min_id = i
    return min_id, min_dist

def angle_norm(theta):
    return (theta + 180) % 360 - 180