#! /usr/bin/env python3

import numpy as np
import logging
from functools import reduce
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

__all__ = [
    "readcube",
    "writecube",
    "writedat",
    "writevmd",
    "cv_silhouette_scorer",
    "pos_dens_grad_matrix",
    "min_distance_clusters",
    "max_distance_cluster",
    "warning_small_cluster",
]


def readcube(filename, verbose=True):
    """Reads in a cube file and returns grid information, atomic position, and a 3D array of cube values.

    Parameters
    ----------
    filename : str
        Cube file name.
    """
    if verbose:
        print("  Reading cube file: {}".format(filename))
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == 2:
                gridinfo1 = line
                n_at = int(gridinfo1.split()[0])
                o1, o2, o3 = (
                    float(gridinfo1.split()[1]),
                    float(gridinfo1.split()[2]),
                    float(gridinfo1.split()[3]),
                )
            elif i == 3:
                gridinfo2 = line
                npx = int(gridinfo2.split()[0])
                incrx = float(gridinfo2.split()[1])
            elif i == 4:
                gridinfo3 = line
                npy = int(gridinfo3.split()[0])
                incry = float(gridinfo3.split()[2])
            elif i == 5:
                gridinfo4 = line
                npz = int(gridinfo4.split()[0])
                incrz = float(gridinfo4.split()[3])
            elif i > 5:
                break

    pts = np.zeros((npx, npy, npz, 3))
    idx = np.indices((npx, npy, npz))
    pts[:, :, :, 0] = o1 + idx[0] * incrx
    pts[:, :, :, 1] = o2 + idx[1] * incry
    pts[:, :, :, 2] = o3 + idx[2] * incrz
    coordinates = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i in range(6, 6 + n_at):
                coord = line
                coordinates.append(
                    coord.split()[0]
                    + ","
                    + coord.split()[2]
                    + ","
                    + coord.split()[3]
                    + ","
                    + coord.split()[4]
                )
            elif i > (6 + n_at):
                break
    if len(coordinates) == n_at:
        pass
    else:
        raise ValueError("There is a problem with the coordinates of the cube file!")

    lines = open(filename).readlines()
    cubeval = []
    for i in lines[n_at + 6 :]:
        for j in i.split():
            cubeval.append(j)
    cube_shaped = np.reshape(cubeval, (npx, npy, npz))
    carray = cube_shaped.astype(np.float64)
    header = []
    with open(filename, "r") as g:
        for i, line in enumerate(g):
            if i in range(0, 6 + n_at):
                header.append(line)
    return header, pts, carray


def writecube(filename, cl, X_iso, labels, header, verbose=True):
    """ Write cube file for each cluster.
    
    Parameters
    ----------
    filename : str
         Common string in cube files name.
    X_iso : np.array
         Array with columns corresponding to space coordinates, sign(l2)*dens and rdg; for data with rdg equal to or below certain isovalue.
    X : np.array
         Array with columns corresponding to space coordinates, sign(l2)*dens and rdg; for all data.
    labels : np.array
         One dimensional array with integers that label the data in X_iso into different clusters.
    header : list of str
         Original cube file header.
    """
    n_clusters = len(set(labels))
    for i, line in enumerate(header):
        if i == 2:
            gridinfo1 = line
            n_at = int(gridinfo1.split()[0])
            o1, o2, o3 = (
                float(gridinfo1.split()[1]),
                float(gridinfo1.split()[2]),
                float(gridinfo1.split()[3]),
            )
        elif i == 3:
            gridinfo2 = line
            npx = int(gridinfo2.split()[0])
            incrx = float(gridinfo2.split()[1])
        elif i == 4:
            gridinfo3 = line
            npy = int(gridinfo3.split()[0])
            incry = float(gridinfo3.split()[2])
        elif i == 5:
            gridinfo4 = line
            npz = int(gridinfo4.split()[0])
            incrz = float(gridinfo4.split()[3])

    cl_idx = np.where(labels == cl)[0]
    X_iso = X_iso[cl_idx]

    # The extremes of the isosurface
    extr_x = [np.amin(X_iso[:, 0]), np.amax(X_iso[:, 0])]
    extr_y = [np.amin(X_iso[:, 1]), np.amax(X_iso[:, 1])]
    extr_z = [np.amin(X_iso[:, 2]), np.amax(X_iso[:, 2])]

    # We will take 5 more pts around extrema
    delta_x = [0, 0]
    delta_y = [0, 0]
    delta_z = [0, 0]
    
    # The original end
    e1 = o1 + (npx-1) * incrx
    # Check if o1, e1 can be changed
    if o1 < extr_x[0]-5*incrx:
        delta_x[0]= 5*incrx
        o1 = extr_x[0] - delta_x[0]
    if extr_x[1]+5*incrx < e1:
        delta_x[1]= 5*incrx

    e2 = o2 + (npy-1) * incry
    if o2 < extr_y[0]-5*incry:
        delta_y[0]= 5*incry
        o2 = extr_y[0] - delta_y[0]
    if extr_y[1]+5*incry < e2:
        delta_y[1]= 5*incry
        
    e3 = o3 + (npz-1) * incrz
    if o3 < extr_z[0]-5*incrz:
        delta_z[0]= 5*incrz
        o3 = extr_z[0] - delta_z[0]
    if extr_z[1]+5*incrz < e3:
        delta_z[1] = 5*incrz

    npx = int(np.round((extr_x[1] - extr_x[0] + delta_x[0] + delta_x[1])/incrx)) + 1
    npy = int(np.round((extr_y[1] - extr_y[0] + delta_y[0] + delta_y[1])/incry)) + 1
    npz = int(np.round((extr_z[1] - extr_z[0] + delta_z[0] + delta_z[1])/incrz)) + 1

    pts = np.zeros((npx, npy, npz, 3))
    pts_idx = np.indices((npx, npy, npz))
    pts[:, :, :, 0] = o1 + pts_idx[0] * incrx
    pts[:, :, :, 1] = o2 + pts_idx[1] * incry
    pts[:, :, :, 2] = o3 + pts_idx[2] * incrz

    # We rebuild grid to find missing points of X_iso
    pts = pts.reshape((npx * npy * npz, 3))
    X_iso_pts = np.copy(X_iso[:, :3])
    iso_idx = np.zeros(len(X_iso))
    for k, pt in enumerate(X_iso_pts):
        npwh_x = np.where(np.absolute(pts[:, 0] - pt[0])<1e-8)[0]
        npwh_y = np.where(np.absolute(pts[:, 1] - pt[1])<1e-8)[0]
        npwh_z = np.where(np.absolute(pts[:, 2] - pt[2])<1e-8)[0]
        iso_idx[k]= reduce(np.intersect1d, (npwh_x, npwh_y, npwh_z)) # where we find pt from X_iso_pts in pts

    iso_idx = iso_idx.astype("int32")

    # This will be a matrix with only significant values in <= isovalue spots
    grad_iso_extended = np.full((len(pts), 1), 1000.)
    dens_iso_extended = np.full((len(pts), 1), 1000.)
    for i, idx in enumerate(iso_idx):
        grad_iso_extended[idx] = X_iso[i, 4]
        dens_iso_extended[idx] = X_iso[i, 3]

    grad_cube = grad_iso_extended.reshape(npx, npy, npz)
    dens_cube = dens_iso_extended.reshape(npx, npy, npz)
    if verbose:
        print(
            "  Writing cube file {}...      ".format(filename + "-cl" + str(cl) + "-grad.cube"),
            end="",
            flush=True,
        )
    with open(filename + "-cl" + str(cl) + "-grad.cube", "w") as f_out:
        f_out.write(" cl" + str(cl) + "_grad_cube\n")
        f_out.write(" 3d plot, gradient\n")
        f_out.write("   1  {:.6f}  {:.6f}  {:.6f}\n".format(o1, o2, o3))
        f_out.write("   {}  {:.6f}  {:.6f}  {:.6f}\n".format(npx, incrx, 0, 0))
        f_out.write("   {}  {:.6f}  {:.6f}  {:.6f}\n".format(npy, 0, incry, 0))
        f_out.write("   {}  {:.6f}  {:.6f}  {:.6f}\n".format(npz, 0, 0, incrz))
        f_out.write("   0   0.0  {:.6f}  {:.6f}  {:.6f}\n".format(o1, o2, o3))
        
    with open(filename + "-cl" + str(cl) + "-grad.cube", "a") as f_out:
        for i in range(0, npx):
            for j in range(0, npy):
                for k in range(0, npz):
                    f_out.write("{:15.5E}".format(grad_cube[i][j][k]))
                    f_out.write("\n")
    if verbose:
        print("done")

        print(
            "  Writing cube file {}...      ".format(filename + "-cl" + str(cl) + "-dens.cube"),
            end="",
            flush=True,
        )
    with open(filename + "-cl" + str(cl) + "-dens.cube", "w") as f_out:
        f_out.write(" cl" + str(cl) + "_dens_cube\n")
        f_out.write(" 3d plot, density\n")
        f_out.write("   1  {:.6f}  {:.6f}  {:.6f}\n".format(o1, o2, o3))
        f_out.write("   {}  {:.6f}  {:.6f}  {:.6f}\n".format(npx, incrx, 0, 0))
        f_out.write("   {}  {:.6f}  {:.6f}  {:.6f}\n".format(npy, 0, incry, 0))
        f_out.write("   {}  {:.6f}  {:.6f}  {:.6f}\n".format(npz, 0, 0, incrz))
        f_out.write("   0   0.0  {:.6f}  {:.6f}  {:.6f}\n".format(o1, o2, o3))
        
    with open(filename + "-cl" + str(cl) + "-dens.cube", "a") as f_out:
        for i in range(0, npx):
            for j in range(0, npy):
                for k in range(0, npz):
                    f_out.write("{:15.5E}".format(dens_cube[i][j][k]))
                    f_out.write("\n")

    if verbose:
        print("done")


def writedat(filename, cl, X_iso, labels, verbose=True):
    if verbose:
        print("  Writing dat file {}...                 ".format(filename + ".dat"), end="", flush=True)
    cl_idx = np.where(labels == cl)[0]
    X_iso = X_iso[cl_idx]
    X_iso_factor = np.zeros((len(X_iso),2))
    X_iso_factor[:, 0] = 0.01 * X_iso[:, 3]
    X_iso_factor[:, 1] =  X_iso[:, 4]
    np.savetxt(filename+"-cl"+str(cl)+".dat", X_iso_factor)
    if verbose:
        print("done")

    
def writevmd(filename, labels, isovalue, verbose=True):
    """ Write vmd script file for each cluster.
    
    Parameters
    ----------
    filename : str
         Common string in cube files name.
    labels : np.array
         One dimensional array with integers that label the data in X_iso into different clusters.
    """

    file_dens = filename + "-dens.cube"
    if verbose:
        print("  Writing vmd file {}...                 ".format(filename + ".vmd"), end="", flush=True)
    with open(filename + ".vmd", "w") as f:
        f.write("#!/usr/local/bin/vmd \n")
        f.write("# Display settings \n")
        f.write("display projection   Orthographic \n")
        f.write("display nearclip set 0.000000 \n")

    with open(filename + ".vmd", "a") as f:
        f.write("# load new molecule \n")
        f.write(
            "mol new "
            + filename
            + "-dens.cube type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all \n"
        )
        f.write("# \n")
        f.write("# representation of the atoms \n")
        f.write("mol delrep 0 top \n")
        # f.write("mol representation Lines 1.00000 \n")
        # f.write("mol color Name \n")
        # f.write("mol selection {all} \n")
        # f.write("mol material Opaque \n")
        # f.write("mol addrep top \n")
        f.write("mol representation CPK 1.000000 0.300000 118.000000 131.000000 \n")
        f.write("mol color Name \n")
        f.write("mol selection {all} \n")
        f.write("mol material Opaque \n")
        f.write("mol addrep top \n")


    for cl in set(labels):
        with open(filename + ".vmd", "a") as f:
            f.write("# load new molecule \n")
            f.write(
                "mol new "
                + filename
                + "-cl"
                + str(cl)
                + "-dens.cube type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all \n"
            )
            f.write(
                "mol addfile "
                + filename
                + "-cl"
                + str(cl)
                + "-grad.cube type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all \n"
            )
            f.write("# \n")
            f.write("# add representation of the surface \n")
            f.write("mol representation Isosurface {:.5f} 1 0 0 1 1 \n".format(isovalue))
            f.write("mol color Volume 0 \n")
            f.write("mol selection {all} \n")
            f.write("mol material Opaque \n")
            f.write("mol addrep top \n")
            f.write("mol selupdate 2 top 0 \n")
            f.write("mol colupdate 2 top 0 \n")
            f.write("mol scaleminmax top 1 -7.0000 7.0000 \n")
            f.write("mol smoothrep top 2 0 \n")
            f.write("mol drawframes top 2 {now} \n")
            f.write("color scale method BGR \n")
            f.write("set colorcmds {{color Name {C} gray}} \n")
    if verbose:
        print("done")
    

def pos_dens_grad_matrix(filename, save_to_csv=False):
    """ Build an array with of size (n_pts, 5), where n_pts is the number of gridpoints. Each row contains spatial coordinates and the values of the sign(l2)*dens and rdg for each gridpoint.
    
    Parameters
    ----------
    filename : str
        Common string in cube files name.
    save_to_csv : boolean, optional
        If True, array is saved into csv file filename.csv.
    """
    densheader, denspts, densarray = readcube(filename + "-dens.cube")
    gradheader, gradpts, gradarray = readcube(filename + "-grad.cube")

    if densarray.shape != gradarray.shape:
        raise ValueError("Files do not match!")

    if not np.allclose(denspts, gradpts):
        raise ValueError("Points in file do not match!")

    nx, ny, nz = densarray.shape
    final = np.zeros((nx, ny, nz, 5))
    final[:, :, :, :3] = denspts
    final[:, :, :, 3] = densarray
    final[:, :, :, 4] = gradarray

    if save_to_csv:
        np.savetxt(filename + ".csv", final.reshape(densarray.size, 5), delimiter=", ")
    return densheader, final.reshape(densarray.size, 5)


def cv_silhouette_scorer(estimator, X):
    """ Score how succesful the assignment of X points to clusters has been, using estimator.
    
    Parameters
    ----------
    estimator : instance of KMeans
             Clustering estimator.
    X : numpy array
             Array with columns corresponding to space coordinates, sign(l2)*dens and rdg.
    """
    estimator.fit(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(X)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return silhouette_score(X, cluster_labels)


def min_distance_clusters(clusters, warning_val=0.5):
    """Get minimum distance between clusters.

    Parameters
    ----------
    clusters : np.array
        Array of arrays that contain elements of each cluster.
    warning_val: float, optional
        If minimun distance is less or equal to it, gives a warning.
    """
    # min_dist = []
    min_dist = np.zeros((len(clusters), len(clusters)))
    for k, cl1 in enumerate(clusters):
        for l in range(k):
            cl2 = clusters[l]
            min12 = np.amin(euclidean_distances(cl1, cl2))
            min_dist[k, l] = min12
            if warning_val is not None:
                if min12 < warning_val:
                    logging.warning(
                        UserWarning(
                            "Minimun distance between clusters {} and {} is very small: {} A".format(
                                k, l, min12
                            )
                        )
                    )
    return min_dist


def max_distance_cluster(clusters, warning_val=3.0):
    """Get maximum distance between elements inside a cluster.

    Parameters
    ----------
    clusters : np.array
        Array of arrays that contain elements of each cluster.
    warning_val: float, optional
        If maximum distance is greater or equal to it, gives a warning.
    """
    for k, cl in enumerate(clusters):
        max1 = np.amax(euclidean_distances(cl, cl))
        if max1 > warning_val:
            logging.warning(
                UserWarning("Maximum distance in cluster {} is very big: {} A".format(k, max1))
            )


def warning_small_cluster(clusters, portion=0.1, size=None):
    """Give a warning if there is a very small cluster.

    Parameters
    ----------
    clusters : np.array
        Array of arrays that contain elements of each cluster.
    portion: float or None, optional
        If None, size value must be given. Else, gives a warning if a cluster is smaller than portion times the biggest. 
    size: int or None, optional
        If None, portion value must be given. Else, gives a warning if a cluster is smaller than size. 
    """
    cls_len = [len(cl) for cl in clusters]
    len_max_cluster = max(cls_len)
    if size is None:
        if portion is None:
            raise ValueError("Either portion or size must be different than None")
        size = int(portion * len_max_cluster)
    for k, cl in enumerate(clusters):
        if len(cl) <= size:
            logging.warning(
                UserWarning("Cluster {} is very small, having only {} elements".format(k, len(cl)))
            )
