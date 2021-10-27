#! /usr/bin/env python3

import time
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from spatial.utils import *
from spatial.divide import *

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plot_2d(X_iso, labels, filename, X=None, verbose=True):
    """ Plot s vs rho with different colors for data corresponding to each nci region.
    
    Parameters
    ----------
    X_iso : np.array
       Array with columns corresponding to space coordinates, sign(l2)*dens and rdg; for data with rdg equal to or below isovalue.
    labels : np.array
       One dimensional array with integers that label the data in X_iso into different clusters.
    filename: str
       Filename for plot figure, without extension.
    X : np.array or None, optional
       If not None, there are two plots: one with all the data and one only with the data below the isovalue. If None, only the latter is saved.
    """

    fig_2d = plt.figure()
    #colormap = plt.cm.get_cmap("Accent")
    N = len(np.unique(labels))
    colormap = discrete_cmap(N, base_cmap='rainbow')
    scatter = plt.scatter(0.01 * X_iso[:, 3], X_iso[:, 4], cmap=colormap, c=labels, s=8)
    cb = plt.colorbar(scatter, spacing='proportional',ticks=np.arange(N))
    cb.set_label('Cluster')
    plt.clim(-0.5, N - 0.5)
    cb.ax.set_yticklabels(np.arange(0,N,1))
    plt.xlabel(r"${\rm sign}(\lambda_2) \rho$")
    plt.ylabel("s")
    if verbose:
        print("  Writing png file {}...              ".format(filename + "-2d.png"), end="", flush=True)
    fig_2d.savefig(filename + "-2d.png")
    plt.close()
    if verbose:
        print("done")

    if X is not None:
        if verbose:
            print(
                "  Writing png file {}...          ".format(filename + "-2d-all.png"),
                end="",
                flush=True,
            )
        fig_all = plt.figure()
        plt.scatter(0.01 * X[:, 3], X[:, 4], c="darkgrey", s=8)
        scatter = plt.scatter(0.01 * X_iso[:, 3], X_iso[:, 4], cmap=colormap, c=labels, s=8)
        cb = plt.colorbar(scatter, spacing='proportional',ticks=np.arange(N))
        cb.set_label('Cluster')
        plt.clim(-0.5, N - 0.5)
        cb.ax.set_yticklabels(np.arange(0,N,1))
        plt.xlim(-0.07, 0.07)
        plt.ylim(0.0, 1.0)
        plt.xlabel(r"${\rm sign}(\lambda_2) \rho$")
        plt.ylabel("s")
        fig_all.savefig(filename + "-2d-all.png")
        plt.close()
        if verbose:
            print("done")


def plot_3d(X_iso, labels, filename, verbose=True):
    """ Plot clusters that correspond to nci regions in 3d, with different colors.
    
    Parameters
    ----------
    X_iso : np.array
       Array with columns corresponding to space coordinates, sign(l2)*dens and rdg; for data with rdg equal to or below isovalue.
    labels : np.array
       One dimensional array with integers that label the data in X_iso into different clusters.
    filename: str
       Filename for plot figure, without extension.
    """
    if verbose:
        print("  Writing png file {}...              ".format(filename + "-3d.png"), end="", flush=True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    N = len(np.unique(labels))
    colormap = discrete_cmap(N, base_cmap='rainbow')
    scat = ax.scatter(X_iso[:, 0], X_iso[:, 1], X_iso[:, 2], cmap=colormap, c=labels, s=8)

    ax2 = fig.add_subplot(1, 17, 17)
    cm = ax2.pcolormesh(np.arange(N).reshape((N,1)), cmap=colormap)
    cb = fig.colorbar(cm, cax=ax2, ticks=np.arange(N))
    #plt.clim(-0.5, N - 0.5)
    cb.ax.set_yticklabels(np.arange(0,N,1))
    plt.figtext(0.95, 0.5, "Cluster", ha="left", va="center", rotation="vertical")
    plt.savefig(filename + "-3d.png")
    plt.close()
    if verbose:
        print("done")


def plot_heatmap_distances(X_iso, labels, filename, verbose=True):
    """ Plot heatmap for distance between clusters.
    
    Parameters
    ----------
    X_iso : np.array
       Array with columns corresponding to space coordinates, sign(l2)*dens and rdg; for data with rdg equal to or below isovalue.
    labels : np.array
       One dimensional array with integers that label the data in X_iso into different clusters.
    filename: str
       Filename for plot figure, without extension.
    """
    if verbose:
        print("  Writing png file {}...              ".format(filename + "-hm.png"), end="", flush=True)
    clusters = []
    for label in set(labels):
        clusters.append(X_iso[np.where(labels == label)[0]])

    arr = min_distance_clusters(clusters, warning_val=None)
    mask = np.triu(np.ones_like(arr))
    min_dist = np.ma.masked_array(arr, mask=mask)

    cl_names = np.array(["cl" + str(n) for n, cl in enumerate(clusters)])
    cl_size = np.array([len(cl) for cl in clusters])
    cl_labels = np.array(["cl" + str(n) + "\n " + str(len(cl)) for n, cl in enumerate(clusters)])

    cl_min_size = np.zeros((len(cl_size), len(cl_size)))

    for i1, sc1 in enumerate(cl_size):
        for i2 in range(i1):
            sc2 = cl_size[i2]
            cl_min_size[i1, i2] = min(sc1, sc2)

    fig, ax = plt.subplots()
    cmap = "rainbow"
    im = ax.imshow(min_dist, cmap=cmap, norm=matplotlib.colors.Normalize())
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Distances between clusters (Angstrom)", rotation=-90, va="bottom")
    ax.set_title("Cluster size")

    # Loop over data dimensions and create text annotations.
    for i in range(len(cl_names)):
        for j in range(i):
            dist_str = "{:.2f}".format(min_dist[i, j])
            text = ax.text(j, i, dist_str, ha="center", va="center", color="w")

    # Labels
    ax.set_xticks(np.arange(len(cl_names)))
    ax.set_yticks(np.arange(len(cl_names)))
    ax.set_xticklabels(cl_labels)
    ax.set_yticklabels(cl_labels)
    ax.set_xticks(np.arange(len(cl_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(cl_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1.5)

    # Set limits
    left, right = plt.xlim()
    bottom, top = plt.ylim()
    plt.xlim((left, right - 1))
    plt.ylim((bottom, top + 1))

    plt.savefig(filename + "-hm.png")
    if verbose:
        print("done")
