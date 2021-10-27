# -*- coding: utf-8 -*-
"""Test spatial.utils module"""

import pytest
import numpy as np
import logging

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

from spatial.utils import readcube, writecube, cv_silhouette_scorer, pos_dens_grad_matrix
from spatial.utils import min_distance_clusters, max_distance_cluster, warning_small_cluster

from spatial.divide import divide_nci_regions


def test_readcube_header():
    """Test that header is correct."""
    header, pts, carray = readcube("example-dens.cube")
    assert header[0] == " dens_cube\n"
    assert header[-1] == "   1  0.0   0.354994   4.286012   6.640892\n"


def test_readcube_pts():
    """Test that gridpoints are correct."""
    header, pts, carray = readcube("example-dens.cube")
    assert pts.shape == (1, 1, 63, 3)
    origin_h = [float(header[2].split()[i]) for i in range(1, 4)]
    assert origin_h == list(pts[0, 0, 0])
    header, pts, carray = readcube("example1.cube")
    assert pts.shape == (3, 3, 3, 3)
    origin_h = [float(header[2].split()[i]) for i in range(1, 4)]
    assert origin_h == list(pts[0, 0, 0])


def test_readcube_carray():
    """Test that carray matrix is correct."""
    header, pts, carray = readcube("example-dens.cube")
    assert carray.shape == (1, 1, 63)
    assert carray[0, 0, 0] == 0.10000e03
    assert carray[0, 0, 62] == -0.23867e00
    header, pts, carray = readcube("example1.cube")
    assert carray.shape == (3, 3, 3)
    assert carray[0, 0, 0] == 0.10000e03
    assert carray[2, 2, 2] == -0.23867e00
    assert carray[1, 2, 0] == -0.14131e01


def test_writecube():
    """Test if cube file is written correctly."""
    filename = "example"
    header, X = pos_dens_grad_matrix(filename)
    X_iso, labels = divide_nci_regions(X, range(2, 10))
    writecube(filename, X_iso, X, labels, header)
    header_, pts_, carray_ = readcube("example-dens.cube")
    header0, pts0, carray0 = readcube("example-cl0-grad.cube")
    header1, pts1, carray1 = readcube("example-cl0-grad.cube")
    assert header[1:] == header0[1:]
    assert header[1:] == header1[1:]
    assert np.allclose(pts_, pts0)
    assert np.allclose(pts_, pts1)


def test_pdg_matrix():
    """Test that matrix with position, density and gradient is correct."""
    header, final = pos_dens_grad_matrix("example")
    header, pts, carray = readcube("example-dens.cube")
    assert final.shape == (63, 5)
    assert final[0, 3] == 0.10000e03
    assert final[62, 3] == -0.23867e00
    assert final[0, 4] == -0.17269e00
    assert final[62, 4] == 0.10000e03
    assert np.allclose(final[0, :3], pts[0, 0, 0])


def test_silhouette_special_cases():
    """Test cv_silhouette_scorer in cases where number of samples is equal to 1 or to number of clusters."""
    X = np.arange(0, 20, 1).reshape((4, 5))
    estimator = KMeans(n_clusters=1)
    score = cv_silhouette_scorer(estimator, X)
    assert score == -1
    estimator = KMeans(n_clusters=4)
    score = cv_silhouette_scorer(estimator, X)
    assert score == -1


def test_silhouette_score():
    """Test cv_silhouette_scorer."""
    X = np.array([[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [11.0, 0.0, 0.0], [13.0, 0.0, 0.0]])
    estimator = KMeans(n_clusters=2)
    estimator.fit(X)
    print(estimator.labels_)
    if estimator.labels_[0] == 1:
        assert np.allclose(estimator.labels_, np.array([1, 1, 0, 0]))
    if estimator.labels_[0] == 0:
        assert np.allclose(estimator.labels_, np.array([0, 0, 1, 1]))
    score = cv_silhouette_scorer(estimator, X)
    assert score == 0.8884293292913983


def test_min_dist(caplog):
    """Test min_distance_clusters."""
    cls = np.array(
        [
            [[-0.1, 0.0, 0.0], [-2.0, 0.0, 0.0], [11.0, 0.0, 0.0], [0.0, 13.0, 0.0]],
            [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0], [-11.0, 0.0, 0.0], [0.0, -13.0, 0.0]],
        ]
    )
    min_dist = min_distance_clusters(cls, warning_val=0.5)
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert "Minimun distance between clusters 1 and 0 is very small: 0.2 A" in caplog.text

    cls = np.array(
        [
            [[-10.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [11.0, 0.0, 0.0], [0.0, 14.0, 0.0]],
            [[10.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-11.0, 0.0, 5.0], [1.0, -13.0, 0.0]],
        ]
    )
    min_dist = min_distance_clusters(cls, warning_val=5.0)
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert "Minimun distance between clusters 1 and 0 is very small: 1.0 A" in caplog.text


def test_max_dist(caplog):
    """Test max_distance_cluster."""
    cl = np.array([[[0.0, -2.0, 0.0], [0.0, 13.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]])
    max_dist = max_distance_cluster(cl, warning_val=3.0)
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert "Maximum distance in cluster 0 is very big: 17.029386365926403 A" in caplog.text


def test_warn_small_cl(caplog):
    """Test warning_small_cluster."""
    cls = np.array(
        [
            [[-0.1, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0], [-11.0, 0.0, 0.0], [0.0, -13.0, 0.0]],
            [
                [0.1, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-11.0, 0.0, 0.0],
                [0.0, -13.0, 0.0],
                [0.1, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-11.0, 0.0, 0.0],
                [0.0, -13.0, 0.0],
                [2.0, 0.0, 0.0],
                [-11.0, 0.0, 0.0],
                [0.0, -13.0, 0.0],
            ],
        ]
    )
    warning_small_cluster(cls, portion=0.1, size=None)
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert "Cluster 0 is very small, having only 1 elements" in caplog.text

    warning_small_cluster(cls, portion=None, size=2)
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert "Cluster 0 is very small, having only 1 elements" in caplog.text

    cls = np.array(
        [
            [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[-11.0, 0.0, 0.0], [0.0, -13.0, 0.0]],
            [
                [0.1, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-11.0, 0.0, 0.0],
                [0.0, -13.0, 0.0],
                [0.1, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-11.0, 0.0, 0.0],
                [0.0, -13.0, 0.0],
                [2.0, 0.0, 0.0],
                [-11.0, 0.0, 0.0],
                [0.0, -13.0, 0.0],
            ],
        ]
    )
    warning_small_cluster(cls, portion=0.2, size=None)
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert "Cluster 1 is very small, having only 2 elements" in caplog.text
    warning_small_cluster(cls, portion=None, size=4)
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert "Cluster 0 is very small, having only 3 elements" in caplog.text
    assert "Cluster 1 is very small, having only 2 elements" in caplog.text
