# -*- coding: utf-8 -*-
"""Test spatial.divide module"""

import pytest
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

from spatial.divide import get_isosurface, divide_nci_regions
from spatial.plot import plot_2d, plot_3d


def test_get_isosurface():
    """Test get_isosurface function."""
    X = np.arange(0, 20, 1).reshape(4, 5)
    X_iso = get_isosurface(X, 20)
    assert np.allclose(X_iso, X)
    X_iso = get_isosurface(X, 5)
    assert np.allclose(X_iso, np.arange(0, 5, 1))
    X_iso = get_isosurface(X, 10)
    assert np.allclose(X_iso, np.arange(0, 10, 1).reshape(2, 5))
    X_iso = get_isosurface(X, 15)
    assert np.allclose(X_iso, np.arange(0, 15, 1).reshape(3, 5))
    X = np.arange(-1, 1, 0.1).reshape(4, 5)
    X_iso = get_isosurface(X, 1)
    assert np.allclose(X_iso, X)
    X_iso = get_isosurface(X, -0.5)
    assert np.allclose(X_iso, np.arange(-1, -0.5, 0.1))
    X_iso = get_isosurface(X, 0)
    assert np.allclose(X_iso, np.arange(-1, 0, 0.1).reshape(2, 5))
    X_iso = get_isosurface(X, 0.5)
    assert np.allclose(X_iso, np.arange(-1, 0.5, 0.1).reshape(3, 5))


def test_divide_nci_size_error():
    """Test error raise in divide_nci_regions"""
    X = np.arange(0, 20, 1).reshape(4, 5)
    with pytest.raises(ValueError) as error:
        divide_nci_regions(X, 2, isovalue=0.3, only_pos=True, size_sample=-1.0)
    assert str(error.value) == "size_sample must be a positive number, your input : -1.0"
    with pytest.raises(ValueError) as error:
        divide_nci_regions(X, 2, isovalue=0.3, only_pos=True, size_sample=0.0)
    assert str(error.value) == "size_sample must be a positive number, your input : 0.0"


def test_divide_nci_size():
    """Test output shapes in divide_nci_regions"""
    X = np.arange(0, 100, 1).reshape(20, 5)
    X_iso = get_isosurface(X, 40)
    X_iso2, labels = divide_nci_regions(X, 2, isovalue=40, only_pos=True, size_sample=1.0)
    assert X_iso.shape == X_iso2.shape
    X_iso2, labels = divide_nci_regions(X, 2, isovalue=40, only_pos=True, size_sample=0.5, method="kmeans")
    assert X_iso.shape[0] == 2 * X_iso2.shape[0]
    assert labels.shape[0] == X_iso2.shape[0]


def test_divide_nci_clusters():
    """Test output of divide_nci_regions"""
    X = np.array(
        [
            [-1.0, 0.0, 0.0, 1.0, 1.0],
            [-2.0, 0.0, 0.0, 1.0, 1.0],
            [11.0, 0.0, 0.0, 1.0, 1.0],
            [13.0, 0.0, 0.0, 1.0, 1.0],
            [12.0, 0.0, 0.0, 1.0, 4.0],
        ]
    )
    X_iso, labels = divide_nci_regions(X, 2, isovalue=2.0, only_pos=True, size_sample=1.0, method="kmeans")
    assert np.any([np.allclose([-1.0, 0.0, 0.0, 1.0, 1.0], X_iso[i]) for i in range(4)])
    assert np.any([np.allclose([-2.0, 0.0, 0.0, 1.0, 1.0], X_iso[i]) for i in range(4)])
    assert np.any([np.allclose([11.0, 0.0, 0.0, 1.0, 1.0], X_iso[i]) for i in range(4)])
    assert np.any([np.allclose([13.0, 0.0, 0.0, 1.0, 1.0], X_iso[i]) for i in range(4)])
    l1 = labels[X_iso[:, 0] == -1.0]
    if l1 == 0:
        l2 = 1
    else:
        l2 = 0
    assert l1 == labels[X_iso[:, 0] == -2.0]
    assert l2 == labels[X_iso[:, 0] == 11.0]
    assert l2 == labels[X_iso[:, 0] == 13.0]


def test_divide_nci_nclusters():
    """Test optimization of number of clusters in divide_nci_regions"""
    X = np.array(
        [
            [-1.0, 0.0, 0.0, 1.0, 1.0],
            [-2.0, 0.0, 0.0, 1.0, 1.0],
            [11.0, 0.0, 0.0, 1.0, 1.0],
            [13.0, 0.0, 0.0, 1.0, 1.0],
            [12.0, 0.0, 0.0, 1.0, 4.0],
        ]
    )
    X_iso, labels = divide_nci_regions(X, [2, 3, 4], isovalue=2.0, only_pos=True, size_sample=1.0, method="kmeans")
    assert len(set(labels)) == 2
    X = np.array(
        [
            [0.0, -1.0, 0.0, 1.0, 1.0],
            [0.0, -2.0, 0.0, 1.0, 1.0],
            [0.0, 11.0, 0.0, 1.0, 1.0],
            [0.0, 13.0, 0.0, 1.0, 1.0],
            [0.0, 20.0, 0.0, 1.0, 1.0],
            [0.0, 21.0, 0.0, 1.0, 1.0],
        ]
    )
    X_iso, labels = divide_nci_regions(X, [2, 3, 4], isovalue=2.0, only_pos=True, size_sample=1.0, method="kmeans")
    assert len(set(labels)) == 3
