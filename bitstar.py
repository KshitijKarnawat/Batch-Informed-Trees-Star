#! /usr/bin/env python3

"""Batch Informed Trees (BIT*) algorithm.
@author: Kshitij Karnawat
@author: Abhishek Reddy 
@date: 04/30/2024
"""


import math
import time
import random as rng
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

class Tree:
    """An explicit tree with a set of vertices which are a subset of X_free and edges {(v,w)} for some v,w which is an element of V.

    """
    def __init__(self, start, goal):
        self.start = start              # Start node
        self.goal = goal                # Goal node

        self.radius = 0.5                    # Radius of the neighborhood to explore
        self.vertices = set()           # Vertices in the tree
        self.edges = set()              # Edges in the tree
        self.queue_vertices = set()     # Vertices in the queue
        self.queue_edges = set()        # Edeges in the queue
        self.old_vertices = set()       # Vertices in the tree in the last iteration

class BITstar:
    def __init__():
        pass

    # Algorithm 1: BIT* Algorithm
    def plan():
        pass

    # Algorithm 2: Expand Vertex
    def expand_vertex():
        pass

    # Algorithm 3: Prune
    def prune():
        pass

    # Other functions used in Algorithm 1
    def best_queue_vertex():
        pass

    def best_queue_edge():
        pass

    def best_in_queue():
        pass

    def best_in_edge():
        pass

    def sample():
        pass

    def calculate_cost():
        pass

    def calculate_g_hat():
        pass

    def calculate_h_hat():
        pass

    def calculate_f_hat():
        pass

    def g_t():
        pass

    def update_radius():
        pass