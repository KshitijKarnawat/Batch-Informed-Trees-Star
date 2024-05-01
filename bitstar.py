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

        self.r = 0.5                    # Radius of the neighborhood to explore
        self.vertices = set()           # Vertices in the tree
        self.edges = set()              # Edges in the tree
        self.queue_vertices = set()     # Vertices in the queue
        self.queue_edges = set()        # Edeges in the queue
        self.old_vertices = set()       # Vertices in the tree in the last iteration

