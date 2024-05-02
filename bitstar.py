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

        self.radius = 0.5               # Radius of the neighborhood to explore
        self.vertices = set()           # Vertices in the tree
        self.edges = set()              # Edges in the tree
        self.queue_vertices = set()     # Vertices in the queue
        self.queue_edges = set()        # Edeges in the queue
        self.old_vertices = set()       # Vertices in the tree in the last iteration

class BITstar:
    def __init__(self, start, goal, max_iter=200) :
        self.start = Node(start[0], start[1])       # Start node
        self.goal = Node(goal[0], goal[1])          # Goal node
        self.max_iter = max_iter                    # Maximum number of iterations
        self.tree = Tree(self.start, self.goal)     # Tree
        self.x_sample = set()                       # Sampled nodes
        self.g_t = dict()                           # Cost to come to a node
        self.delta = 0.1                            # Step size
        self.eta = 1                                # Eta as a tuning parameter  


    # Algorithm 1: BIT* Algorithm
    def plan(self):
        self.tree.vertices.add(self.start)          # Add start node to the tree
        self.x_sample.add(self.goal)                # Add goal node to the sample set
        
        self.g_t[self.start] = 0.0                  # Cost to come to the start node is 0
        self.g_t[self.goal] = math.inf              # Cost to come to the goal node is infinity (since we don't know the cost yet)

        c_min, theta = self.calculate_distance_and_angle(self.start, self.goal) # Calculate the distance between the start and goal nodes
        
        center = np.array([[(self.start.x + self.goal.x) / 2], 
                           [(self.start.y + self.goal.y) / 2],
                           [0,0]])                  # Calculate the center of the start and goal nodes

        for i in range(self.max_iter):
            if self.tree.queue_vertices is not None and self.tree.queue_edges is not None:
                if i == 0:
                    num_samples = 400
                else:
                    num_samples = 250

                # Backtrack here
                if self.goal.parent is not None:
                    self.backtrack()

                self.prune(self.g_t[self.goal])
                self.x_sample.update(self.sample(num_samples,
                                                self.g_t[self.goal]
                                                ))

                self.tree.old_vertices = self.tree.vertices
                self.tree.vertices = self.tree.vertices

                self.tree.radius = self.update_radius(self.tree.vertices, self.x_sample)

            while self.best_queue_vertex() <= self.best_queue_edge():
                self.expand_vertex(self.best_in_queue_vertex)

            vm, xm = self.best_in_queue_edge()

            self.tree.queue_edges.remove((vm,xm))
            
            if self.g_t[vm] + self.calculate_euclidean_distance(vm, xm) + self.calculate_h_hat(xm) < self.g_t[self.goal]:
                if self.calculate_g_hat(vm) + self.calculate_cost(vm, xm) + self.calculate_h_hat(xm) < self.g_t[self.goal]:
                    if self.g_t[vm] + self.calculate_cost(vm, xm) < self.g_t[xm]:
                        for v in self.tree.vertices:
                            if xm in self.tree.vertices:
                                self.tree.edges.remove((v, xm))
                            else:
                                self.x_sample.remove(xm)
                                self.tree.vertices.add(xm)
                                self.tree.queue_vertices.add(xm)
                            self.tree.edges.add((v, xm))
                            for (v,xm) in self.tree.queue_edges:
                                if self.g_t[v] + self.calculate_euclidean_distance(v, xm) >= self.g_t[xm]:
                                    self.tree.queue_edges.remove((v, xm))
            else:
                self.tree.queue_edges = set()
                self.tree.queue_vertices = set()
        
        return self.tree

    # Algorithm 2: Expand Vertex
    def expand_vertex(self, v):
        x_near = set()
        v_near = set()

        self.tree.queue_vertices.remove(v)

        for x in self.x_sample:
            if np.linalg.norm(x-v) <= self.tree.radius:
                x_near.add(x)

        for x in x_near:
            if self.calculate_g_hat(v) + self.calculate_euclidean_distance(v,x) + self.calculate_h_hat(x) < self.g_t[self.goal]:
                self.tree.queue_edges.add(v,x)
        
        if v not in self.tree.old_vertices:
            for w in self.tree.vertices:
                if self.np.linalg.norm(x-v) <= self.tree.radius:
                    v_near.add(w)

        for (v,w) in np.cross(self.tree.vertices, v_near):
            if (v,w) not in self.tree.edges:
                if self.calculate_g_hat(v) + self.calculate_euclidean_distance(v,w) + self.calculate_h_hat(w) < self.g_t[self.goal]:
                    if self.g_t(v) + self.calculate_euclidean_distance(v,w) < self.g_t[w]:
                        self.tree.queue_edges.add(v,w)
            

    # Algorithm 3: Prune
    def prune(self, c_best):
        self.x_sample = {x for x in self.x_sample if self.calculate_f_hat(x) < c_best}
        self.tree.vertices = {vertex for vertex in self.tree.vertices if self.calculate_f_hat(vertex) <= c_best}
        self.tree.edges = {(vertex, w) for vertex, w in self.tree.edges
                       if self.calculate_f_hat(vertex) <= c_best and self.calculate_f_hat(w) <= c_best}
        self.x_sample.update({vertex for vertex in self.tree.vertices if self.g_T[vertex] == np.inf})
        self.tree.vertices = {vertex for vertex in self.tree.vertices if self.g_T[vertex] < np.inf}

    # Other functions used in Algorithm 1
    def best_queue_vertex(self):
        if not self.tree.queue_vertices:
            return np.inf
        
        return min(self.g_t[vertex] + self.calculate_h_hat(vertex) for vertex in self.tree.queue_vertices)

    def best_queue_edge(self):
        if not self.tree.queue_edges:
            return np.inf

        return min(self.g_T[vertex] + self.calculate_distance(vertex, x) + self.calculate_h_hat(x)
                   for vertex, x in self.tree.queue_edges)
    
    def best_in_queue_vertex(self):
        if not self.tree.queue_vertices:
            print("Vertices queue in tree is empty")
            return None

        vertex_value = {vertex: self.g_T[vertex] + self.calculate_h_hat(vertex) for vertex in self.tree.queue_vertices}

        return min(vertex_value, key=vertex_value.get)

    def best_in_queue_edge():
        pass

    def sample():
        pass

    def calculate_cost():
        pass

    def calculate_euclidean_distance(self, node1, node2):
        """Calculate the cost to come to a node.

        Args:
            v (Node): The first node
            w (Node): The second node

        Returns:
            float: The cost to come to the node
        """
        return math.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)

    def calculate_g_hat():
        pass

    def calculate_h_hat():
        pass

    def calculate_f_hat():
        pass

    def update_radius():
        pass

    def backtrack():
        pass

    def calculate_distance_and_angle(self, node1, node2):
        """Calculate the Euclidean distance between two nodes and the angle between them.

        Args:
            node1 (Node): The first node
            node2 (Node): The second node

        Returns:
            float: The distance between the two nodes
        """
        return self.calculate_euclidean_distance(node1,node2), math.atan2((node2.y - node1.y), (node2.x - node1.x))
    
