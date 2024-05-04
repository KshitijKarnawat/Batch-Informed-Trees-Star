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
from scipy.spatial.transform import Rotation


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
        self.bloat = 0.1                            # Step size


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

                # self.tree.radius = self.update_radius(len(self.tree.vertices) + len(self.x_sample))

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

        for (v,x) in np.cross(self.tree.vertices, x_near):
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
    def prune(self, c):

        for x in self.x_sample:
            if self.calculate_f_hat(x) >= c:
                self.x_sample.remove(x)
        
        for v in self.tree.vertices:
            if self.calculate_f_hat(v) > c:
                self.tree.vertices.remove(v)

        for (v,w) in self.tree.edges:
            if self.calculate_f_hat(v) > c or self.calculate_f_hat(w) > c:
                self.tree.edges.remove((v,w))

        for v in self.tree.vertices:
            if self.g_t[v] == np.inf:
                self.x_sample.add(v)
                self.tree.vertices.remove(v)

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

    def best_in_queue_edge(self):
        if not self.tree.queue_edges:
            print("Edges queue in tree is empty")
            return None

        edge_value = {(vertex, x): self.g_T[vertex] + self.calculate_distance(vertex, x) + self.calculate_h_hat(x)
                   for vertex, x in self.tree.queue_edges}

        return min(edge_value, key=edge_value.get)

    def sample(self, num_samples, c_max, c_min, center, C):
        sample_set = set()
        samples_created = 0

        if c_max < math.inf:
            # Sample from the ellipse
            radius = [c_max / 2, 
                      math.sqrt(c_max ** 2 - c_min ** 2) / 2,
                      math.sqrt(c_max ** 2 - c_min ** 2) / 2]
            
            l = np.diag(radius)

            while samples_created < num_samples:
                ball = self.sample_unit_ball()
                rand = np.dot(np.dot(C, l), ball) + center

                node = Node(rand[(0,0)], rand[(1,0)])

                # check if the node is in the free space
                check_in_obstacle = in_obstacle(node)
                
                if self.x_range[0] + self.bloat <= node.x <= self.x_range[1] - self.bloat:
                    check_x = True
                else:
                    check_x = False

                if self.y_range[0] + self.bloat <= node.y <= self.y_range[1] - self.bloat:
                    check_y = True
                else:
                    check_y = False

                if not check_in_obstacle and check_x and check_y:
                    sample_set.add(node)
                    samples_created += 1

        else:
            # Sample from the free space
            while samples_created < num_samples:
                node = Node(rng.uniform(self.x_range[0] + self.bloat, self.x_range[1] - self.bloat),
                            rng.uniform(self.y_range[0] + self.bloat, self.y_range[1] - self.bloat))
                if in_obstacle(node):
                    continue
                else:
                    sample_set.add(node)
                    samples_created += 1
        
        return sample_set

    def calculate_cost(self, node1, node2):
        if in_obstacle(): # TODO: Implement this function
            return np.inf
        else:
            return self.calculate_euclidean_distance(node1, node2)

    def calculate_euclidean_distance(self, node1, node2):
        """Calculate the cost to come to a node.

        Args:
            v (Node): The first node
            w (Node): The second node

        Returns:
            float: The cost to come to the node
        """
        return math.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)

    # Helper functions for the algorithm (Makes it easier to write code from the given pseudocode in the paper)
    def calculate_g_hat(self, node):
        return self.calculate_euclidean_distance(self.start, node)

    def calculate_h_hat(self, node):
        return self.calculate_euclidean_distance(node, self.goal)

    def calculate_f_hat(self, node):
        return self.calculate_g_hat(node) + self.calculate_h_hat(node)

    def update_radius(self, length):
        pass

    def backtrack():
        # TODO: Implement this function
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
    
    def draw_map(self):
        plt.cla()
        figure, axes = plt.subplots()

        axes.add_patch(patches.Rectangle((0, 0), 1, 30))
        axes.add_patch(patches.Rectangle((0, 30), 50, 1))
        axes.add_patch(patches.Rectangle((1, 0), 50, 1))
        axes.add_patch(patches.Rectangle((50, 1), 1, 30))

        axes.add_patch(patches.Circle((5, 5), 0.5))
        axes.add_patch(patches.Circle((9, 6), 1))
        axes.add_patch(patches.Circle((7, 5), 1))
        axes.add_patch(patches.Circle((1, 5), 1))
        axes.add_patch(patches.Circle((7, 9), 1))

        plt.plot(self.start.x, self.start.y, "rs", linewidth=3)
        plt.plot(self.goal.x, self.goal.y, "gs", linewidth=3)
        plt.axis("equal")
        plt.title("Map")
    
    def visualize_plan(self, c_max, c_min, theta, center):
        self.draw_map()

        for vertex in self.x_sample:
            plt.plot(vertex.x, vertex.y, color='orange', markersize='2', marker='.')

        for vertex, w in self.tree.edges:
            plt.plot([vertex.x, w.x], [vertex.y, w.y], '-g')
        
        if c_max >= np.inf:
            return
        
        a = math.sqrt(c_max ** 2 - c_min ** 2) / 2.0
        b = c_max / 2.0
        angle = math.pi / 2.0 - theta
        cx = center[0]
        cy = center[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.2)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rotation.from_euler('z', -angle).as_dcm()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, marker='.', color='darkorange')
        plt.plot(px, py, linestyle='--', color='darkorange', linewidth=2)

def main():
    start = (1, 1)
    goal = (9, 9)

    bitstar = BITstar(start, goal)
    tree = bitstar.plan()
    bitstar.visualize_plan()

if __name__ == "__main__":
    main()
