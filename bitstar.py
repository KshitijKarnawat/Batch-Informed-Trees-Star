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
        self.map_size = [[0, 10], [0, 10]]          # Map size
        self.bounds = [[0, 0, 1, 10],
                       [0, 10, 10, 1],
                       [1, 0, 10, 1],
                       [10, 1, 1, 10]]              # Boundaries of the map
        self.obstacles = [[5, 5, 0.5], 
                          [9, 6, 1],
                          [7, 5, 1],
                          [1, 5, 1],
                          [7, 9, 1]]                # Obstacles in the map
        self.fig, self.ax = plt.subplots()          # Plotting Map


    # Algorithm 1: BIT* Algorithm
    def plan(self):
        self.tree.vertices.add(self.start)          # Add start node to the tree
        self.x_sample.add(self.goal)                # Add goal node to the sample set
        
        self.g_t[self.start] = 0.0                  # Cost to come to the start node is 0
        self.g_t[self.goal] = math.inf              # Cost to come to the goal node is infinity (since we don't know the cost yet)

        c_min, theta = self.calculate_distance_and_angle(self.start, self.goal) # Calculate the distance between the start and goal nodes
        
        C = self.rotate_to_world_frame(self.start, 
                                       self.goal,
                                       c_min)
            
        center = np.array([[(self.start.x + self.goal.x) / 2.0], 
                           [(self.start.y + self.goal.y) / 2.0],
                           [0.0]])                  # Calculate the center of the start and goal nodes

        for i in range(self.max_iter):
            if self.tree.queue_vertices is not None and self.tree.queue_edges is not None:
                if i == 0:
                    num_samples = 400
                else:
                    num_samples = 250

                # Backtrack here
                if self.goal.parent is not None:
                    path = self.backtrack()
                    plt.plot(path[0], path[1], linewidth=2, color='r')
                    plt.pause(0.5)

                self.prune(self.g_t[self.goal])
                self.x_sample.update(self.sample(num_samples,
                                                self.g_t[self.goal],
                                                c_min,
                                                center,
                                                C
                                                )
                                    )

                self.tree.old_vertices = self.tree.vertices
                self.tree.queue_vertices = self.tree.vertices

            while self.best_queue_vertex() <= self.best_queue_edge():
                self.expand_vertex(self.best_in_queue_vertex())

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

            if i % 10 == 0:
                self.visualize(center, self.g_t[self.goal], c_min, theta)

        path = self.backtrack()
        plt.plot(path[0], path[1], linewidth=2, color='r')
        plt.pause(1)
        plt.show()

    # Algorithm 2: Expand Vertex
    def expand_vertex(self, v):
        x_near = set()
        v_near = set()

        self.tree.queue_vertices.remove(v)

        for x in self.x_sample:
            if self.calculate_euclidean_distance(x,v) <= self.tree.radius:
                x_near.add(x)

        for x in x_near:
            if self.calculate_g_hat(v) + self.calculate_euclidean_distance(v,x) + self.calculate_h_hat(x) < self.g_t[self.goal]:
                self.g_t[x] = np.inf
                self.tree.queue_edges.add((v,x))
        
        if v not in self.tree.old_vertices:
            for w in self.tree.vertices:
                if self.calculate_euclidean_distance(x, v) <= self.tree.radius:
                    v_near.add(w)

            for w in v_near:
                if (v,w) not in self.tree.edges:
                    if self.calculate_g_hat(v) + self.calculate_euclidean_distance(v,w) + self.calculate_h_hat(w) < self.g_t[self.goal]:
                        if self.g_t(v) + self.calculate_euclidean_distance(v,w) < self.g_t[w]:
                            self.tree.queue_edges.add(v,w)
                            if w not in self.g_t:
                                self.g_t[w] = np.inf
            

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
    def best_queue_vertex_value(self):
        if not self.tree.queue_vertices:
            return np.inf
        
        best = np.inf
        for v in self.tree.queue_vertices:
            best = min(best, self.g_t[v] + self.calculate_h_hat(v))
        return best

    def best_queue_edge_value(self):
        if not self.tree.queue_edges:
            return np.inf

        best = np.inf
        for (v,x) in self.tree.queue_edges:
            best = min(best, self.g_t[v] + self.calculate_euclidean_distance(v,x) + self.calculate_h_hat(x))
        return best
    
    def best_in_queue_vertex(self):
        if not self.tree.queue_vertices:
            print("Vertices queue in tree is empty")
            return None

        vertex_value = {}
        for v in self.tree.queue_vertices:
            vertex_value[v] = self.g_t[v] + self.calculate_h_hat(v)
        best = min(vertex_value, key=vertex_value.get)
        return best
    
    def best_in_queue_edge(self):
        if not self.tree.queue_edges:
            print("Edges queue in tree is empty")
            return None

        edge_value = {}
        for (v,x) in self.tree.queue_edges:
            edge_value[(v,x)] = self.g_t[v] + self.calculate_euclidean_distance(v,x) + self.calculate_h_hat(x)
        best = min(edge_value, key=edge_value.get)
        return best
    
    def sample(self, num_samples, c_max, c_min, center, C):
        sample_set = set()
        samples_created = 0
        x_range = self.map_size[0]
        y_range = self.map_size[1]

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
                if x_range[0] + self.bloat <= node.x <= x_range[1] - self.bloat:
                    check_x = True
                else:
                    check_x = False

                if y_range[0] + self.bloat <= node.y <= y_range[1] - self.bloat:
                    check_y = True
                else:
                    check_y = False

                if not self.in_obstacle(node) and check_x and check_y:
                    sample_set.add(node)
                    samples_created += 1

        else:
            # Sample from the free space
            while samples_created < num_samples:
                node = Node(rng.uniform(x_range[0] + self.bloat, x_range[1] - self.bloat),
                            rng.uniform(y_range[0] + self.bloat, y_range[1] - self.bloat))
                if self.in_obstacle(node):
                    continue
                else:
                    sample_set.add(node)
                    samples_created += 1
        
        return sample_set

    def calculate_cost(self, node1, node2):
        if self.path_through_obstacle(node1, node2):
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
    
    def in_obstacle(self, node):
        bloat = self.bloat

        for (x, y, r) in self.obstacles:
            if math.hypot(node.x - x, node.y - y) <= r + bloat:
                return True

        for (x, y, w, h) in self.bounds:
            if 0 <= node.x - (x - bloat) <= w + 2 * bloat \
                    and 0 <= node.y - (y - bloat) <= h + 2 * bloat:
                return True

        return False

    # Helper functions for the algorithm (Makes it easier to write code from the given pseudocode in the paper)
    def calculate_g_hat(self, node):
        return self.calculate_euclidean_distance(self.start, node)

    def calculate_h_hat(self, node):
        return self.calculate_euclidean_distance(node, self.goal)

    def calculate_f_hat(self, node):
        return self.calculate_g_hat(node) + self.calculate_h_hat(node)
    
    def backtrack(self):
        node = self.goal
        path = []

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        
        return path

    def calculate_distance_and_angle(self, node1, node2):
        """Calculate the Euclidean distance between two nodes and the angle between them.

        Args:
            node1 (Node): The first node
            node2 (Node): The second node

        Returns:
            float: The distance between the two nodes
        """
        return self.calculate_euclidean_distance(node1,node2), math.atan2((node2.y - node1.y), (node2.x - node1.x))
    
    def sample_unit_ball(self):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x ** 2 + y ** 2 <= 1:
                return np.array([[x], [y], [0.0]])
            
    def rotate_to_world_frame(self, start, goal, L):
        A = np.array([[goal.x - start.x],
                      [goal.y - start.y], 
                      [0.0]])
        
        E = np.array([[1.0],
                      [0.0], 
                      [0.0]])
        
        M = np.matmul(A, E.T)

        U, S, V_T = np.linalg.svd(M, True, True)

        C = np.matmul(np.matmul(U, np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T)])), V_T)

        return C

    def visualize(self, center, c_max, c_min, theta):
        plt.cla()
        self.plot_grid()

        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        
        for v in self.x_sample:
            plt.plot(v.x, v.y, marker='.', color='lightgray', markersize='2')

        if c_max < math.inf:
            self.plot_ellipse(center, c_max, c_min, theta)

        for v,w in self.tree.edges:
            plt.plot([v.x, w.x], [v.y, w.y], '-g')

        plt.pause(0.001)

    def plot_grid(self):
        for (x, y, w, h) in self.bounds:
            self.ax.add_patch(
                patches.Rectangle(
                    (x, y), w, h,
                    fill=True,
                    edgecolor='black',
                    facecolor='black'
                )
            )

        for (x, y, r) in self.obstacles:
            self.ax.add_patch(
                patches.Circle(
                    (x, y), r,
                    fill=True,
                    edgecolor='black',
                    facecolor='black'
                )
            )

        plt.plot(self.start.x, self.start.y, marker='o', color='b', markersize=5)
        plt.plot(self.goal.x, self.goal.y, marker='o', color='r', markersize=5)
        plt.axis('equal')

    def plot_ellipse(self, center, c_max, c_min, theta):
        a = math.sqrt(c_max ** 2 - c_min ** 2) / 2
        b = c_max / 2.0
        angle = math.pi / 2 - theta
        t = np.arange(0, 2 * math.pi + 0.1, 0.2)
        x = []
        y = []
        for i in t:
            x.append(a * math.cos(i))
            y.append(b * math.sin(i))
        rot = Rotation.from_euler('z', angle).as_matrix()[0:2, 0:2]
        fx = np.matmul(rot, np.array([x, y]))
        px = np.array(fx[0, :] + center[0]).flatten()
        py = np.array(fx[1, :] + center[1]).flatten()
        plt.plot(px, py, linestyle='--', color='blue')
        
    def path_through_obstacle(self, start, end):
        if self.in_obstacle(start) or self.in_obstacle(end):
            return True
        
        dir = [end.x - start.x, end.y - start.y]

        for (x, y, r) in self.obstacles:
            if np.dot(dir, dir) == 0:
                continue
            
            t = np.dot([x - start.x, y - start.y], dir) / np.dot(dir, dir)
            if 0 <= t <= 1:
                shot = Node(start.x + t * dir[0], start.y + t * dir[1])
                if self.calculate_euclidean_distance(shot, Node(x, y)) <= r:
                    return True
                
        return False

def main():
    start = (2, 2)
    goal = (9, 9)
    bitstar = BITstar(start, goal)
    bitstar.plan()

if __name__ == "__main__":
    main()
