#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:08:03 2021

@author: hiroshima
"""
import numpy as np
import networkx as nx
import os
import Ball as bl
import math
import itertools as it
from random import uniform
import pylab as pl
import random

import Line as ln


class FasterNetwork:
    
    def __init__(self, m_edges = int(), N = int(),animate = False, visual = False, random = False, mixed = False): #, animate = False, networkx = False):
        
        self.m_edges = m_edges
        self.N = N
        self.animate = animate
        self.visual = visual
        self.random = random
        self.mixed = mixed
        self.c = bl.Ball(4.0e50,-10.0,
                         np.array([0.0,0.0]),
                         np.array([0.0,0.0]),
                         Container = True)
    
    def create_matrix(self):
        
        self.vertex_count = 0
        self.vertex_count += self.m_edges
        self.list_vertex = []
        self.list_edges = []
        self.num = self.m_edges + 1
        self.edge_count = [0]*(self.num + self.N)
        self.edge_connect = []
        
        self.sub_list = [[]]
        self.b = [self.c]
        self.line_list = [] 
        
        

        for i in range(self.num):
            self.add_vertex(i+1)
            for j in range(self.m_edges):
                self.list_edges.append(i+1)
            self.edge_count[i] += self.m_edges
            
            
        if self.animate == True:
            for i in range(self.m_edges):
                B = FasterNetwork.particle_production(self,0.1,1,0.2, i)
                self.b.append(B)
                if i != 0:
                    Li = FasterNetwork.line_production(self, self.b[i]._pos, self.b[i+1]._pos, i, i+1)
                    self.line_list.append(Li)
        
        if self.animate == True and self.m_edges != 2:
            Li = FasterNetwork.line_production(self, self.b[1]._pos, self.b[-1]._pos, 1, -1)
            self.line_list.append(Li)

        if self.visual == True:
            if self.m_edges == 2:
                self.edge_connect.append([1,2])
            else:
                for i in range(self.m_edges):
                    if i == 0:
                        self.edge_connect.append([1, self.m_edges])
                    else:
                        self.edge_connect.append([i,i+1])

        


    def add_vertex(self, vertex_number):
        
        self.list_vertex.append(vertex_number)
        
        
        
        self.vertex_count += 1
        if self.animate == True:
            B = FasterNetwork.particle_production(self,0.1,1,0.2,1)
            self.b.append(B)
        
    def add_edges(self, vertex_number):
        
        self.new_lines = []
        new_v = []
        
        for i in range(self.m_edges):
            number = 2
            if self.mixed != False:
                number = np.random.choice((0,1), p = (self.mixed,1-self.mixed))
                if number == 0:
                    chosen_vertex = random.choice(self.list_edges)
                    if i > 0:
                        while new_v.count(chosen_vertex) == 1:
                            chosen_vertex = random.choice(self.list_edges)
                else:
                    chosen_vertex = random.choice(self.list_vertex)
                    if i > 0:
                        while new_v.count(chosen_vertex) == 1:
                            chosen_vertex = random.choice(self.list_vertex)
                new_v.append(chosen_vertex)

            elif self.random == False:
                
                chosen_vertex = self.list_edges[random.randint(1,len(self.list_edges) -1)]
                if i > 0:
                    while new_v.count(chosen_vertex) == 1:
                        chosen_vertex = self.list_edges[random.randint(1,len(self.list_edges) -1)]
                new_v.append(chosen_vertex)
                    
            else:
                chosen_vertex = random.choice(self.list_vertex)
                if i > 0:
                    while new_v.count(chosen_vertex) == 1:
                        chosen_vertex = random.choice(self.list_vertex)
                new_v.append(chosen_vertex)
            
            
            
            if self.animate == True:
                Li = FasterNetwork.line_production(self, 
                                             self.b[self.vertex_count]._pos, 
                                             self.b[int(chosen_vertex)]._pos, 
                                             self.vertex_count, 
                                             chosen_vertex)
                self.new_lines.append(Li)
                print('Connecting', self.vertex_count, 'to', chosen_vertex)
        for i in new_v:
            self.list_edges.append(vertex_number)
            self.list_edges.append(i)
            self.edge_count[vertex_number-1] += 1
            self.edge_count[i-1] += 1

        if self.visual == True:
            for i in new_v:
                self.edge_connect.append([vertex_number, i])
            
    def run(self):
        
        self.create_matrix()
        large_k = []
        list1 = [10,32, 100,316,1000,3162,10000, 31622, 100000,316227, 1000000, 3162277, 10000000]
        if self.animate == False:
            for i in range(self.N):
                num = self.num + i + 1
                self.add_vertex(num)
                self.add_edges(num)
                if i in list1 :
                    large_k.append(max(self.edge_count))

                
                

        if self.animate == True:
            f = pl.figure()
            f.patch.set_facecolor('pink')
            ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
            ax.add_artist(self.c.get_patch())
            for i in self.b:
                ax.add_patch(i.get_patch()) 
            for i in self.line_list:
                ax.add_patch(i.get_patch())
            for i in range(self.N):
                num = self.m_edges + 1 + i
                self.add_vertex(num)
                ax.add_patch(self.b[i+1+self.m_edges].get_patch()) 
                self.add_edges(num)
                
    

                pl.pause(0.75)
                for i in self.new_lines:
                    ax.add_patch(i.get_patch())
                pl.pause(0.75)
                self.next_collision(self.vertex_count)
        
        
        return self.list_vertex, self.list_edges, self.edge_count, self.edge_connect, large_k
    

    def line_production(self, b1pos, b2pos, i, j):
        
        Li = ln.Line()

        Li.pos = b1pos
        Li.pos_end = b2pos
        Li.ball1 = i
        Li.ball2 = j
        
        return Li
    

    def particle_production(self, rad, mass, vel_max, i):
        
        B = bl.Ball(mass,rad)
        if self.vertex_count == self.m_edges:
            B._pos = np.array([position_list[i][0], 
                           position_list[i][1]])
        else:
            B._pos = np.array([position_list[self.vertex_count -1][0], 
                               position_list[self.vertex_count -1][1]])
        
        B._vel = np.array([uniform(-vel_max,vel_max),uniform(-vel_max,
                                                             vel_max)])
        return B
            
    def next_collision(self, n):
        
        """ Code adapted from my Y2 Thermodynamics Ball Project """

        """ Finds the shortest time_to_collision amongst all the balls, and
            moves the simulation forward by that time via the move and collide
            methods. """
        
        time_array = []
        indexi = []
        indexj = []
        # 'for' statements manipulated to ensure the prevention of double 
        # readings.
        for i in np.linspace(1, n + 1, n + 1, dtype = int):
            # Using linspace over range out of preference, as the first item
            # in the list if the container whcih isn't always included in 
            # calculations.
            for j in np.linspace(1 + (i), n + 1 , n - (i - 1),
                                 dtype = int):
                if i == j:
                    continue 
                else:
                    t = self.b[i - 1].time_to_collision(self.b[j - 1]) 
                    # [i-1] done to include the container.
                    time_array.append(t)
                    indexi.append(i - 1)
                    indexj.append(j - 1)
        
        short_time = min(time_array)
        # Finding the shortest time for a ball to collide
        index_position = time_array.index(short_time)
        # How to find the ball which actually has that shortest time
        for i in np.linspace(1, n + 1, n + 1, dtype = int):
                self.b[i - 1].move(3)    
                
        # Moves every particle in the simulation for a time = short_time
        self.b[indexi[index_position]].collide(self.b[indexj[index_position]])
        
        """ Calculates the momentum of the balls which collide with the 
            container. """
            
        momentum_list = []     
        
        if (indexi[index_position] == int(0)): 
            # First item in the ball list is the container - hence if object in 
            # collision is the container, find the change in momentum of the
            # ball.
            p = self.b[indexj[index_position]].momentum()
            momentum_list.append(p)
        elif (indexj[index_position] == int(0)):
            p = self.b[indexj[index_position]].momentum()
            momentum_list.append(p)
        else:
            momentum_list.append(0) 
            # If no collision with container during a certain frame a zero is
            # added to the list to ensure temporal consistency.
        
        return (momentum_list, short_time) 
    

        

            
        
        
#%%      
""" For simulation          
        if self.visual == True:
            if self.m_edges == 2:
                self.edge_connect.append([1,2])
            else:
                for i in range(self.m_edges):
                    if i == 0:
                        self.edge_connect.append([1, self.m_edges])
                    else:
                        self.edge_connect.append([i,i+1])

        if self.visual == True:
            for i in new_v:
                self.edge_connect.append([vertex_number, i])
"""
N = 50
rad = 0.2
n = int(np.ceil(np.sqrt(N))) # np.ciel ensures the int() doesnt 
cont_radius = np.abs(-10) # round the value down.
max_length = (np.sqrt(cont_radius) - rad) * 2 
n_range = np.linspace(-max_length, max_length, n)
max_number = (max_length * 2)**2/(math.pi * rad**2) 
position_list = list(it.product(n_range, n_range))      

class Network:
    
    def __init__(self, c, m_edges = int(), N = int()):
        
        self.m_edges = m_edges
        self.N = N
        self.c = c

    def create_matrix(self, animate):
        
        """ Creates an initial matrix with vertex_number > m_edges """
        
        print('Creating initial seed')
        self.b = [self.c]
        self.line_list = []
        G = nx.Graph()
        input_edge = self.m_edges
        edges_count = input_edge 
        matrix_size = input_edge
        if self.m_edges == 2:
            edges_count = 1
        
        edges = [2]*input_edge
        if self.m_edges == 2:
            edges = [1,1]
        init_matrix =[0]*input_edge
        
        for i in range(input_edge):
            init_matrix[i] = [0]*input_edge
            for j in range(input_edge):
                if i == j:
                    if i+1 < input_edge:
                        init_matrix[i][i+1] = 1
                    else:
                        init_matrix[i][0] = 1
                    init_matrix[i][i-1] = 1
                
        for i in range(input_edge):
            G.add_node(i)
            if animate == True:
                B = Network.particle_production(self,0.1,1,0.2, init_matrix, i)
                self.b.append(B)
                if i != 0:
                    Li = Network.line_production(self, self.b[i]._pos, self.b[i+1]._pos, i, i+1)
                    self.line_list.append(Li)
        
            
            if i < input_edge:
                G.add_edge(i,i+1)
                
        if animate == True and self.m_edges != 2:
            Li = Network.line_production(self, self.b[1]._pos, self.b[-1]._pos, 1, -1)
            self.line_list.append(Li)
        
        return init_matrix, edges, edges_count, G, matrix_size
    
    def line_production(self, b1pos, b2pos, i, j):
        
        Li = ln.Line()

        Li.pos = b1pos
        Li.pos_end = b2pos
        Li.ball1 = i
        Li.ball2 = j
        
        return Li
    

    def particle_production(self, rad, mass, vel_max, init_matrix, i):
        
        B = bl.Ball(mass,rad)
        if len(init_matrix) == self.m_edges:
            B._pos = np.array([position_list[i][0], 
                           position_list[i][1]])
        else:
            B._pos = np.array([position_list[len(init_matrix) -1][0], 
                               position_list[len(init_matrix) -1][1]])
        
        B._vel = np.array([uniform(-vel_max,vel_max),uniform(-vel_max,
                                                             vel_max)])
        return B
        

        
    def add_vertex(self, init_matrix, animate, matrix_size):
        
        
        

        for i in range(matrix_size):
            init_matrix[i].append(0)
            
        init_matrix.append([0]*(matrix_size+1))
        if animate == True:
            B = Network.particle_production(self,0.1,1,0.2, init_matrix, 1)
            self.b.append(B)
        matrix_size += 1
        
        return init_matrix, matrix_size

    def add_edges(self, init_matrix, edges, edges_count, G,animate, matrix_size):
        
        
        G.add_node(matrix_size -1)
        p_array = np.array(edges)/(2*edges_count)

        edges.append(0)
        self.new_lines = []
        v_list = []
        for i in range(self.m_edges):
            

            vertex = np.random.choice(np.linspace(1,matrix_size - 1, 
                                            matrix_size -1), p = p_array)
            if i > 0:
                while v_list.count(vertex) == 1:
                    vertex = np.random.choice(np.linspace(1,matrix_size- 1, 
                                            matrix_size -1), p = p_array)
                
            v_list.append(vertex)
            init_matrix[int(vertex) - 1][matrix_size - 1] = 1
            init_matrix[matrix_size - 1][int(vertex) - 1] = 1
            G.add_edge(matrix_size - 1, vertex - 1)
            
            if animate == True:
                Li = Network.line_production(self, 
                                             self.b[matrix_size]._pos, 
                                             self.b[int(vertex)]._pos, 
                                             (matrix_size), 
                                             vertex)
                self.new_lines.append(Li)
                print('Connecting', matrix_size, 'to', vertex)
                
        
            
        edges_count += self.m_edges
        
        for i in v_list:
            edges[int(i)-1] += 1
        edges[matrix_size -1] = self.m_edges
        return init_matrix, edges, edges_count, G, p_array
        
    def simulate(self, animate):
                
        if (animate == True) and self.N > 15:
            raise Exception('Number of N too high')

        init_matrix, edges, edges_count, G, matrix_size = Network.create_matrix(self, animate)
        if animate == True:
            f = pl.figure()
            f.patch.set_facecolor('pink')
            ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
            ax.add_artist(self.c.get_patch())
            for i in self.b:
                ax.add_patch(i.get_patch()) 
            for i in self.line_list:
                ax.add_patch(i.get_patch())
            for i in range(self.N):
                init_matrix, matrix_size = Network.add_vertex(self,init_matrix, animate, matrix_size)
                ax.add_patch(self.b[i+1+self.m_edges].get_patch()) 
                
    
    
                

                (init_matrix, edges, edges_count, G, p_array) = Network.add_edges(self,
                                            init_matrix, 
                                            edges, 
                                            edges_count,
                                            G,
                                            animate,
                                            matrix_size)
                pl.pause(0.75)
                for i in self.new_lines:
                    ax.add_patch(i.get_patch())
                pl.pause(0.75)
                self.next_collision(len(init_matrix))

        if animate == False:
            for i in range(self.N):
                print(i)
                init_matrix, matrix_size = Network.add_vertex(self,init_matrix, animate, matrix_size)
                (init_matrix, edges, edges_count, G, p_array) = Network.add_edges(self,
                                            init_matrix, 
                                            edges, 
                                            edges_count,
                                            G,
                                            animate,
                                            matrix_size)

        return init_matrix, G, edges
        
    
    def visualisation(self):
        
        init_matrix, G, p_array = Network.simulate(self, False)
        filenameroot='er_N'+str(self.N)+'_k'
        outputdir='/Users/hiroshima/Desktop/Imperial College London/Complexity and Networks/Networks'

        filename=filenameroot+'.net'
        outputfile=os.path.join(outputdir,filename) # os module is best way to deal with file names
        print ("Writing network to "+outputfile)
        nx.write_pajek(G, outputfile) # write out .net file, comment out if don't want this
        

        
    def next_collision(self, n):
        
        """ Code adapted from my Y2 Thermodynamics Ball Project """

        """ Finds the shortest time_to_collision amongst all the balls, and
            moves the simulation forward by that time via the move and collide
            methods. """
        
        time_array = []
        indexi = []
        indexj = []
        # 'for' statements manipulated to ensure the prevention of double 
        # readings.
        for i in np.linspace(1, n + 1, n + 1, dtype = int):
            # Using linspace over range out of preference, as the first item
            # in the list if the container whcih isn't always included in 
            # calculations.
            for j in np.linspace(1 + (i), n + 1 , n - (i - 1),
                                 dtype = int):
                if i == j:
                    continue 
                else:
                    t = self.b[i - 1].time_to_collision(self.b[j - 1]) 
                    # [i-1] done to include the container.
                    time_array.append(t)
                    indexi.append(i - 1)
                    indexj.append(j - 1)
        
        short_time = min(time_array)
        # Finding the shortest time for a ball to collide
        index_position = time_array.index(short_time)
        # How to find the ball which actually has that shortest time
        for i in np.linspace(1, n + 1, n + 1, dtype = int):
                self.b[i - 1].move(3)    
                
        # Moves every particle in the simulation for a time = short_time
        self.b[indexi[index_position]].collide(self.b[indexj[index_position]])
        
        """ Calculates the momentum of the balls which collide with the 
            container. """
            
        momentum_list = []     
        
        if (indexi[index_position] == int(0)): 
            # First item in the ball list is the container - hence if object in 
            # collision is the container, find the change in momentum of the
            # ball.
            p = self.b[indexj[index_position]].momentum()
            momentum_list.append(p)
        elif (indexj[index_position] == int(0)):
            p = self.b[indexj[index_position]].momentum()
            momentum_list.append(p)
        else:
            momentum_list.append(0) 
            # If no collision with container during a certain frame a zero is
            # added to the list to ensure temporal consistency.
        
        return (momentum_list, short_time) 
        
        
        
        
        
    