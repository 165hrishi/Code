#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:25:53 2021

@author: hiroshima
"""

import numpy as np
import matplotlib.pyplot as plt
import Ball as bl

from timeit import default_timer as timer

import Network as net
import networkx as nx

#%%
n = 20000
plt.figure()
m_edge = 2

Container = bl.Ball(4.0e50,-10.0,np.array([0.0,0.0]),np.array([0.0,0.0]), 
             Container = True)


a1 = net.Network(Container, m_edges = m_edge, N = n)
matrix, G, p_array = a1.simulate(False)

edges = []
for i in range(n):
    edges.append(matrix[i].count(1))
    

#%%
def logbin(data, i, scale = 1.1, zeros = False):
   
    if scale < 1:
        raise ValueError('Function requires scale >= 1.')
    count = np.bincount(data)
    tot = np.sum(count)
    #smax = np.max(data)

    
    if scale > 1:
        if i == 1:
            smax= np.exp(19)
        elif i == 0:
            smax = np.exp(13)
        elif i ==2 :
            smax = np.exp(23)
        elif i ==3:
            smax = np.exp(26)
        elif i ==4:
            smax = np.exp(31)

            
        jmax = np.ceil(np.log(smax)/np.log(scale))
        if zeros:
            binedges = scale ** np.arange(jmax + 1)
            binedges[0] = 0
        else:
            binedges = scale ** np.arange(1,jmax + 1)
            # count = count[1:]
        binedges = np.unique(binedges.astype('uint64'))
        x = (binedges[:-1] * (binedges[1:]-1)) ** 0.5
        y = np.zeros_like(x)

    
        count = count.astype('float')
        std = []
        for i in range(len(y)):
            y[i] = np.sum(count[binedges[i]:binedges[i+1]]/(binedges[i+1] - binedges[i]))
            
            #
            #print(binedges[i],binedges[i+1])
        #print('smax', smax,'jmax', jmax,'bin',binedges,'x',x)
        #print(x,y)


    else:
        x = np.nonzero(count)[0]
        y = count[count != 0].astype('float')
        if zeros != True and x[0] == 0:
            x = x[1:]
            y = y[1:]
    y /= tot
    x = x[y!=0]
    y = y[y!=0]
    return x,y
#%%
def logbin(data, i, scale = 1.1, zeros = False):
   
    if scale < 1:
        raise ValueError('Function requires scale >= 1.')
    count = np.bincount(data)
    tot = np.sum(count)
    smax = np.max(data)

    
    if scale > 1:

        jmax = np.ceil(np.log(smax)/np.log(scale))
        if zeros:
            binedges = scale ** np.arange(jmax + 1)
            binedges[0] = 0
        else:
            binedges = scale ** np.arange(1,jmax + 1)
            # count = count[1:]
        binedges = np.unique(binedges.astype('uint64'))
        x = (binedges[:-1] * (binedges[1:]-1)) ** 0.5
        y = np.zeros_like(x)

    
        count = count.astype('float')
        std = []
        for i in range(len(y)):
            y[i] = np.sum(count[binedges[i]:binedges[i+1]]/(binedges[i+1] - binedges[i]))
        print(y)   
            #
            #print(binedges[i],binedges[i+1])
        #print('smax', smax,'jmax', jmax,'bin',binedges,'x',x)
        #print(x,y)


    else:
        x = np.nonzero(count)[0]
        y = count[count != 0].astype('float')
        if zeros != True and x[0] == 0:
            x = x[1:]
            y = y[1:]
    
    y /= tot
    x = x[y!=0]
    y = y[y!=0]
    return x,y, tot
#%% Task 3
m_edges_list = [4]
n = 12000
repeats = 20

Container = bl.Ball(4.0e50,-10.0,np.array([0.0,0.0]),np.array([0.0,0.0]), 
             Container = True)

data = []
raw_data = []

edges_data = []
raw_edges_data = []
for i in m_edges_list:
    data_rep = []
    data_sum = [0]*(n+i -1)
    edges_rep = []
    edges_sum = [0]*(n+i -1)
    
    for j in range(repeats):
        net_obj = net.Network(Container, m_edges = i, N = n)
        matrix, G, edges_array = net_obj.simulate(False)
        
        edges_rep.append(p_array)
        
        for l in range(n + i  -1):
            edges_sum[l] += edges_array[l]
            
        
        edges = []
        for k in range(n + i -1):
            edges.append(matrix[k].count(1))
            data_sum[k] += edges[k]
            
        
        data_rep.append(np.array(edges))
    raw_data.append(data_rep)
    data.append(data_sum)
    
    raw_edges_data.append(edges_rep)
    edges_data.append(edges_sum)
    
avg_list = []
for i in range(len(m_edges_list)):
    new_list = np.array(edges_data[i])/repeats
    actual_list = list(new_list)
    avg_list.append(actual_list)

        
        
    

    


#%%
    
for i in data:
    plt.plot(np.array(i)/repeats)
plt.xscale('log')
plt.yscale('log')

for i in range(len(prob_data)):
    x, y = logbin(prob_data[i][0])
    plt.plot(x,y)
    plt.plot(np.linspace(1,n + m_edges_list[i] ,n + m_edges_list[i] ),np.array(prob_data[i])/repeats)
m,b = np.polyfit(np.log(np.linspace(1, n +2 , n +2)),np.log(np.array(prob_data[0])/repeats),1)
    
#%% SIMULATION - TESTING - Great for Debugging

Container = bl.Ball(4.0e50,-10.0,np.array([0.0,0.0]),np.array([0.0,0.0]), 
             Container = True)
        
m_edge = 3
a1 = net.Network(Container, m_edges = m_edge, N = 10)
init, G, p_array = a1.simulate(True)

#%% Matrix Method qas too slow so have reprogrammed using a list method

a2 = net.FasterNetwork(m_edges = 3, N = 100000)
vertex_list, edge_list = a2.run()

org_edge = []
for i in vertex_list:
    count = edge_list.count(i) # Takes too much time, need to eliminate
    org_edge.append(1)
    
x,y = logbin(org_edge)
m,b = np.polyfit(np.log(x),np.log(y),1)
#%% NEW CODE V2 - implemented internal counter
start = timer()
a2 = net.FasterNetwork(m_edges = 3, N = 10000000)
vertex_list, edge_list, edges = a2.run()
end = timer()
x,y = logbin(edges)
m,b = np.polyfit(np.log(x),np.log(y),1)

time = start - end # 42.37 seconds

#%% + animation
start = timer()
a2 = net.FasterNetwork(animate = False, m_edges = 3, N = 1000000)
vertex_list, edge_list, edges = a2.run()
end = timer()
x,y = logbin(edges)
m,b = np.polyfit(np.log(x),np.log(y),1)

time = start - end # 44.9 seconds w/ animate == False


#%% + Networkx rep
start = timer()
a2 = net.FasterNetwork(animate = False, m_edges = 3, N = 10000000)
vertex_list, edge_list, edges, edge_connect = a2.run()
end = timer()
x,y = logbin(edges)
m,b = np.polyfit(np.log(x),np.log(y),1)

time = start - end # 67 seconds w/ animate == False - might want to eliminate code responsible fortgis
#%% 
start = timer()
a2 = net.FasterNetwork(animate = False, m_edges = 3, N =10000000, visual = True)
vertex_list, edge_list, edges, edge_connect  = a2.run()
end = timer()
x,y = logbin(edges)
m,b = np.polyfit(np.log(x),np.log(y),1)

time = start - end # 44.6 seconds w/ animate and visual ==  False - might want to eliminate code responsible fortgis
                    # 96 seconds visual = True
def visuals(x, y):
    G = nx.Graph()
    G.add_nodes_from(x)
    G.add_edges_from(y)
    nx.draw(G, node_size=20)
    
    return G

#%% Time analysis + edge analysis
time_req = []
n_sample = [100,1000,10000,100000,1000000, 10000000]
m_edges = [1,2,3,4,5]
edges_list = []
colors = [0,'red', 'green', 'pink','k', 'purple']
markers = ['+','s','1','d','o','h','x']

for i in m_edges:
    print(i)
    for j in n_sample:
        print(j)
        start = timer()
        object_net = net.FasterNetwork(animate = False, m_edges = i, N =j, visual = False)
        vertex_list, edge_list, edges, edge_connect  = object_net.run()
        end= timer()
        time = end- start
        print(time)
        time_req.append(time)
        edges_list.append(edges)

for i in np.linspace(1, len(m_edges),len(m_edges)):
    for j in range(len(n_sample)):
        x,y = logbin(edges_list[int(i)*int(j)])
        plt.plot(x,y, color = colors[int(i)],marker = markers[int(j)] )
    
plt.xscale('log')
plt.yscale('log')
grad = []
for j in range(len(n_sample)):
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    for i in np.linspace(1, len(m_edges),len(m_edges)):
    
        x,y = logbin(edges_list[int(i)*int(j)])
        plt.plot(x,y, color = colors[int(i)],marker = markers[int(j)] )
        m,b = np.polyfit(np.log(x),np.log(y),1)
        grad.append(m)
        
#%% Theoretical Fits


from scipy.stats import chisquare
from scipy import stats
    
n_sample = 50000
m_edges = [2,4,8,16,32, 64]
edges_list = []
colors = ['darkred', 'crimson', 'SeaGreen','DarkGreen','MediumVioletRed','PaleVioletRed', 'k']
markers = ['s','1','d','o','h','x']
repeats = 10
for i in range(len(m_edges)):
    print(i)
    edges_list.append([])
    for j in range(repeats):
        print(j)
        start = timer()
        object_net = net.FasterNetwork(animate = False, m_edges = m_edges[i], N =n_sample, visual = False, random = True)
        vertex_list, edge_list, edges, edge_connect,k  = object_net.run()
        end= timer()
        time = end- start
        print(time)
        edges_list[i].append(edges)
plt.figure()   
residual = []   
x_res = [] 
prob = []
p_ = []
lies = []
for i in range(len(m_edges)):
    lie = np.concatenate(edges_list[i])
    x, y, tot = logbin(lie, i,scale = 1.181)

    plt.plot(x,y, marker = '.', color = colors[int(i)], ls = ' ', label = 'm = {}'.format(m_edges[i]))
    m = m_edges[int(i)]
    k = np.array(x)
    #p_inf = 2*m*(m+1)/(k*(k+1)*(k+2))
    p_inf = m**(np.array(x)-m)/(1+m)**(1+np.array(x)-m)
    plt.plot(x, p_inf, lw = 0.7, ls = '--', color = colors[int(i)])
    g = chisquare(y, f_exp = p_inf)
    print(stats.ks_2samp(y, p_inf))
    #print(g) 
    residual.append((p_inf - y)/p_inf)
    x_res.append(x)
    prob.append(y)
    p_.append(p_inf)
    lies.append(lie)
plt.legend(frameon = False)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('$p_{N}(k)$')
#%% rESIDUAKL 
plt.figure()
for i in range(len(residual)):
    plt.plot(x_res[i], abs(residual[i]*100), marker = '.', ls = ' ', color = colors[i], label = 'm = {}'.format(m_edges[i]))
plt.xlabel('k')
plt.ylabel(r'$\frac{p_{N}(k)}{p_\infty(k)}$ %')

#%%
import statsmodels.api as sm
from matplotlib import pyplot as plt
fig = plt.figure()
for i in range(len(m_edges)):
    res = stats.probplot(lies[i], plot=plt)
    sm.qqplot(redge_list[i], color = colors[i])
    
#%% num =100
a = net.FasterNetwork(animate = False, m_edges = 2, N =10000000, visual = False)
vertex_list, edge_list, edges, edge_connect, k  = a.run()
#%%
import Network as net
m_edges = [2]
n_sample = [ 100,1000,10000,100000]
edges_list = [0]

colors = ['red', 'green', 'pink','k', 'purple', 'orange']
max_degree = []
repeats_ = 5
full_list = []
for i in m_edges:
    print(i)
    for j in n_sample:
        repeats = []
        for k in range(repeats_):
            print(j)
            start = timer()
            object_net = net.FasterNetwork(animate = False, m_edges = i, N =j, visual = False, random = True)
            vertex_list, edge_list, edges, edge_connect, k  = object_net.run()
            end= timer()
            time = end- start
            print(time)
            repeats.append(edges)
        array = np.array([0]*len(repeats[0]))
        for l in range(len(repeats)):
            print(l, 'summing')
            array += np.array(repeats[l])
            print(array)
        edges_list.append(array/repeats_)
            
plt.figure()
for i in np.linspace(1, len(m_edges),len(m_edges)):
    for j in range(len(n_sample)):
        x, y = logbin(list(lies[j]), scale = 1.05)
        plt.plot(x,y, marker = '.')
        m = 2
        k = np.array(x)
        p_inf = 2*float(m)*(float(m)+1)/(k*(k+1)*(k+2))
        
        plt.plot(x, p_inf, color = 'k')
        #g = chisquare(y, f_exp = p_inf)
        #max_degree.append(max(x))

        
plt.xscale('log')
plt.yscale('log')

plt.figure()
for i in np.linspace(1, len(m_edges),len(m_edges)):
    for j in range(len(n_sample)):
        x, y = logbin(C), scale = 1.05)
        m = m_edges[int(i)-1]
        k = np.array(x)
        p_inf = 2*m*(m+1)/(k*(k+1)*(k+2))
        plt.plot(x/max(x), np.array(p_inf)/np.array(p_inf), ls = ' ',color = 'k')
        plt.plot(x/max(x),np.array(y)/np.array(p_inf), ls = ' ',marker ='.')

        #g = chisquare(y, f_exp = p_inf)
        #max_degree.append(max(x))
        #print(stats.kstest(y, p_inf))
        #print(g)
plt.xscale('log')
plt.yscale('log')        
#%%
data_x =[]
data_y =[]
repeats_ = 300 #300
start = timer()

m_edges = [1]
n_sample = [100,1000, 10000,100000,1000000]
edges_list = []
colors = ['red', 'green', 'pink','k', 'purple', 'orange']
markers = ['s','1','d','o','h','x']

for j in range(len(n_sample)):
    edges_list.append([])
    print(n_sample[j])
    for k in range(repeats_):
        print(k)
        object_net = net.FasterNetwork(animate = False, m_edges = 2, N =n_sample[j], visual = False, random = False, mixed = 2/3)
        vertex_list, edge_list, edges, edge_connect, k  = object_net.run()

        edges_list[j].append(edges)
lies = []      
for i in edges_list:
    lie = np.concatenate(i)
    lies.append(lie)
    
plt.figure()
for i in np.linspace(1, len(m_edges),len(m_edges)):
    for j in range(len(n_sample)):
        start = timer()
        x, y, to = logbin(list(lies[j]),i, scale = 1.25)
        end = timer()
        print(end-start)
        plt.plot(x,y, marker = '.', color = colors[j], label = 'N = {}'.format(j), lw = 0.7)
        m = 2
        k = np.array(x)
        p_inf = 2*float(m)*(float(m)+1)/(k*(k+1)*(k+2))
        #p_inf = m**(np.array(x)-m)/(1+m)**(1+np.array(x)-m)
        plt.plot(x, p_inf, color = 'k', ls= '--', lw = 0.7)
        #g = chisquare(y, f_exp = p_inf)
        #max_degree.append(max(x))

        
plt.xscale('log')
plt.yscale('log')

plt.figure()
for i in np.linspace(1, len(m_edges),len(m_edges)):
    for j in range(len(n_sample)):
        x, y = logbin(lies[j], scale = 1.3)
        m = m_edges[int(i)-1]
        k = np.array(x)
        p_inf = 2*m*(m+1)/(k*(k+1)*(k+2))
        plt.plot(x/max(x), np.array(p_inf)/np.array(p_inf), ls = ' ',color = 'k')
        plt.plot(x/max(x),np.array(y)/np.array(p_inf), ls = ' ',marker ='.')

        #g = chisquare(y, f_exp = p_inf)
        #max_degree.append(max(x))
        #print(stats.kstest(y, p_inf))
        #print(g)
plt.xscale('log')
plt.yscale('log') 

    
#%%
datax = []
datay = []
for i in range(len(edges_list)):
    for j in range(repeats_):
            
            x, y = logbin(list(edges_list[i][j]), scale = 1.1)
            
            #p_inf = m**(np.array(x)-m)/(1+m)**(1+np.array(x)-m)
            data_x.append(np.array(x))
            data_y.append(np.array(y))
            plt.plot(x,y,'.', ls = '--')
            #plt.plot(x, p_inf)
    p = 0 
    index  = 0 
    for j in range(len(data_x)):
        if len(data_x[j]) > p:
            p = len(data_x[j])
            index = j
    print(index)
    ydata = [0]*len(data_x[index])
    for k in data_y:
        print(len(k))
        plt.plot(k)
        for l in range(len(k)):
            ydata[l] += k[l]
    datay.append(ydata)
    datax.append(data_x[index])
    data_x = []
    data_y = []

plt.figure()
for i in range(len(datax)):
    plt.plot(datax[i], np.array(datay[i])/repeats_, marker= '.')
    x = datax[i]
    m = 2
    p_inf = m**(np.array(x)-m)/(1+m)**(1+np.array(x)-m)
    plt.plot(x, p_inf)
plt.xscale('log')
plt.yscale('log')
end = timer()

#%% Var 2
import Network as net
plt.figure()
m_edge = 2
n_sample = 10000001
k = []
repeats = 200
x = [10,32, 100,316,1000,3162,10000, 31622, 100000, 316227,1000000, 3162277, 10000000]
#x2 = np.linspace(500, 9500, 19)
#x3 = np.linspace(10000,100000, 10)
logspace = np.linspace(1,7,20)




for i in range(repeats):
    print(i)
    n_obj = net.FasterNetwork(animate = False, m_edges = m_edge, N =n_sample, visual = False, random = True)
    vertex_list, edge_list, edges, edge_connect, k_array  = n_obj.run()
    k.append(k_array)
    
#plt.plot(np.sqrt(np.linspace(1,1000000,100)), k_array)
k_total = np.array([0]*len(k[0]))
for i in k:
    plt.plot(x,i, color = '0.3', lw = 0.7, ls = '--')
    
std_array = []
for j in range(len(k[0])) : 
    std = []
    for i in range(len(k)):
        k_total[j] += k[int(i)][int(j)]
        std.append( k[int(i)][int(j)])
    std_array.append(np.std(std)/200)


averaged = k_total/repeats
plt.errorbar(x, averaged, yerr = std_array, lw = 0.8, capsize = 3, color = 'darkred' , fmt = '^')
plt.plot(x, averaged)
plt.xscale('log')
plt.yscale('log')
    
#%%
start = timer()
for i in range(10000):
    edges[4] += 1
    edges[4] += 1
    edges[4] += 1

end = timer()

#%% Random attatchment
from scipy.stats import chisquare
from scipy import stats
import Network as net

    
n_sample = 50000
m_edges = [2,4,8,16, 32, 64, 128]
edges_list = []
colors = [0,'red', 'green', 'pink','k', 'purple', 'orange']
markers = ['s','1','d','o','h','x']
repeats_ = 250


for i in m_edges:
    list_ed = []
    for j in range(repeats_):
        start = timer()
        object_net = net.FasterNetwork(animate = False, m_edges = i, N = n_sample, visual = False, random = True)
        vertex_list, edge_list, edges, edge_connect, g  = object_net.run()
        end= timer()
        time = end- start
        list_ed.append(edges)
    array = np.array([0]*len(list_ed[0]))
    for l in range(len(list_ed)):
        print(l, 'summing')
        array += np.array(list_ed[l])
    edges_list.append(array/repeats_)   
               
plt.figure()    
  
for i in range(len(edges_list)):
    m = m_edges[i]
    
    x, y = logbin(list(edges_list[i]), scale = 1.01)
    
    p_inf = m**(np.array(x)-m)/(1+m)**(1+np.array(x)-m)
    
    plt.plot(x,y,'.', ls = '--')
    plt.plot(x, p_inf)

plt.xscale('log')
plt.yscale('log')     
#%% random check proper one


from scipy.stats import chisquare
from scipy import stats
    
n_sample = 400000
m_edges = [2,4,8,16, 32, 64]
edges_list = []
colors = ['darkred', 'crimson', 'SeaGreen','DarkGreen','MediumVioletRed','PaleVioletRed', 'k']
markers = ['s','1','d','o','h','x']
repeats = 100
for i in range(len(m_edges)):
    print(i)
    edges_list.append([])
    for j in range(repeats):
        print(j)
        start = timer()
        object_net = net.FasterNetwork(animate = False, m_edges = m_edges[i], N =n_sample, visual = False, random = True)
        vertex_list, edge_list, edges, edge_connect,k  = object_net.run()
        end= timer()
        time = end- start
        edges_list[i].append(edges)
plt.figure()  
scaling = 1.18
residual = []
x_res = []
prob = []
p_ = []     
standard = []
for i in range(len(m_edges)):
    lie = np.concatenate(edges_list[i])
    #x, y, tot = logbin(lie,i, scale = scaling)
    x = x_logbin[i]
    y = y_logbin[i]
    plt.plot(x,y, marker = '.', color = colors[int(i)], ls = ' ', label = 'm = {}'.format(m_edges[i]))
    m = m_edges[int(i)]
    k = np.array(x)
    bottom = ((1+m)*(1+m)**(np.array(x)))
    top =  ((m**(np.array(x)-m))*(1+m)**m)
    print(top, bottom)
    p_inf = (1/(m+1))*(m/(m+1))**(k-m)
    plt.plot(x, p_inf, lw = 0.7, ls = '--', color = colors[int(i)])
    g = chisquare(y, f_exp = p_inf)
    print(stats.kstest(y, p_inf))
    print(g) 
    residual.append((p_inf - y)/p_inf)
    x_res.append(x)
    prob.append(y)
    p_.append(p_inf)
    #standard.append(count)
plt.legend(frameon = False)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('$p_{N}(k)$')
y_list_ = []
sizes = []
for i in range(len(m_edges)):
    print(i)
    y_list = []
    max_ = 0
    for j in range(100):
        x, y , tot= logbin(list(edges_list[i][j]), i,scale= scaling)
        y_list.append(y)
        if len(x) > max_:
            max_ = len(x)
    sizes.append(max_)
    y_list_.append(y_list)

total_list = []
for i in range(len(m_edges)):
    list_num = []
    for k in range(sizes[i]):
        list_num.append([])
    for j in range(100):
        for k in range(len(y_list_[i][j])):
            list_num[k].append(y_list_[i][j][k])
    total_list.append(list_num)
    

std__ = []
for i in range(len(m_edges)):
    std__.append([])
    for j in range(sizes[i]):
        std__[i].append(np.std(total_list[i][j]))
plt.figure()  
import matplotlib.pyplot as plt
from matplotlib import gridspec

fig = plt.figure(figsize=(9, 6))
main = fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.xlabel('$k$')
plt.ylabel('$p(k)$')
main_ax = fig.add_subplot(121)

sub = fig.add_subplot(122, frameon = False)
plt.ylabel(r'$\sigma_{p}/p(k)$ (%)')

sub.tick_params(labelcolor="none", bottom=False, left=False)

a = fig.add_subplot(622)
a.tick_params(labelsize = 7)

b = fig.add_subplot(624)
b.tick_params(labelsize = 7)

c = fig.add_subplot(626)
c.tick_params(labelsize = 7)

d = fig.add_subplot(628)
d.tick_params(labelsize = 7)

e = fig.add_subplot(6,2,10)
e.tick_params(labelsize = 7)

f = fig.add_subplot(6,2,12)
f.tick_params(labelsize = 7)



for i in [0,1,2,3,4,5]:
    lie = np.concatenate(edges_list[i])
    x, y = logbin(lie, scale = scaling)


   

    if len(y) > len(std__[i]):
        z = y[:len(std__[i])]
        z1 = x[:len(std__[i])]
        y = z
        x = z1
    else:
        z = std__[i][:len(y)]
        std__[i] = z
    m = m_edges[int(i)]
    k = np.array(x)

    #p_inf = (1/m) * ((m/(m+1))**(k-m))
    p_inf = m**(np.array(x)-m)/(1+m)**(1+np.array(x)-m)
    main_ax.plot(x, p_inf, lw = 0.7, ls = '--', color = colors[int(i)])

    #p_inf = (m**(np.array(x)-m))/((1+m)**(1+np.array(x)-m))
    g = chisquare(y, f_exp = p_inf)
    #print(stats.kstest(y, p_inf))
    #print(g) 
    print(stats.ks_2samp(y, p_inf))
    m, b_ = np.polyfit(x, np.log(y), 1)
    print(m)
    main_ax.errorbar(x, y, yerr =np.array(std__[i]), color = colors[int(i)],marker = '.', ls = ' ', capsize = 2, lw = 0.5, label='m = {}'.format(m_edges[i])) 
    if i ==0:
        a.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/y)*100,color = colors[int(i)], ls = ' ', capsize = 2)
    elif i ==1:
        b.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/y)*100,color = colors[int(i)], ls = ' ', capsize = 2)
       
    elif i ==2:        
        c.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/y)*100,color = colors[int(i)], ls = ' ', capsize = 2)
    elif i ==3:        
        #d.errorbar(x, [0]*len(x),yerr =((np.array(std__[i])**2)/y)*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(m_edges[i]))
        d.errorbar(x, [0]*len(x),yerr =(np.array(std__[i])/y)*100,color = colors[int(i)], ls = ' ', capsize = 2)

    elif i ==4:        
        e.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/y)*100,color = colors[int(i)], ls = ' ', capsize = 2)
    elif i ==5:        
        f.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/y)*100,color = colors[int(i)], ls = ' ', capsize = 2)

      
    main_ax.set_yscale('log')    

    #main_ax.plot(x,y, marker = '.', color = colors[int(i)], ls = ' ', label = 'm = {}'.format(m_edges[i]))
main_ax.legend(frameon = False)

    
#%%
import matplotlib.pyplot as plt
from matplotlib import gridspec

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

#%% rESIDUAKL 
plt.figure()
for i in range(len(residual)):
    plt.plot(x_res[i], abs(residual[i]*100), marker = '.', ls = ' ', color = colors[i], label = 'm = {}'.format(m_edges[i]))
plt.xlabel('k')
plt.ylabel(r'$Residual$ (%)')

#%% Repeating after logbin

data_x =[]
data_y =[]
repeats_ = 100
start = timer()

n_sample = 400000
m_edges = [2,4,8,16, 32, 64]
edges_list = []
colors = [0,'red', 'green', 'pink','k', 'purple', 'orange']
markers = ['s','1','d','o','h','x']
for i in range(len(m_edges)):
    edges_list.append([])
    for j in range(repeats_):
        object_net = net.FasterNetwork(animate = False, m_edges = m_edges[i], N = n_sample, visual = False, random = False, mixed = 2/3)
        vertex_list, edge_list, edges, edge_connect, g  = object_net.run()

        edges_list[i].append(edges)
        

datax = []
datay = []
for i in range(len(edges_list)):
    for j in range(repeats_):
            
            x, y = logbin(list(edges_list[i][j]), scale = 1.05)
            
            #p_inf = m**(np.array(x)-m)/(1+m)**(1+np.array(x)-m)
            data_x.append(np.array(x))
            data_y.append(np.array(y))
            plt.plot(x,y,'.', ls = '--')
            #plt.plot(x, p_inf)
    p = 0 
    index  = 0 
    for j in range(len(data_x)):
        if len(data_x[j]) > p:
            p = len(data_x[j])
            index = j
    print(index)
    ydata = [0]*len(data_x[index])
    for k in data_y:
        print(len(k))
        plt.plot(k)
        for l in range(len(k)):
            ydata[l] += k[l]
    datay.append(ydata)
    datax.append(data_x[index])
    data_x = []
    data_y = []

plt.figure()
for i in range(len(datax)):
    plt.plot(datax[i], np.array(datay[i])/repeats_, marker= '.')
    x = datax[i]
    m = m_edges[i]
    p_inf = m**(np.array(x)-m)/(1+m)**(1+np.array(x)-m)
    plt.plot(x, p_inf)
plt.xscale('log')
plt.yscale('log')
end = timer()

#%%
import Network as net
plt.figure()
m_edge = 2
n_sample = 1000001
k = []
repeats = 200
x = [10,32, 100,316,1000,3162,10000, 31622, 100000, 316227,1000000]
#x2 = np.linspace(500, 9500, 19)
#x3 = np.linspace(10000,100000, 10)
logspace = np.linspace(1,7,20)




for i in range(repeats):
    print(i)
    n_obj = net.FasterNetwork(animate = False, m_edges = m_edge, N =n_sample, visual = False, random = False, mixed = 2/3)
    vertex_list, edge_list, edges, edge_connect, large_k = n_obj.run()
    k.append(large_k)
plt.figure()
#plt.plot(np.sqrt(np.linspace(1,1000000,100)), k_array)
k_total = np.array([0]*len(k[0]))
plt.figure()
for i in k:
    plt.plot(x,i, color = '0.7', lw = 0.3, ls = '--')
plt.plot(x,k[1], color = '0.7', lw = 0.3, ls = '--', label = 'Repeated Runs')
std_array = []
for j in range(len(k[0])) : 
    std = []
    for i in range(len(k)):
        k_total[j] += k[int(i)][int(j)]
        std.append( k[int(i)][int(j)])
    std_array.append(np.std(std))


averaged = k_total/repeats
y_theory  = m_edge + (-1/(np.log((m_edge/(m_edge+1)))))*np.log(x)
y_theory = (-1 + (4*np.array(x)*m_edge*(1 + m_edge) + 1)**(1/2))*0.5
plt.plot(x, y_theory, color = 'forestgreen',ls = '--', label = ' Theoretical Fit - PA')
plt.errorbar(x, averaged, yerr = std_array, lw = 0.8, capsize = 3, color = 'darkred' , fmt = '.',ls = ' ',marker ='.',fillstyle = 'none', markersize = 12, label = r'$Averaged \quad k_1 $ ')
m, b = np.polyfit(np.log(x[-6:]),np.log(averaged[-6:]),1, cov = True)
#plt.plot(np.exp([10000,10000000]), np.exp(m[0]*np.array([10000,10000000]) + m[1]), color = 'k')
y_theory  = m_edge + (-1/(np.log((m_edge/(m_edge+1)))))*np.log(x)
plt.plot(x, y_theory, color = 'darkblue',ls = '--', label = ' Theoretical Fit - RND')



plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$N$', fontsize = 15)
plt.ylabel(r'$\langle k_1 \rangle$ ', fontsize = 15)

print(m, np.sqrt(np.diag((b))))
#%% 
n_sample = [100,1000,10000,100000,1000000]
edges_list = []
for i in range(len(n_sample)):
    start = timer()
    edges_list.append(np.loadtxt('degrees_N_testing_data{}.csv'.format(i), delimiter=','))
    end = timer()
    print(end-start)


#%%

edges_list = []
for i in range(len(n_sample)):
    start = timer()
    edges_list.append(np.loadtxt('varying_N_300REP_ba_{}.csv'.format(i), delimiter=','))
    end = timer()
    print(end-start)
#%%
edges_list = np.loadtxt('400000_100_rnd.csv',delimiter=',')

#%%
plt.figure()
diff_avg_edge = []
for i in range(5):        
    array = np.array([0.0]*(len(edges_list[i][0])))
    for l in range(300):
        
        array += np.array(edges_list[i][l])
        len(edges_list[i][j])
    diff_avg_edge.append(array/300)
for i in diff_avg_edge:
    x, y = logbin(list(i), i, scale = scaling)
    plt.plot(x, y,'.')

#%%


from scipy.stats import chisquare
from scipy import stats
plt.figure()  
scaling = 1.2 #1.03
residual = []
colors = ['darkred', 'crimson', 'SeaGreen','DarkGreen','MediumVioletRed','PaleVioletRed', 'k']
m_edges = 2
x_res = []
prob = []
p_ = []     
standard = []
for i in range(len(n_sample)):  
    print(i)
    start = timer()
    #lie = np.concatenate(edges_list[i])
    x, y = logbin(list(lies[i]),i, scale = scaling)

    plt.plot(x,y, marker = '.', color = colors[int(i)], label = 'N = {}'.format(n_sample[i]))
    m = 2
    k = np.array(x)
    p_inf = (m**(np.array(x)-m))/((1+m)**(1+np.array(x)-m))
   
    plt.plot(x, p_inf, lw = 0.7, ls = '--', color = colors[int(i)])
    #g = chisquare(y, f_exp = p_inf)
    #print(stats.kstest(y, p_inf))
    #print('two',stats.ks_2samp(y, p_inf))
    residual.append((p_inf - y)/p_inf)
    x_res.append(x)
    prob.append(y)
    p_.append(p_inf)
    end = timer()
    print(end- start)
plt.legend(frameon = False)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('$p_{N}(k)$')
y_list_ = []
sizes = []
for i in range(len(n_sample)):
    print(i)

    y_list = []
    max_ = 0
    for j in range(300):
        x, y = logbin(list(edges_list[i][j]), i,scale= scaling)
        y_list.append(y)
        if len(x) > max_:
            max_ = len(x)
    sizes.append(max_)
    y_list_.append(y_list)

total_list = []
for i in range(len(n_sample)):
    list_num = []
    for k in range(sizes[i]+10):
        list_num.append([])
    for j in range(300):
        for k in range(len(y_list_[i][j])):
            list_num[k].append(y_list_[i][j][k])
    total_list.append(list_num)
    

std__ = []
for i in range(len(n_sample)):
    std__.append([])
    for j in range(sizes[i]):
        std__[i].append(np.std(total_list[i][j])/np.sqrt(300))
plt.figure()  
import matplotlib.pyplot as plt
from matplotlib import gridspec
#%%

fig = plt.figure(figsize=(9, 6))
main = fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
colors = ['purple', 'forestgreen','blue', 'orange', 'red','k']
plt.xlabel('k', fontsize = 15)
plt.ylabel('p(k)', fontsize = 13)
main_ax = fig.add_subplot(121)
sub = fig.add_subplot(122, frameon = False)
plt.ylabel(r'$ \sigma_\overline{p}/ p(k) \quad  \% $', fontsize = 13)
sub.tick_params(labelcolor="none", bottom=False, left=False)

a = fig.add_subplot(622)
a.tick_params(labelsize = 7)

b = fig.add_subplot(624)
b.tick_params(labelsize = 7)

c = fig.add_subplot(626)
c.tick_params(labelsize = 7)

d = fig.add_subplot(628)
d.tick_params(labelsize = 7)



e = fig.add_subplot(6,2,10)
e.tick_params(labelsize = 7)
f = fig.add_subplot(6,2,12)
f.tick_params(labelsize = 7)
std__ = std_
residual = []
from scipy import stats

for i in [0,1,2,3,4,5]:
    #lie = np.concatenate(edges_list[i])
    #x, y, tot = logbin(list(lies[i]), i,scale = scaling)

   
    x = x_logbin[i]
    y = y_logbin[i]
    #for j in range(len(std__[i])):
    #    if std__[i][j] == 0:
    #        std__[i][j] = std__[i][j-2]
            
            
            
   # while len(y) > len(std__[i]):
    #    std__[i].append(std__[i][-1])

    #while len(std__[i]) > len(y):
    #    z = std__[i][:len(y)]
    #    std__[i] = z
    m = 2
    k = np.array(x)
    p_inf = 2*float(m)*(float(m)+1)/(k*(k+1)*(k+2))
    #p_inf = (m**(np.array(x)-m))/((1+m)**(1+np.array(x)-m))
    #g = chisquare(y, f_exp = p_inf)
    print(stats.ks_2samp(y, p_inf))
    residual.append(abs(y-p_inf)/p_inf)

    #print(g) 
    main_ax.plot(x, p_inf, lw = 0.7, ls = '--', color = colors[int(i)], fillstyle = None)

    #m, b_ = np.polyfit(x, np.log(y), 1)
    #print(m)
    main_ax.errorbar(x, y, yerr = np.array(std__[i]/(np.sqrt(300))), fillstyle = 'none', markersize = 12,color = colors[i] ,marker = '.', ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i]))
    if i ==0:
        a.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/(y*np.sqrt(300)))*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i])) 
        a.set_yscale('log')
        #a.set_xscale('log')
    elif i ==1:
        b.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/(y*np.sqrt(300)))*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i])) 
        b.set_yscale('log')        
        #b.set_xscale('log')

    elif i ==2:        
        c.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/(y*np.sqrt(300)))*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i]))
        c.set_yscale('log')
        #c.set_xscale('log')

    elif i ==3:        
        d.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/(y*np.sqrt(300)))*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i])) 
        d.set_yscale('log')
        #d.set_xscale('log')

    elif i ==4:        
        e.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/(y*np.sqrt(300)))*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i])) 
        e.set_yscale('log')
        #e.set_xscale('log')

    elif i ==5:        
        f.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/(y*np.sqrt(300)))*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i])) 
        f.set_yscale('log')
        #f.set_xscale('log')

    main_ax.set_yscale('log')    
    main_ax.set_xscale('log')    

    #main_ax.plot(x,y, marker = '.', color = colors[int(i)], ls = ' ', label = 'm = {}'.format(m_edges[i]))

main_ax.legend(frameon = False)
#%%

fig = plt.figure(figsize=(9, 6))
main = fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
colors = ['purple', 'forestgreen','blue', 'orange', 'red']
plt.xlabel('k', fontsize = 15)
plt.ylabel('p(k)', fontsize = 13)
main_ax = fig.add_subplot(121)
sub = fig.add_subplot(122, frameon = False)
plt.ylabel(r'$ \sigma_\overline{p}/ p(k) \quad  \% $', fontsize = 13)
sub.tick_params(labelcolor="none", bottom=False, left=False)

a = fig.add_subplot(522)
a.tick_params(labelsize = 7)

b = fig.add_subplot(524)
b.tick_params(labelsize = 7)

c = fig.add_subplot(526)
c.tick_params(labelsize = 7)

d = fig.add_subplot(528)
d.tick_params(labelsize = 7)



e = fig.add_subplot(5,2,10)
e.tick_params(labelsize = 7)
std__ = std_
residual = []
from scipy import stats

for i in [0,1,2]:
    #lie = np.concatenate(edges_list[i])
    #x, y, tot = logbin(list(lies[i]), i,scale = scaling)

   
    x = x_logbin[i]
    y = y_logbin[i]
    #for j in range(len(std__[i])):
    #    if std__[i][j] == 0:
    #        std__[i][j] = std__[i][j-2]
            
            
            
   # while len(y) > len(std__[i]):
    #    std__[i].append(std__[i][-1])

    #while len(std__[i]) > len(y):
    #    z = std__[i][:len(y)]
    #    std__[i] = z
    m = m_edges[i]
    k = np.array(x)
    #p_inf = (1/m) * ((m/(m+1))**(k-m))

    p_inf = (m**(np.array(x)-m))/((1+m)**(1+np.array(x)-m))
    #_inf = 2*float(m)*(float(m)+1)/(k*(k+1)*(k+2))
    #g = chisquare(y, f_exp = p_inf)
    print(stats.ks_2samp(y, p_inf))
    residual.append(abs(y-p_inf)/p_inf)

    #print(g) 
    main_ax.plot(x, p_inf, lw = 0.7, ls = '--', color = colors[int(i)], fillstyle = None)

    #m, b_ = np.polyfit(x, np.log(y), 1)
    #print(m)
    main_ax.errorbar(x, y, yerr = np.array(std__[i]/(np.sqrt(300))), fillstyle = 'none', markersize = 12,color = colors[i] ,marker = '.', ls = ' ', capsize = 2,label = 'N = {}'.format(n_sample[i]))
    if i ==0:
        a.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/(y*np.sqrt(300)))*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i])) 
        #a.set_yscale('log')
        #a.set_xscale('log')
    elif i ==1:
        b.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/(y*np.sqrt(300)))*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i])) 
        #b.set_yscale('log')        
        #b.set_xscale('log')

    elif i ==2:        
        c.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/(y*np.sqrt(300)))*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i])) 
        #c.set_yscale('log')
        #c.set_xscale('log')

    elif i ==3:        
        d.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/(y*np.sqrt(300)))*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i])) 
        #d.set_yscale('log')
        #d.set_xscale('log')

    elif i ==4:        
        e.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/(y*np.sqrt(300)))*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i])) 
        #e.set_yscale('log')
        #e.set_xscale('log')

    elif i ==6:        
        f.errorbar(x, [0]*len(x),yerr =((np.array(std__[i]))/(y*np.sqrt(300)))*100,color = colors[int(i)], ls = ' ', capsize = 2,label = 'm = {}'.format(n_sample[i])) 
        #f.set_yscale('log')
        #f.set_xscale('log')

    main_ax.set_yscale('log')    
    #main_ax.set_xscale('log')   
    #main_ax.plot(x,y, marker = '.', color = colors[int(i)], ls = ' ', label = 'm = {}'.format(m_edges[i]))

main_ax.legend(frameon = False)
#%%
plt.figure()
for i in range(len(residual)):
    plt.plot(x_logbin[i], abs(residual[i]*100),fillstyle = 'none', markersize = 12,marker = '.', ls = ' ', color = colors[i], label = 'm = {}'.format(m_edges[i]))
plt.xlabel(r'$k$', fontsize = 15)
plt.ylabel(r'$Residual \quad \% $', fontsize = 15)
plt.xscale('log')
plt.yscale('log')
#%%

plt.figure()
for j in range(len(n_sample)):
    x = x_logbin[j]
    y = y_logbin[j]
    
    m = 2
    k = np.array(x)
    p_inf = (m**(np.array(x)-m))/((1+m)**(1+np.array(x)-m))
    p_inf = (6*m*(2*m+1)*(2*m+2))/((k+m)*(k+m+1)*(k+m+2)*(k+m+3))
    plt.plot(x/max(x),np.array(y)/np.array(p_inf), color = colors[j],label ='N = {}'.format(n_sample[j]),ls = ' ',marker ='.',fillstyle = 'none', markersize = 7)

    if j ==4:
        plt.plot(x/max(x), np.array(p_inf)/np.array(p_inf), label ='Theoretical Fit',fillstyle = 'none',marker = '.', markersize = 7,ls = ' ',color = 'k')

        #g = chisquare(y, f_exp = p_inf)
        #max_degree.append(max(x))
        #print(stats.kstest(y, p_inf))
        #print(g)
plt.xscale('log')
plt.yscale('log') 

plt.ylabel(r'$ p(k)/p_\infty (k) $', fontsize = 15)
plt.xlabel(r'$ k/k_1 $', fontsize = 15)
plt.legend(frameon = False)

#%%
scaling = 1.15 #1.181 phae 1-m
lies = []    
sorted_list = []
total =[]
x_logbin = []
y_logbin = []
for i in edges_list:
    lie = np.concatenate(i)
    lies.append(lie)
for i in lies:
    sorted_list.append(i.sort())
for i in edges_list:
    lie = np.concatenate(i)
    lies.append(lie)
x_ = []
plt.figure()
for i in range(5):
    x, y, tot = logbin(list(lies[i]), i,scale = scaling)
    x_logbin.append(x)
    y_logbin.append(y)
    plt.plot(x,y,'.')
    x_.append(x)
    total.append(tot)
seperate_bins = []
for i in range(len(x_)):
    start__ = timer()
    print(i)
    seperate_bins.append([])
    start = 0
    finish = len(lies[i])
    for j in range(len(x_[i])):
        print(j)
        seperate_bins[i].append([])
        print('s',start)
        new_start = 0
        for k in range(start, finish):
            if j == 0:
                if lies[i][k] >= 0 and lies[i][k] <x_[i][j]:
                    seperate_bins[i][j].append(lies[i][k]/(x_[i][j]))
            else:
                if lies[i][k] >= x_[i][j-1] and lies[i][k]  < x_[i][j]:
                    seperate_bins[i][j].append(lies[i][k]/(x_[i][j] - x_[i][j-1]))
                    new_start = k
        start = new_start
    end__ = timer()
    print(end__-start__)
std_ = []
for i in range(len(seperate_bins[:5])):
    std_.append([])
    for j in range(len(seperate_bins[i])):
        std_[i].append((np.std(seperate_bins[i][j])/total[i]))


error = []
plt.figure()
for i in [0,1,2]:
    print(i)
    #x, y, tot = logbin(list(lies[i]), i,scale = scaling)
    #xerr, yerr_ = logbin(list(np.concatenate(seperate_bins[i])),i, scale = scaling)
    #lenth = len(std_[i])
    #print(x)
    #plt.errorbar(x[i],y[i],yerr = np.array(std_[i])/np.sqrt(300))
    plt.plot(x_logbin[i],y_logbin[i],'.', color = colors[i], label = '{}'.format(i))
    
        
plt.xscale('log')
plt.yscale('log')

#%%
scaling = 1.181 #1.15 for random 1.2 for ba - data collaps stuff
lies = []    
sorted_list = []
total =[]
x_logbin = []
y_logbin = []
for i in edges_list:
    lie = np.concatenate(i)
    lies.append(lie)
for i in lies:
    sorted_list.append(i.sort())
for i in edges_list:
    lie = np.concatenate(i)
    lies.append(lie)
x_ = []
plt.figure()
for i in range(6):
    x, y, tot = logbin(list(lies[i]), i,scale = scaling)
    x_logbin.append(x)
    y_logbin.append(y)
    plt.plot(x,y,'.')
    x_.append(x)
    total.append(tot)
seperate_bins = []
for i in range(len(x_)):
    start__ = timer()
    print(i)
    seperate_bins.append([])
    start = 0
    finish = len(lies[i])
    for j in range(len(x_[i])):
        print(j)
        seperate_bins[i].append([])
        print('s',start)
        new_start = 0
        for k in range(start, finish):
            if j == 0:
                if lies[i][k] >= 0 and lies[i][k] <x_[i][j]:
                    seperate_bins[i][j].append(lies[i][k]/(x_[i][j]))
            else:
                if lies[i][k] >= x_[i][j-1] and lies[i][k]  < x_[i][j]:
                    seperate_bins[i][j].append(lies[i][k]/(x_[i][j] - x_[i][j-1]))
                    new_start = k
        start = new_start
    end__ = timer()
    print(end__-start__)

                    

#%% Random Walk 
n_sample = 1000000
q = [2/3]
repeats = 300
m_edges = 2
edges_list = []
for i in range(len(q)):
    print(i)
    edges_list.append([])
    for j in range(repeats):
        print(j)
        object_net = net.FasterNetwork(animate = False, m_edges = 2, N =n_sample, visual = False, mixed = q[i])
        vertex_list, edge_list, edges, edge_connect,k  = object_net.run()
        edges_list[i].append(edges)
        

scaling = 1.181 #1.15 for random 1.2 for ba - data collaps stuff
lies = []    
sorted_list = []
total =[]
x_logbin = []
y_logbin = []
for i in edges_list:
    lie = np.concatenate(i)
    lies.append(lie)
for i in lies:
    sorted_list.append(i.sort())  


x_ = []
plt.figure()
for i in range(len(lies)):
    x, y, tot = logbin(list(lies[i]), i,scale = scaling)
    x_logbin.append(x)
    y_logbin.append(y)
    plt.plot(x,y,'.')
    x_.append(x)
    total.append(tot)
seperate_bins = []
for i in range(len(x_)):
    start__ = timer()
    print(i)
    seperate_bins.append([])
    start = 0
    finish = len(lies[i])
    for j in range(len(x_[i])):
        print(j)
        seperate_bins[i].append([])
        print('s',start)
        new_start = 0
        for k in range(start, finish):
            if j == 0:
                if lies[i][k] >= 0 and lies[i][k] <x_[i][j]:
                    seperate_bins[i][j].append(lies[i][k]/(x_[i][j]))
            else:
                if lies[i][k] >= x_[i][j-1] and lies[i][k]  < x_[i][j]:
                    seperate_bins[i][j].append(lies[i][k]/(x_[i][j] - x_[i][j-1]))
                    new_start = k
        start = new_start
    end__ = timer()
    print(end__-start__)
#%%
plt.close('all')
fig = plt.figure(figsize=(9, 6))
main = fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
colors = ['purple', 'forestgreen','blue', 'orange', 'red']
plt.xlabel('k', fontsize = 15)
plt.ylabel('p(k)', fontsize = 13)
main_ax = fig.add_subplot(121)
sub = fig.add_subplot(122, frameon = False)
plt.ylabel(r'$ \sigma_\overline{p}/ p(k) \quad  \% $', fontsize = 13)
sub.tick_params(labelcolor="none", bottom=False, left=False)

a = fig.add_subplot(322)
a.tick_params(labelsize = 7)

b = fig.add_subplot(324)
b.tick_params(labelsize = 7)
c = fig.add_subplot(326)
c.tick_params(labelsize = 7)
"""
d = fig.add_subplot(528)
d.tick_params(labelsize = 7)



e = fig.add_subplot(5,2,10)
e.tick_params(labelsize = 7)
"""
residual = []
q = [str('0'),'2/3', '1']
from scipy import stats

for i in [0,1,2,3,4,5]:
    #lie = np.concatenate(edges_list[i])
    #x, y, tot = logbin(list(lies[i]), i,scale = scaling)
   
    x = x_logbin[i]
    y = y_logbin[i]
    #for j in range(len(std__[i])):
    #    if std__[i][j] == 0:
    #        std__[i][j] = std__[i][j-2]
            
            
            
   # while len(y) > len(std__[i]):
    #    std__[i].append(std__[i][-1])

    #while len(std__[i]) > len(y):
    #    z = std__[i][:len(y)]
    #    std__[i] = z
    m = 2
    k = np.array(x)
    p_inf = (m**(np.array(x)-m))/((1+m)**(1+np.array(x)-m))

    print(stats.ks_2samp(y, p_inf))

    main_ax.set_yscale('log')    
    main_ax.set_xscale('log')   
    #main_ax.plot(x,y, marker = '.', color = colors[int(i)], ls = ' ', label = 'm = {}'.format(m_edges[i]))
    residual.append(abs(y-p_inf)/p_inf)
    
    
main_ax.legend(frameon = False)