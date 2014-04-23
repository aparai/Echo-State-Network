# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:48:03 2014

@author: parai
"""

from esn_class import esn_design, esn_prediction
import MySQLdb as mdb
import matplotlib.pyplot as plt
import numpy as np

conn = mdb.connect('localhost', 'root', 'iitkgp12', 'Ed_Kiers')
c=conn.cursor()
c.execute("SELECT * FROM week WHERE the_day=1")
data = c.fetchall()
k=[]
for ind in range(len(data))[1:]:
    k.append(1 - data[ind][3]*1.0/data[ind-1][3])

data_tr=[]
out_tr = []
N = 30

l=[]
for ind in range(N):
    l.append(k[ind])   

for ind in range(len(k))[N+1:]:
    data_tr.append(l)
    out_tr.append(k[ind])
    l = l[1:]
    l.append(k[ind])

data_pred = []
out_pred = []  
    
c.execute("SELECT * FROM week WHERE the_day=2")
data = c.fetchall()
k2=[]
l = []
   
for ind in range(len(data))[1:]:
    k2.append(1 - data[ind][3]/data[ind-1][3])
    
for ind in range(N):
    l.append(k2[ind])
    
for ind in range(len(k2))[N+1:]:
    data_pred.append(l)
    out_pred.append(k2[ind])
    l = l[1:]
    l.append(k2[ind])

u =[]
y = []
l = []
for ind in range(100):
    l.append(ind)
    l.append(ind+1)
    u.append(l)
    l = []
    y.append(ind+2)

esn_instance = esn_design()
esn_instance.esn_training(data_tr, out_tr)


p = esn_prediction(esn_instance)

g = []
for ind in range(len(data_pred)):
    g.append(p.predict(data_pred[ind]).item((0,0)))

g_final = []
out_p = []
n = 0
for ind in range(len(g)):
    if(ind==0):
        g_final.append(g[ind])
        out_p.append(out_pred[ind])
    else:
        g_final.append(g[ind]+g[-1])
        out_p.append(out_p[-1] + out_pred[-1])
    
    if(g[ind]*1.0/out_pred[ind] >= 0):
        n +=1

 
print n*1.0/len(g)       

#plt.plot(range(len(out_pred)), g, range(len(out_pred)), out_pred)
#plt.show()
    


    