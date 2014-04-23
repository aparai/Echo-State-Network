# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 00:04:42 2014

@author: parai
"""
import numpy as np

from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
import random
import math


class esn_design(object):
    
    def __init__(self, K=30, N=100, L=1, pr=5, alpha=0.9, eig=0.95, win_up=3, sol=1):
        
        self.K = int(K)+1  # Added 1 for constant input
        self.N = int(N)  
        self.L = int(L)
        if (pr < 100):
            self.pr = (100-pr)*1.0/100
        else:
            self.pr = 0.95
            print "Sparse fraction must be less than 100! Setting default of 5%\n"
            
        self.alpha = alpha
        self.sol_type = 1 # 1 = Ridge regression, 2 = Pseudo Inverse Solution
        if(sol==1 or sol==2):
            self.sol_type = sol
        else:
            print "Illegitimate training type: use 1 for Ridge Regression, and 2 for Pseudo Inverse Method\n"
           
            
        if(eig < 1):
            self.eig = eig
        else:
            self.eig = 0.95
            print "Sorry! The eigenvalue (variable 'eig') must be less than 1!\n Created class instance with default target eigenvalue of 0.95!\n"
        
        self.Win = np.matrix((np.random.random((self.N, self.K)))*win_up*1.0)
        self.W = np.matrix(np.zeros((self.N, self.N)))
        self.X = 0
        self.Y = 0
        self.Wout = 0        
        self.status = 0
        
        self.Wsparse()
    
    def Wsparse(self):        
        print "Applying sparsity condition on the reservoir matrix. This may take a while!\n\n"        
        target_ele = int(self.N*self.N*self.pr)        
        l = []
        for ind in xrange(target_ele):
            chk = 1
            while chk==1:
                a = random.randint(0, self.N-1)
                b = random.randint(0, self.N-1)
                new_tuple = (a, b)
                if new_tuple not in l: # making sure indices are not repeated!
                    np.put(self.W, [new_tuple], [np.random.random(1)[0]])
                    l.append(new_tuple)
                    chk = 0           
        print "Sparsity condition successfully applied!\n\n"
        self.Wnormalization() # now normalize this matrix
    
    # scaling W such that eigenvalue is less than 1!     
    def Wnormalization(self):   
        w, v = np.linalg.eig(self.W)
        the_big_eig = max(w)
        self.W = (1.0/the_big_eig)*(self.eig)*self.W
    
    def esn_x(self, U):        
        x_init = np.matrix(np.ones(self.N))
        
        for ind in xrange(len(U)):
            u = U[ind]
            u.insert(0, 1) # Adding constant input
            u = (np.matrix(u)).T
            a = self.Win*u
            b = self.W*(x_init.T)
            res = []
            num = a.shape[0]
            for count in xrange(num):
                res.append(math.tanh(a.item(count, 0) + b.item(count, 0)))
            c = np.matrix(res).T
            # Now update with alpha
            c_updated = self.alpha*c + (1 - self.alpha)*(x_init.T)
            if(ind==0):
                start_X = np.concatenate((u, c_updated), axis=0)
                self.X = start_X
            else:
                start_X = np.concatenate((u, c_updated), axis=0)
                self.X = np.concatenate((self.X, start_X), axis=1)
    
    def esn_training(self, U, Y_target):
        
        # First prepare the X matrix
        self.esn_x(U)
        # Now start the training
        self.X[8]
        
        
        for ind in xrange(len(Y_target)):
            y = np.matrix(np.array(Y_target[ind])).T
            if(ind==0):
                self.Y = y
            else:
                self.Y = np.concatenate((self.Y, y), axis=1)
        # So we have all the ingredients now to solve
        
        # We are using Ridge Regression for type 1
        if(self.sol_type==1):
            self.Wout = (self.Y)*((self.X).T)*np.linalg.inv((self.X*((self.X).T) + 1.0*np.matrix(np.identity(self.N + self.K))))
        
        """
        Moore Penrose Pseudo Inverse Method - careful: the system must be 
        overdetermined. In other words (K + N) << number of samples!
        """
        if(self.sol_type==2):
            X_mp = np.linalg.pinv(self.X)
            self.Wout = self.Y*X_mp
        self.status = 1


"""
We are done with the training, now let's do
prediction. You can create RPC server to so the same in
convenient manner!
"""
    
class esn_prediction(object):
    
    def __init__(self, esn_object): # pass the training instance created above here
        self.Win = esn_object.Win
        self.W = esn_object.W
        self.Wout = esn_object.Wout
        self.alpha = esn_object.alpha
        win_size = self.W.shape
        self.x = np.matrix(np.ones(win_size[1]))
        self.Yout = 0
        w, v = np.linalg.eig(self.W)
        print max(w)
        
    def predict(self, u):
        
        # update x variable first
        u.insert(0,1) # Add constant of 1 to the input
        u = (np.matrix(np.array(u))).T
        a = self.Win*u        
        b = self.W*((self.x).T)
        res = []
        num = a.shape[0]
        for count in xrange(num):
            res.append(math.tanh(a.item(count, 0) + b.item(count, 0)))
        c = np.matrix(res).T
        c_updated = self.alpha*c + (1 - self.alpha)*((self.x).T)        
        
        # now calculate the output by using input and reservoir state
        X = np.concatenate((u, c_updated), axis=0)
        self.Yout = self.Wout*X
        
        # Now update x - this is a row vector
        self.x = c_updated.T
    
       
        return self.Yout
        
         
  
 
#
#class RequestHandler(SimpleXMLRPCRequestHandler):
#    rpc_paths = ('/RPC2',)
#
#class start_esn_rpc(esn_design):
#    
#    def __init__(self):
#        """
#        The code below starts a RPC server
#        """
#        server = SimpleXMLRPCServer(("localhost", 8000),
#                            requestHandler=RequestHandler)
#        server.register_introspection_functions()    
#        nlp = esn_design()
#        server.register_function(nlp.esn_x, 'results')
#
#        server.serve_forever()
    
    

    
    
    