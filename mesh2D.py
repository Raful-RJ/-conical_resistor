__all__ = ['mesh2D']

import numpy as np
import matplotlib.pyplot as plt

def mesh2D(h,x0,x1,f, path = ['']):

    def generate_pid(i,j):
        return str(i) + '.' + str(j)

    def F(f,X):
        Y = []
        for x in X:
            Y.append(f(x))
        return np.array(Y)

    X = np.arange(x0,x1+h,h)
    Y = F(f,X)
    y_max = max(Y)

    lim_fun = np.array([X,Y])

    Y,X = np.mgrid[y_max+h:-y_max-h:-h,x0:x1+h:h]
    coord = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if abs(Y[i][j]) <= f(X[i][j]):
                pid = generate_pid(i,j)
                
                #test for up neighbor
                try:
                    if abs(Y[i+1][j]) <= f(X[i+1][j]):
                        up = generate_pid(i+1,j)
                    else: up = '-1'
                except: up = '-1'
                
                #test for down neighbor
                try:
                    if (abs(Y[i-1][j]) <= f(X[i-1][j])) and (i-1)>=0:
                        down = generate_pid(i-1,j)
                    else: down = '-1'
                except: down = '-1'
                #test for left neighbor
                try:
                    if (abs(Y[i][j-1]) <= f(X[i][j-1])) and (j-1)>=0:
                        left = generate_pid(i,j-1)
                    else: left = '-1'
                except: left = '-1'
                #test for right neighbor
                try:
                    if abs(Y[i][j+1]) <= f(X[i][j+1]):
                        right = generate_pid(i,j+1)
                    else: right = '-1'
                except: right = '-1'

                #setting boundary values
                if X[i][j] == x0:
                    potential_value = 10
                elif X[i][j] == x1:
                    potential_value = 0
                elif '-1' in [up,down,left,right]:
                    potential_value = 0
                else: potential_value = np.nan
                
                coord.append([pid,X[i][j],Y[i][j],potential_value,0, 0,up,down,left,right])

    general_data = np.array([x0,x1,h])
     
    features = np.array(['Id','x','y','potential','Jx','Jy','up_neighbor','down_neighbor','left_neighbor','right_neighbor'])
    coord = np.reshape(np.array(coord, dtype = 'object'),(len(coord),features.shape[0]))
    
    return X,Y,lim_fun,features,coord, general_data


        
    

    
