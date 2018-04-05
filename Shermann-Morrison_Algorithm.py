#####################################################################
#                                                                   #
#   Python 3.6 program, solving tridiagonal sparse matrix using     #
#   Shermann-Morrison algorithm which was made to solve A^-1*b=z    #
#   equation and Shermann-Morrison formula for solving a sum of     #
#   inverted matrix A1 and diyadic product uv^T. His time           #
#   complexity is total O(n).                                       #
#                                                   Piotr Wlazło    #
#                                                   19.12.2017      #
#####################################################################

# -*- coding: cp1250 -*-
import numpy as np
import copy

#Create Matrices
A1 = np.diag([2.0,1.0,1.0,2.0,1.0,2.0,1.0,2.0],-1)\
    +np.diag([4.0,8.0,4.0,3.0,4.0,5.0,3.0,8.0,5.0])\
    +np.diag([2.0,1.0,1.0,2.0,1.0,2.0,1.0,2.0],1)
A1[0][8] =1; A1[8][0]=1
b = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.]) #b vector
u = np.array([1.,0.,0.,0.,0.,0.,0.,0.,1.]) #u vector
u_v = np.zeros(shape=[9,9])
u_v[0][0]=1
u_v[8][8]=1

A = A1 - u_v #creating tridiagonal matrix

#To solve this simultaneous equations I have to solve two times equation
#with tridiagonal matrix, that's why I use Thomas Algorithm 
#Thomas Algorithm - LU decomposition
L=np.diag([1.0]*9,0)
U=np.zeros(shape=[9,9])
U[0][0]=A[0][0] #because f1 = 1*f1'
#LU decomposition
for i in range(0,8):
    U[i][i+1]=A[i][i+1]
    L[i+1][i] = (A[i+1][i]/U[i][i])
    U[i+1][i+1]=A[i+1][i+1]-L[i+1][i]*U[i][i+1]

#Creating two copies of vectors for solving
b_temp=copy.copy(b)
u_temp=copy.copy(u)
#Creating z and q vectors
z = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
q = [0.,0.,0.,0.,0.,0.,0.,0.,0.]

#Making forward-substitution
for x in range(1,9):
    b_temp[x] = b[x]-((L[x][x-1])*(b_temp[x-1]))
    u_temp[x] = u[x]-((L[x][x-1])*(u_temp[x-1])) 

#Making back-substitution
z[8]=b_temp[8]/U[8][8]
q[8]=u_temp[8]/U[8][8]
for i in range(7,-1,-1):
    z[i]=(b_temp[i]-((A[i][i+1])*(z[i+1])))/U[i][i]
    q[i]=(u_temp[i]-((A[i][i+1])*(q[i+1])))/U[i][i]

#Create x vector with results
Results = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,])
#Solving Shermann-Morrison equation
for i in range(9):
    Results[i] = z[i]-(np.dot(u,z)*q[i])/((1+np.dot(u,q)))
for i in range(9):
    print("x%d"%(i+1)+" = %f"%Results[i])
    

    




    

