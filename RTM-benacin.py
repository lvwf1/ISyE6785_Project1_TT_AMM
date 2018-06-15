# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:36:45 2015

@author: tomb
"""

from math import exp
from math import log
import time

start_time = time.time()


S=90.25

K=100.0
H=90.0
r=0.1
sigma=0.25
h=log(S)-log(H)
T=1.0
lbd=3.0
k=(h**2)/(lbd*sigma**2)
q=0.0
a=r-q-(sigma**2)/2
p_u=0.5*((sigma**2)*(k/h**2)+(a*k/h)**2 + a*k/h)
p_d=0.5*((sigma**2)*(k/h**2)+(a*k/h)**2 - a*k/h)
p_m=1.0-p_u-p_d
N=int(T/k)
print "number of steps"
print N

s_vec=[log(S)]
for i in range(N):
    s_next=[]
    s_next= [s_vec[0]+h] + s_vec + [s_vec[-1]-h]
    s_vec=s_next


c_mat=s_next
for i in range(2*N+1):
    if(s_next[i]<=log(H)):
        c_mat[i]=0.0
    else:
        c_mat[i]=max(exp(s_next[i])-K,0.0)


for j in range(1,N+1):
    c_back=[]
    for i in range(1,2*(N-j)+2): # number of nodes at step j
        c_back.append(0.0)
        if(j!=N):
            if(log(S)+(N-j-i+1)*h>log(H)):            
                c_back[-1]=exp(-r*k)*(p_u*c_mat[i-1]+p_m*c_mat[i]+p_d*c_mat[i+1])
        else:
            c_back[-1]=exp(-r*k)*(p_u*c_mat[i-1]+p_m*c_mat[i]+p_d*c_mat[i+1])
    c_mat=c_back
print c_back

print("execution time: %s seconds" % (time.time() - start_time))