# -*- coding: utf-8 -*-
#!/usr/local/bin/python
#coding: utf8
# import system module

from math import log
from math import exp 
from math import floor 
from math import ceil
import time 
"""
Created on Thu Jun 18 19:43:43 2015

@author: tluquet
"""

def AMM(M,S0,H,K,greeks):
    #Constants
    X0 = log(S0)
    lgK = log(K)
    sigma = 0.25
    r = 0.1 
    alpha = r-(sigma**2)/2
    T = 1.0 
    K = 1
    #Barrier
    lgH = log(H)
    def computeProbas(h,k):

        pU = 0.5*((sigma**2)*(k/(h**2))+(alpha**2)*((k**2)/(h**2))+alpha*(k/h))
        pD = 0.5*((sigma**2)*(k/(h**2))+(alpha**2)*((k**2)/(h**2))-alpha*(k/h))
        pM = 1-pD-pU
        return [pD,pM,pU]
    def computeValue(kAMM,U,M,D,pU,pM,pD):
    
        C = exp(-r*kAMM)*(pD*D+pM*M+pU*U)
        return C
    def buildRTMPriceTree(timeStep,h,X0):
        priceTree = []
        for i in range(2*(timeStep)+1):
            priceTree.append((X0-timeStep*h)+i*h)
        return priceTree
    
    def RTMpayOff(lgPrice,lgK):
        K = exp(lgK)
        Price= exp(lgPrice)
        if(Price>K):
            value = Price-K
        else :
            value = 0.0 
        return value
    
    def buildRTMOptionPriceTree(timeStep,previousValues,trinTree,lgH,pD,pM,pU,kRTM):
        opTree = []
        for i in range(2*timeStep + 1):
            if(trinTree[i]>lgH):
                C = exp(-r*kRTM)*(pD*previousValues[i]+pM*previousValues[i+1]+pU*previousValues[i+2])
            else:
                C = 0.0
            opTree.append(C)
              
        return opTree 
    
    def computeGreeks(C0,e,S):
        delta = (C0[2]-C0[0])/(2.0*e*S)
        gamma = (1.0/S**2)*(((C0[2]+C0[0]-2*C0[1])/(e**2))-(C0[2]-C0[0])/(2.0*e))      
        return [delta,gamma]
    
    h = (2**M)*(X0-lgH)
    lbda = 3.0 
    k = T/int(lbda*(sigma**2)/((h**2)*T))
    steps = int(floor(T/k))
    print ("------- AMM Method ---------")
    print ("\nS0 = " + str(exp(X0)))
    print ("K (strike price) = " + str(exp(lgK)))
    print ("H (barrier) = " + str(exp(lgH)))
    print ("Number of steps for lattice A : " + str(steps))
    #Create the corresponding tree
    trinTree = []
    for i in range(steps+1):
        treek = buildRTMPriceTree(i,h,lgH+h)
        trinTree.append(treek)
    startTime = time.time()
    #Compute the payoff of the lattice A 
    optionTreeA = []
    finalPayoffA = []
    for i in range(len(trinTree[steps])):
        finalPayoffA.append(RTMpayOff(trinTree[steps][i],lgK))
    optionTreeA.append(finalPayoffA)
    #Calculate pU, pD and pM 
    pURTM = 0.5*((sigma**2)*(k/(h**2))+(alpha**2)*((k**2)/(h**2))+alpha*(k/h))
    pDRTM = 0.5*((sigma**2)*(k/(h**2))+(alpha**2)*((k**2)/(h**2))-alpha*(k/h))
    pMRTM = 1-pDRTM-pURTM
    for i in range(1,steps+1):
        treek = buildRTMOptionPriceTree(steps-i,optionTreeA[0],trinTree[steps-i],lgH,pDRTM,pMRTM,pURTM,k)
        optionTreeA.insert(0,treek)
    finalValue = optionTreeA[0][0]
    # Construction of the lattice B,C and D
    if( M > 0):
        optionTreeB= []
        optionTreeB.append([0,0,trinTree[steps][steps]])
        j = 1
        B = [0,0,0]
        for i in range(1,steps*4+1):
            stepA = int(ceil(steps-i/4.0))
            p1 = computeProbas(h/2,k/4)
            B[1] = computeValue(k/4,optionTreeB[0][2],optionTreeB[0][1],0,p1[2],p1[1],p1[0])
            if(j >0):
                p2 = computeProbas(h,j*k/4)
                B[2] = computeValue(j*k/4,optionTreeA[stepA][stepA+1],optionTreeA[stepA][stepA],0,p2[2],p2[1],p2[0])
            else:
                B[2] = optionTreeA[stepA][stepA]
                
            optionTreeB.insert(0,B)
            j += 1 
            if(j>3):
                j = 0
        finalValue=optionTreeB[0][1]
        if(greeks):
            [delta,gamma]=computeGreeks(optionTreeB[0],h/2,S0)
        if(M>1):
            optionTreeC= []
            optionTreeC.append([0,0,trinTree[steps][steps]])
            j = 1
            C = [0,0,0]
            for i in range(1,steps*16+1):
                stepA = int(ceil(4*steps-i/4.0))
                p1 = computeProbas(h/4,k/16)
                C[1] = computeValue(k/16,optionTreeC[0][2],optionTreeC[0][1],0,p1[2],p1[1],p1[0])
                if(j >0):
                    p2 = computeProbas(h/2,j*k/16)
                    C[2] = computeValue(j*k/16,optionTreeB[stepA][2],optionTreeB[stepA][1],0,p2[2],p2[1],p2[0])
                else:
                    C[2] = optionTreeB[stepA][1]
                    
                optionTreeC.insert(0,C)
                j += 1 
                if(j>3):
                    j = 0 
            finalValue = optionTreeC[0][1]
            if(greeks):
                [delta,gamma]=computeGreeks(optionTreeC[0],h/4,S0)
            if(M>2):
                optionTreeD= []
                optionTreeD.append([0,0,0])
                j = 1
                D = [0,0,0]
                for i in range(1,steps*64+1):
                    stepA = int(ceil(16*steps-i/4.0))
                    p1 = computeProbas(h/8,k/64)
                    D[1] = computeValue(k/64,optionTreeD[0][2],optionTreeD[0][1],0,p1[2],p1[1],p1[0])
                    if(j >0):
                        p2 = computeProbas(h/4,j*k/64)
                        D[2] = computeValue(j*k/64,optionTreeC[stepA][2],optionTreeC[stepA][1],0,p2[2],p2[1],p2[0])
                    else:
                        D[2] = optionTreeC[stepA][1]
                        
                    optionTreeD.insert(0,D)
                    j += 1 
                    if(j>3):
                        j = 0 
                finalValue = optionTreeD[0][1] 
                if(greeks):
                    [delta,gamma]=computeGreeks(optionTreeD[0],h/8,S0)
                if(M>3):
                    optionTreeE= []
                    optionTreeE.append([0,0,0])
                    j = 1
                    E = [0,0,0]
                    for i in range(1,steps*256+1):
                        stepA = int(ceil(16*steps-i/4.0))
                        p1 = computeProbas(h/16,k/256)
                        E[1] = computeValue(k/256,optionTreeE[0][2],optionTreeE[0][1],0,p1[2],p1[1],p1[0])
                        if(j >0):
                            p2 = computeProbas(h/16,j*k/256)
                            E[2] = computeValue(j*k/256,optionTreeD[stepA][2],optionTreeD[stepA][1],0,p2[2],p2[1],p2[0])
                        else:
                            E[2] = optionTreeD[stepA][1]
                            
                        optionTreeE.insert(0,E)
                        j += 1 
                        if(j>3):
                            j = 0 
                    finalValue = optionTreeE[0][1] 
                    if(greeks):
                        [delta,gamma]=computeGreeks(optionTreeE[0],h/16,S0)
                        
    print("CPU time: %s seconds" % (time.time() - startTime))
    print("\nFinal down and out option value with AMM-" + str(M) + " : " + str(finalValue) )
    if(greeks):
        print ("Delta = " + str(delta))
        print ("Gamma = " + str(gamma))
    
# Here you can change the values or remove the delta and gamma computation
    # Max value for M: 4
AMM(1,91,90,100,True)
AMM(2,90.5,90,100,True)
AMM(3,90.25,90,100,True)
AMM(4,90.125,90,100,True)  
    
    
    
    
    
    
    







