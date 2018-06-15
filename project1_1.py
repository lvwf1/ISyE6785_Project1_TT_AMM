#!/usr/local/bin/python
#coding: utf8
# import system module

from math import sqrt 
from math import log
from math import exp 
from math import ceil 
import time 
import sys

"""
Project 1, part1 for ISYE 6785

@author: Thomas Luquet
"""

#Need 3 arguments in the command line : initial price S0 , Barrier , 
#strike price and nb steps
if(len(sys.argv)==5):
    #Constants
    S = float(sys.argv[1])
    B = log(float(sys.argv[2]))
    
    X0 = log(S)
    K = log(float(sys.argv[3]))
    sigma = 0.3
    r = 0.1 

    # Define the number of steps 
    T = 0.6 #Option Maturity
    steps = int(sys.argv[4])
    k = T/steps
    
    #Probabilities pU,pM and pD 
    pU = 1.0/6.0 
    pM = 2.0/3.0
    pD = 1.0/6.0 
    h = sigma*sqrt(3*k)
    
    #Create the trinomial tree at one step
    def buildPriceTree(timeStep,h,X0):
        priceTree = []
        for i in range(2*(timeStep)+1):
            priceTree.append((X0-timeStep*h)+i*h)
        return priceTree
    
    def payOff(lgPrice,lgK,lgB,steps,X0):
        Kp = exp(lgK)
        Price= exp(lgPrice)
        if(Price>K):
            stepMin=ceil((X0-lgB)/h) 
            maxPrice = exp((X0-stepMin*h)+(steps-stepMin)*h)
            if(Price<=maxPrice):
                value = Price-Kp
            else:
                value = 0.0 
        else :
            value = 0.0 
        return value
        
    def buildOptionPriceTree(timeStep,previousValues,trinTree,lgB):
        opTree = []
        stepMin=ceil((X0-lgB)/h) 
        if(stepMin<timeStep):
            maxPrice = exp((X0-stepMin*h)+(timeStep-stepMin)*h)
            for i in range(2*timeStep + 1):
                if(trinTree[i]<lgB or exp(trinTree[i])<=maxPrice):
                    C = exp(-r*k)*(pD*previousValues[i]+pM*previousValues[i+1]+pU*previousValues[i+2])
                elif(trinTree[i]-h*(steps-timeStep)<lgB):
                    # verification of the possibility to later cross the barrier and go above K
                    value = trinTree[i]
                    j = 0
                    while(value>lgB):
                        value = value - h
                        j = j+1
                    while(value<=K):
                        value=value+h
                        j=j+1
                    if(j > steps-timeStep):#impossible
                        C = 0.0
                    else:
                        C = exp(-r*k)*(pD*previousValues[i]+pM*previousValues[i+1]+pU*previousValues[i+2])
                else:
                    C = 0.0
                
                opTree.append(C)
        else:
            for i in range(2*timeStep + 1):
                if(trinTree[i]<lgB):
                    C = exp(-r*k)*(pD*previousValues[i]+pM*previousValues[i+1]+pU*previousValues[i+2])
                elif(trinTree[i]-h*(steps-timeStep)<lgB):
                    # verification of the possibility to later cross the barrier and go above K
                    value = trinTree[i]
                    j = 0
                    while(value>lgB):
                        value = value - h
                        j = j+1
                    while(value<=K):
                        value=value+h
                        j=j+1
                    if(j > steps-timeStep):#impossible
                        C = 0.0
                    else:
                        C = exp(-r*k)*(pD*previousValues[i]+pM*previousValues[i+1]+pU*previousValues[i+2])
                else:
                    C = 0.0
                opTree.append(C)                
        return opTree 
          
    # -------------------- Part 1: Trinomial Lattice ----------------------------   
    startTime = time.time() 
    print ("Number of steps to maturity: " + str(steps))
    #Creation of the trinomial tree
    #print "Creation of the trinomal tree (log Price)"
    trinTree = []
    for i in range(steps+1):
        treek = buildPriceTree(i,h,X0)
        trinTree.append(treek)
        #print treek
    
    #Creation of the Option price Tree
    optionTree = []
    finalPayoff = []
    print ("K (strike price) = " + str(exp(K)))
    print ("B (barrier) = " + str(exp(B)))
    for i in range(len(trinTree[steps])):
        finalPayoff.append(payOff(trinTree[steps][i],K,B,steps,X0))
    optionTree.append(finalPayoff)
    
    for i in range(1,steps+1):
        treek = buildOptionPriceTree(steps-i,optionTree[0],trinTree[steps-i],B)
        optionTree.insert(0,treek)
    
    #print "\nOption Value Tree:"
    #for i in range(steps+1):
        #print optionTree[i]

    print ("Final down and in option value = " + str(optionTree[0]))
    print ("CPU time: %s seconds" % (time.time() - startTime))
else:
    print ("You need 4 arguments in the command line : initial price S0 , Barrier, strike price and nb steps")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
