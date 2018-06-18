#!/usr/bin/env python
import numpy as np
import importlib
import datetime as date
import matplotlib.pyplot as plt
import argparse

def main(cmdLineArgs):
    module = importlib.import_module(cmdLineArgs.file[0])
    Nf     = 1
    dofs   = []
    times  = []
    flops  = []
    errors = []
    for f in range(Nf): errors.append([])
    level  = 0
    while level >= 0:
      stageName = "ConvEst Refinement Level "+str(level)
      if stageName in module.Stages:
        dofs.append(module.Stages[stageName]["ConvEst Error"][0]["dof"])
        times.append(module.Stages[stageName]["SNESSolve"][0]["time"])
        flops.append(module.Stages[stageName]["SNESSolve"][0]["flop"])
        for f in range(Nf): errors[f].append(module.Stages[stageName]["ConvEst Error"][0]["error"][f])
        level = level + 1
      else:
        level = -1

    dofs   = np.array(dofs)
    times  = np.array(times)
    flops  = np.array(flops)
    errors = np.array(errors)
    
    lstSqMeshConv = np.empty([2]) 

    #Least squares solution for Mesh Convergence
    lstSqMeshConv[0], lstSqMeshConv[1] = leastSquares(dofs, errors[0])    
    
    
    print("Least Squares Data")
    print("==================")
    print("Mesh Convergence")
    print("Alpha: {} \n  {}".format(lstSqMeshConv[0], lstSqMeshConv[1])) 

    
    plt.title('Mesh Convergence')
    plt.xlabel('Problem Size $\log N$')
    plt.ylabel('Error $\log |x - x^*|$')
    meshConv, = plt.loglog(dofs, errors[0])
    meshConvLstSq, = plt.loglog(dofs, (dofs**lstSqMeshConv[0] * 10**lstSqMeshConv[1])) 
    plt.legend([meshConv, meshConvLstSq], ['Original Data', 'Least Squares + delta'])
    plt.show()
    plt.savefig('meshConvergence' + date.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '.png')


    plt.title('Static Scaling')
    plt.xlabel('Time (s)')
    plt.ylabel('Flop Rate (F/s)')
    plt.loglog(times, flops/times, label = module.__name__)
    plt.legend()
    plt.show()
    plt.savefig('staticScaling' + date.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '.png')

    plt.title('Efficacy')
    plt.xlabel('Time (s)')
    plt.ylabel('Action (s)')
    plt.loglog(times, errors[0]*times, label = module.__name__)
    plt.legend()
    plt.show()
    plt.savefig('efficacy' + date.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '.png')

def leastSquares(x, y):
    """
    This function takes 2 numpy arrays of data and out puts the least squares solution,
       y = m*x + c.  The solution is obtained by finding the result of y = Ap, where A
       is the matrix of the form [[x 1]] and p = [[m], [c]].
       
       :param x: Contains the x values for the data.
       :type x: numpy array
       :param y: Contains the y values for the data.
       :type y: numpy array

       :returns: alpha -- the slope fo the least squares solution
       :returns: c -- the constant of the least squares solution.
    """


    x = np.log10(x)
    y = np.log10(y)
    X = np.hstack((np.ones((x.shape[0],1)),x.reshape((x.shape[0],1))))
    

    beta = np.dot(np.linalg.pinv(np.dot(X.transpose(),X)),X.transpose())
    beta = np.dot(beta,y.reshape((y.shape[0],1)))
    A = np.hstack((np.ones((x.shape[0],1)),x.reshape((x.shape[0],1))))
    

    AtranA = np.dot(A.T,A)

    invAtranA = np.linalg.pinv(AtranA)
   
    return beta[1][0], beta[0][0]

if __name__ == "__main__":
    cmdLine = argparse.ArgumentParser(
           description = 'This is part of the PETSc toolkit for evaluating solvers using\n\
                   Time-Accuracy-Size(TAS) spectrum analysis')
    
    cmdLine.add_argument('-file', '--file', metavar = '<filename>', nargs = '*', help = 'List of files to import for TAS analysis')
    
    cmdLine.add_argument('-version', '--version', action = 'version', version = '%(prog)s 1.0')
    
    cmdLineArgs = cmdLine.parse_args()
    
    print cmdLineArgs
    print len(cmdLineArgs.file)
    for x in cmdLineArgs.file:
        print x
    
    print ("******New Section***********")
    print cmdLineArgs.file[0]
    #print cmdLineArgs.file[1]
    #m = importlib.import_module("run")
    #print type(m.size)
    main(cmdLineArgs)
