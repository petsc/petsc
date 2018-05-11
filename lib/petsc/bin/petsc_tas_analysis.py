#!/usr/bin/env python
import numpy as np
import ex13

def main():
    Nf     = 1
    dofs   = []
    times  = []
    flops  = []
    errors = []
    for f in range(Nf): errors.append([])
    level  = 0
    while level >= 0:
      stageName = "ConvEst Refinement Level "+str(level)
      if stageName in ex13.Stages:
        dofs.append(ex13.Stages[stageName]["ConvEst Error"][0]["dof"])
        times.append(ex13.Stages[stageName]["SNESSolve"][0]["time"])
        flops.append(ex13.Stages[stageName]["SNESSolve"][0]["flop"])
        for f in range(Nf): errors[f].append(ex13.Stages[stageName]["ConvEst Error"][0]["error"][f])
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

    
    print dofs
    print times
    print flops
    print errors
    
    print("Least Squares Data")
    print("==================")
    print("Mesh Convergence")
    print("y = {}*x + {}".format(lstSqMeshConv[0], lstSqMeshConv[1])) 

    import matplotlib.pyplot as plt

    plt.title('Mesh Convergence')
    plt.xlabel('Problem Size $\log N$')
    plt.ylabel('Error $\log |x - x^*|$')
    plt.loglog(dofs, errors[0])
    plt.show()
    
    #Least Squares fit plot
    plt.title('Mesh Convergence using Least Squares')
    plt.xlabel('Problem Size $\log N$')
    plt.ylabel('Error $\log |x - x^*|$')
    plt.loglog(dofs, lstSqMeshConv[0]*dofs + lstSqMeshConv[1])
    plt.show()


    plt.title('Static Scaling')
    plt.xlabel('Time (s)')
    plt.ylabel('Flop Rate (F/s)')
    plt.loglog(times, flops/times)
    plt.show()

    plt.title('Efficacy')
    plt.xlabel('Time (s)')
    plt.ylabel('Action (s)')
    plt.loglog(times, errors[0]*times)
    plt.show()


def leastSquares(x, y):
    """This function takes 2 numpy arrays of data and out puts the least squares solution,
       y = m*x + c.  The solution is obtained by finding the result of y = Ap, where A
       is the matrix of the form [[x 1]] and p = [[m], [c]].
       
       Arguments:
       x -- Type: numpy array, contains the x values for the data.
       y -- Type: numpy array, contains the y values for the data.
       
       Return Values:
       m -- the slope fo the least squares solution
       c -- the constant of the least squares solution.
    """

    A = np.vstack([x, np.ones(len(x))]).T

    m, c = np.linalg.lstsq(A, y)[0]

    return m, c

if __name__ == "__main__":
    main()
