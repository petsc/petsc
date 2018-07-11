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

    plt.style.use('petsc_tas_style') #uses the specified style sheet for generating the plots
    
    fig = plt.figure()
    
    end = dofs.size-1

    ax = fig.add_subplot(111)
    ax.set(xlabel ='Problem Size $\log N$', ylabel ='Error $\log |x - x^*|$' , title ='Mesh Convergence')
    meshConv, = ax.loglog(dofs, errors[0], 'ro')
    slope = ((((dofs[end]**lstSqMeshConv[0] * 10**lstSqMeshConv[1]))-((dofs[0])**lstSqMeshConv[0] * 10**lstSqMeshConv[1])) \
            /(dofs[end]-dofs[0]))

    print('Slope: {}'.format(slope))

    ##Start Mesh Convergance graph
    slope = 'Slope : ' + str(slope)
    meshConvLstSq, = ax.loglog(dofs, ((dofs)**lstSqMeshConv[0] * 10**lstSqMeshConv[1]), 'g--') 
    ax.legend([meshConv, meshConvLstSq], ['Original Data', 'Least Squares'])
    ax.annotate(slope, xy = (5,5),  xycoords = 'data', arrowprops = dict(facecolor = 'black', shrink = 0.05))
    plt.savefig(module.__name__ + 'meshConvergence' + date.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '.png')
    plt.show()
    
    ##Start Static Scaling Graph
    plt.title('Static Scaling')
    plt.xlabel('Time (s)')
    plt.ylabel('Flop Rate (F/s)')
    plt.loglog(times, flops/times, label = module.__name__)
    plt.legend()
    plt.savefig(module.__name__ + 'staticScaling' + date.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '.png')
    plt.show()
    
    ##Start Efficacy graph
    plt.title('Efficacy')
    plt.xlabel('Time (s)')
    plt.ylabel('Action (s)')
    plt.loglog(times, errors[0]*times, label = module.__name__)
    plt.legend()
    plt.savefig(module.__name__ + 'efficacy' + date.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '.png')
    plt.show()
   
def dataProces(cmdLineArgs):
    """
    This function takes the list of data files supplied as command line arguments and parses them into a multi-level
        dictionary, whose top level key is the file name, followed by data type, i.e. dofs, times, flops, errors, and 
        the finale value is a NumPy array of the data to plot.  

        data[<file name>][<data type>]: <numpy array>
    
        :param cmdLineArgs: Contains the file names to import and parse
        :type cmdLineArgs: numpy array

        :returns: data -- A dictionary containing the parsed data from the files specified on the command line.
    """
    data = {}
    print(cmdLineArgs)
    for module in cmdLineArgs:
        module = importlib.import_module(module)
        Nf     = module.size
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
        data[module.__name__] = {}
        data[module.__name__]["dofs"]   = dofs
        data[module.__name__]["times"]  = times
        data[module.__name__]["flops"]  = flops
        data[module.__name__]["errors"] = errors

    print('**************data*******************')
    for k,v in data.items():
        print(" {} corresponds to {}".format(k,v))
        print(v["dofs"])
    
    return data

def graphGen(data):
    
    lstSqMeshConv = np.empty([2])

    #Set up plots with labels
    plt.style.use('petsc_tas_style') #uses the specified style sheet for generating the plots
    
    meshConvFig = plt.figure()
    meshConvOrigHandles = []
    meshConvLstSqHandles = []
    axMeshConv = meshConvFig.add_subplot(1,1,1)
    axMeshConv.set(xlabel ='Problem Size $\log N$', ylabel ='Error $\log |x - x^*|$' , title ='Mesh Convergence')
    

    statScaleFig = plt.figure()
    statScaleHandles = []
    axStatScale = statScaleFig.add_subplot(1,1,1)
    axStatScale.set(xlabel = 'Time(s)', ylabel = 'Flope Rate (F/s)', title = 'Static Scaling')
    
    efficFig = plt.figure()
    efficHandles = []
    axEffic = efficFig.add_subplot(1,1,1)
    axEffic.set(xlabel = 'Time(s)', ylabel = 'Error Time', title = 'Efficacy')

    #Loop through each file and add the data/line for that file to the Mesh Convergance, Static Scaling, and Efficacy Graphs
    for fileName, value in data.items():
        #Least squares solution for Mesh Convergence
        lstSqMeshConv[0], lstSqMeshConv[1] = leastSquares(value['dofs'], value['errors'][0])    
    
    
        print("Least Squares Data")
        print("==================")
        print("Mesh Convergence")
        print("Alpha: {} \n  {}".format(lstSqMeshConv[0], lstSqMeshConv[1])) 
    
        end = value['dofs'].size-1

        slope = ((((value['dofs'][end]**lstSqMeshConv[0] * 10**lstSqMeshConv[1]))-((value['dofs'][0])**lstSqMeshConv[0] * 10**lstSqMeshConv[1])) \
            /(value['dofs'][end]-value['dofs'][0]))

        print('Slope: {} of {} data'.format(slope,fileName))

        ##Start Mesh Convergance graph
        slope = 'Slope : ' + str(slope)
        x, = axMeshConv.loglog(value['dofs'], value['errors'][0], 'ro', label = fileName + 'Orig Data')
        meshConvOrigHandles.append(x)

        x, = axMeshConv.loglog(value['dofs'], ((value['dofs']**lstSqMeshConv[0] * 10**lstSqMeshConv[1])), 'g--', 
                label = 'Least Squares Slope: ' + slope) 
            

        meshConvLstSqHandles.append(x)         
        
        ##Start Static Scaling Graph
        x, =axStatScale.loglog(value['times'], value['flops']/value['times'], label = fileName)

        statScaleHandles.append(x)    
        ##Start Efficacy graph
        x, = axEffic.loglog(value['times'], value['errors'][0]*value['times'], label = fileName)

        efficHandles.append(x)    
    
    meshConvHandles = meshConvOrigHandles + meshConvLstSqHandles
    meshConvLabels = [h.get_label() for h in meshConvOrigHandles]
    meshConvFig.legend(handles = meshConvOrigHandles, labels = meshConvLabels)
    meshConvFig.savefig('meshConvergence' + date.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '.png')
    meshConvFig.show()
    
    statScaleLabels = [h.get_label() for h in statScaleHandles]
    statScaleFig.legend(handles = statScaleHandles, labels = statScaleLabels)
    statScaleFig.savefig('staticScaling' + date.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '.png')
    statScaleFig.show()
    
    efficLabels = [h.get_label for h in efficHandles]
    efficFig.legend(handles = efficHandles, labels = efficLabels)
    efficFig.savefig('efficacy' + date.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '.png')
    efficFig.show()


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
    test = cmdLineArgs.file
    print(test)
    test = importlib.import_module(test[0])
    print("Size: ", test.size)
    data = dataProces(cmdLineArgs.file)
    graphGen(data)
    #print cmdLineArgs.file[1]
    #m = importlib.import_module("run")
    #print type(m.size)
    #main(cmdLineArgs)
