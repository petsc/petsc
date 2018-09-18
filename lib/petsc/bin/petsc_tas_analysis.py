#!/usr/bin/env python
import numpy as np
import os
os.environ['MPLCONFIGDIR'] = os.environ.get('PETSC_DIR')+'/share/petsc/xml/'
print os.environ.get('MPLCONFIGDIR')
import importlib
import datetime as date
import matplotlib.pyplot as plt
import argparse

def main(cmdLineArgs):
    data = dataProces(cmdLineArgs)
    graphGen(data, cmdLineArgs.graph_flops_scaling, cmdLineArgs.dim)

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
    for module in cmdLineArgs.file:
        module = importlib.import_module(module)
        Nf     = 1
        nProcs = module.size
        dofs   = []
        times  = []
        flops  = []
        errors = []
        for f in range(Nf): errors.append([])
        for f in range(Nf): dofs.append([])
        level  = 0
        while level >= 0:
            stageName = "ConvEst Refinement Level "+str(level)
            if stageName in module.Stages:
                timeTemp = module.Stages[stageName]["SNESSolve"][0]["time"]
                flopTemp = module.Stages[stageName]["SNESSolve"][0]["flop"]
                #This loops is used to grab the greatest time and flop when run in parallel
                for n in range(1, nProcs):
                    timeTemp = timeTemp if timeTemp >= module.Stages[stageName]["SNESSolve"][n]["time"] \
                            else module.Stages[stageName]["SNESSolve"][n]["time"]
                    flopTemp = flopTemp if flopTemp >= module.Stages[stageName]["SNESSolve"][n]["flop"] \
                            else module.Stages[stageName]["SNESSolve"][n]["flop"]


                times.append(timeTemp)
                flops.append(flopTemp)

                for f in range(Nf):
                    dofs[f].append(module.Stages[stageName]["ConvEst Error"][0]["dof"][f])
                    errors[f].append(module.Stages[stageName]["ConvEst Error"][0]["error"][f])

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
    print module.__name__
    for k,v in data.get(module.__name__).items():
        print(" {} corresponds to {}".format(k,v))
        #print(v["dofs"][0])

    return data

def getNf(errorList):
    """"
    This simple function takes the supplied error list and loops through that list until it encounters -1.  The default
    convention is that each field from the problem has an entry in the error list with at most 8 fields.  If there are
    less thatn 8 fields those entries are set to -1.
    Example:
      A problem with 4 fields would have a list of the form [.01, .003, .2, .04, -1, -1, -1, -1]

    :param errorList: contains a list of floating point numbers with the errors from each level of refinement.
    :type errorList: List containg Floating point numbers.

    :returns: Nf and integer that represents the number of fields.
    """
    i = 0
    Nf = 1
    while errorList[i] != -1:
        Nf = Nf + 1
    return Nf

def graphGen(data, graph_flops_scaling, dim):
    """
    This function takes the supplied dictionary and plots the data from each file on the Mesh Convergence, Static Scaling, and
        Efficacy graphs.

        :param data: Contains the data to be ploted on the graphs, assumes the format: data[<file name>][<data type>]: <numpy array>
        :type data: Dictionary
        :param graph_flops_scaling: Controls creating the scaling graph that uses flops/second.  The default is not to.  This option
                                    is specified on the command line.
        :type graph_flops_scaling: Integer
        :param dim: Contains the number of dimension of the mesh.  This is specified on the command line.
        :type dim: Integer


        :returns: None
    """
    lstSqMeshConv = np.empty([2])

    counter = 0

    #Set up plots with labels
    plt.style.use('petsc_tas_style.mplstyle') #uses the specified style sheet for generating the plots

    meshConvFig = plt.figure()
    meshConvOrigHandles = []
    meshConvLstSqHandles = []
    axMeshConv = meshConvFig.add_subplot(1,1,1)
    axMeshConv.set(xlabel ='Problem Size $\log N$', ylabel ='Error $\log |x - x^*|$' , title ='Mesh Convergence')


    statScaleFig = plt.figure()
    statScaleHandles = []
    axStatScale = statScaleFig.add_subplot(1,1,1)
    axStatScale.set(xlabel = 'Time(s)', ylabel = 'Flop Rate (F/s)', title = 'Static Scaling')

    statScaleFig = plt.figure()
    statScaleHandles = []
    axStatScale = statScaleFig.add_subplot(1,1,1)
    axStatScale.set(xlabel = 'Time(s)', ylabel = 'DoF Rate (DoF/s)', title = 'Static Scaling')

    efficFig = plt.figure()
    efficHandles = []
    axEffic = efficFig.add_subplot(1,1,1)
    axEffic.set(xlabel = 'Time(s)', ylabel = 'Error Time', title = 'Efficacy')

    #Loop through each file and add the data/line for that file to the Mesh Convergance, Static Scaling, and Efficacy Graphs
    for fileName, value in data.items():
        #Least squares solution for Mesh Convergence
        #TODO need to update index for multifield
        lstSqMeshConv[0], lstSqMeshConv[1] = leastSquares(value['dofs'][0], value['errors'][0])


        print("Least Squares Data")
        print("==================")
        print("Mesh Convergence")
        print("Alpha: {} \n  {}".format(lstSqMeshConv[0], lstSqMeshConv[1]))

        convRate = lstSqMeshConv[0] * -dim
        print('convRate: {} of {} data'.format(convRate,fileName))

        ##Start Mesh Convergance graph
        convRate = str(convRate)
        #TODO have to update index for multiple fields
        x, = axMeshConv.loglog(value['dofs'][0], value['errors'][0], label = fileName + ' Orig Data')
        meshConvOrigHandles.append(x)
        #TODO need to update index for multifield
        y, = axMeshConv.loglog(value['dofs'][0], ((value['dofs'][0]**lstSqMeshConv[0] * 10**lstSqMeshConv[1])),
                label = fileName + " Convergence rate =  " + convRate )
        meshConvLstSqHandles.append(y)

        ##Start Static Scaling Graph, only if graph_flops_scaling equals 1.  Specified on the command line.
        #if graph_flops_scaling == 1 :
        #    x, =axStatScale.loglog(value['times'], value['flops']/value['times'], label = fileName)

        #statScaleHandles.append(x)
        ##Start Static Scaling with DoFs Graph
        x, =axStatScale.loglog(value['times'], value['dofs'][0]/value['times'], label = fileName)

        statScaleHandles.append(x)
        ##Start Efficacy graph
        x, = axEffic.semilogx(value['times'], -np.log10(value['errors'][0]*value['times']), label = fileName)

        efficHandles.append(x)

        counter = counter + 1
    meshConvHandles = meshConvOrigHandles + meshConvLstSqHandles
    meshConvLabels = [h.get_label() for h in meshConvOrigHandles]
    meshConvLabels = meshConvLabels + [h.get_label() for h in meshConvLstSqHandles]
    print(meshConvHandles)
    print(meshConvLabels)
    meshConvFig.legend(handles = meshConvHandles, labels = meshConvLabels)
    meshConvFig.savefig('meshConvergence' + date.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '.png')
    meshConvFig.show()

    statScaleLabels = [h.get_label() for h in statScaleHandles]
    print(statScaleHandles)
    print(statScaleLabels)
    statScaleFig.legend(handles = statScaleHandles, labels = statScaleLabels)
    statScaleFig.savefig('staticScaling' + date.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '.png')
    statScaleFig.show()

    efficLabels = [h.get_label() for h in efficHandles]
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

       :returns: alpha -- the convRate fo the least squares solution
       :returns: c -- the constant of the least squares solution.
    """


    x = np.log10(x)
    y = np.log10(y)
    X = np.hstack((np.ones((x.shape[0],1)),x.reshape((x.shape[0],1))))
    print y


    beta = np.dot(np.linalg.pinv(np.dot(X.transpose(),X)),X.transpose())
    beta = np.dot(beta,y.reshape((y.shape[0],1)))
    A = np.hstack((np.ones((x.shape[0],1)),x.reshape((x.shape[0],1))))


    AtranA = np.dot(A.T,A)

    invAtranA = np.linalg.pinv(AtranA)

    return beta[1][0], beta[0][0]

if __name__ == "__main__":
    #TODO Need to add cmd line argument for
    cmdLine = argparse.ArgumentParser(
           description = 'This is part of the PETSc toolkit for evaluating solvers using\n\
                   Time-Accuracy-Size(TAS) spectrum analysis')

    cmdLine.add_argument('-file', '--file', metavar = '<filename>', nargs = '*', help = 'List of files to import for TAS analysis')

    cmdLine.add_argument('-version', '--version', action = 'version', version = '%(prog)s 1.0')

    cmdLine.add_argument('-graph_flops_scaling', '--graph_flops_scaling', type = int, default = 0, choices = [0, 1],
    help = 'Enables graphing flop rate static scaling graph. Default: %(default)s  do not print the graph. 1 to print the graph')

    cmdLine.add_argument('-dim', '--dim', type = int, default = 2, help = 'Specifies the number of dimensions of the mesh. \
        Default: %(default)s .')

    cmdLineArgs = cmdLine.parse_args()


    main(cmdLineArgs)
