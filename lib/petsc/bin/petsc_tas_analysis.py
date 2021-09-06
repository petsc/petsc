#!/usr/bin/env python3
import numpy as np
import os
os.environ['MPLCONFIGDIR'] = os.environ.get('PETSC_DIR')+'/share/petsc/xml/'
import importlib
import datetime as date
import matplotlib.pyplot as plt
import argparse
import math
import configureTAS as config
import sys
import traceback
import pandas as pd
from tasClasses import File
from tasClasses import Field


def main(cmdLineArgs):
    data = []
    files = getFiles(cmdLineArgs)
    if len(files['module']) != 0:
        for fileName in files['module']:
            data.append(dataProces(cmdLineArgs, fileName ))
    if len(files['csv']) != 0:
        for fileName in files['csv']:
            data.append(dataProcesCSV(cmdLineArgs, fileName ))
    for item in data:
        graphGen(item, cmdLineArgs.enable_graphs, cmdLineArgs.graph_flops_scaling, cmdLineArgs.dim)

def getFiles(cmdLineArgs):
    """
    This function first determins if it should look in the pathway specifed in filePath['absoluteData']
    in the configurationTAS.py file or a file name given as a command line argument using -f or -file.
    It then builds lists of file names and stores them in a dictionary, where they keys correspond to
    the type of file, ie, module(ASCII type) or CSV.

    :param cmdLineArgs: Contains command line arguments.

    :returns:   files, a dictionary with keys whose values are lists of file names, grouped by type
                of file.
    """
    dataPath = config.filePath['absoluteData']
    sys.path.append(dataPath)
    files = {'module': [], 'csv': []}

    if(cmdLineArgs.file == None):
        try:
            filesTemp = os.listdir(dataPath)
            for f in filesTemp:
                if f[-3:] == '.py':
                    files['module'].append(f[0:len(f)-3])
                elif f[-4:] == '.pyc':
                    files['module'].append(f[0:len(f)-4])
                elif f[-4:] == '.csv':
                    files['csv'].append(f)
            if len(filesTemp) == 0 or len(files) == 0:
                raise IOError()
        except IOError:
            sys.exit ("No valid data files in " + dataPath + " and -file/-f argument is empty. \n"
            "Please check for .py, .pyc, or .csv files in " + dataPath + " or specify one with the -file/-f "
            "argument.")
    else:
        if cmdLineArgs.file[0][-4:] == '.csv':
            print('csv file')
            files['csv'].append(cmdLineArgs.file[0])
        else:
            files['module'].append(cmdLineArgs.file[0])
    for key in files.keys():
        print(f'key: {key}, items {files[key]}')
    return files

def dataProcesCSV(cmdLineArgs, fileName):
    """
    This function takes the list of data files in CSV format supplied as a list and parses them into a tasClasses
    object, whose top level key is the file name, followed by data type, i.e. dofs, times, flops, errors, and
    the finale value is a NumPy array of the data to plot.

        data[<file name>][<data type>]:<numpy array>

    :param cmdLineArgs: Contains command line arguments.
    :param fileNames: Contains the CSV file names.
    :type string:

    :returns:   data a tasClasses file object containing the parsed data from the files specified on the command line.
    """
    data = {}
    results = []
    df                 = pd.read_csv(fileName)
    Nf                 = getNfCSV(df)
    nProcs             = int(df.columns.tolist()[24])
    dofs               = []
    errors             = []

    times              = []
    timesMin           = []
    meanTime           = []
    timeGrowthRate     = []

    flops              = []
    flopsMax           = []
    flopsMin           = []
    meanFlop           = []
    flopGrowthRate     = []

    luFactor           = []
    luFactorMin        = []
    luFactorMean       = []
    luFactorGrowthRate = []

    file               = File(fileName[0:len(fileName)-4])

    #filters for using in df.loc[]
    SNESSolveFilter    = (df['Event Name']=="SNESSolve")
    MatLUFactorFilter  = ((df['Event Name'] == "MatLUFactorNum") | \
        (df['Event Name'] == "MatLUFactorSym"))
    ConvEstErrorFilter = (df['Event Name']=='ConvEst Error')
    rankFilter = (df['Rank'] == 0)


    for f in range(Nf): errors.append([])
    for f in range(Nf): dofs.append([])

    #level set to 1 due to problems with coarse grid effecting measurements
    level  = 1
    while level >= 1:
        print('in while loop')
        if ("ConvEst Refinement Level " + str(level) in df['Stage Name'].values):
            stageName = "ConvEst Refinement Level "+str(level)
            #Level dependent filters
            stageNameFilter = (df['Stage Name'] == stageName)
            fieldFilter = stageNameFilter & ConvEstErrorFilter & rankFilter

            SNESSDf = df.loc[stageNameFilter & SNESSolveFilter]

            MatLUFactorDf = df.loc[(stageNameFilter & MatLUFactorFilter), ['Time','Rank']]
            #groupby done in order to get the sum of MatLUFactorNum and MatLUFactorSym
            #For each Rank/CPU
            MatLUFactorDf = MatLUFactorDf.groupby(['Rank']).sum()

            meanTime.append((SNESSDf['Time'].sum())/nProcs)
            times.append(SNESSDf['Time'].max())
            timesMin.append(SNESSDf['Time'].min())


            meanFlop.append((SNESSDf['FLOP'].sum())/nProcs)
            flops.append(SNESSDf['FLOP'].sum())
            flopsMax.append(SNESSDf['FLOP'].max())
            flopsMin.append(SNESSDf['FLOP'].min())

            if level > 1:
                timeGrowthRate.append(meanTime[level-1]/meanTime[level-2])
                flopGrowthRate.append(meanFlop[level-1]/meanFlop[level-2])

            luFactorMean.append(MatLUFactorDf.sum()/nProcs)
            luFactor.append(MatLUFactorDf.max())
            luFactorMin.append(MatLUFactorDf.min())

            for f in range(Nf):
                dofs[f].append((df.loc[fieldFilter])['dof'+str(f)].values[0])
                errors[f].append((df.loc[fieldFilter])['e'+str(f)].values[0])


            level = level + 1
        else:
            level = -1

    dofs   = np.array(dofs)
    errors = np.array(errors)

    times              = np.array(times)
    meanTime           = np.array(meanTime)
    timesMin           = np.array(timesMin)
    timeGrowthRate     = np.array(timeGrowthRate)

    flops              = np.array(flops)
    meanFlop           = np.array(meanFlop)
    flopsMax           = np.array(flopsMax)
    flopsMin           = np.array(flopsMin)
    flopGrowthRate     = np.array(flopGrowthRate)

    luFactor           = np.array(luFactor)
    luFactorMin        = np.array(luFactorMin)
    luFactorMean       = np.array(luFactorMean)
    luFactorGrowthRate = np.array(luFactorGrowthRate)


    data["Times"]                 = times
    data["Mean Time"]             = meanTime
    data["Times Range"]           = times-timesMin
    data["Time Growth Rate"]      = timeGrowthRate

    data["Flops"]                 = flops
    data["Mean Flops"]            = meanFlop
    data["Flop Range"]            = flopsMax - flopsMin
    data["Flop Growth Rate"]      = flopGrowthRate

    data["LU Factor"]             = luFactor
    data["LU Factor Mean"]        = luFactorMean
    data["LU Factor Range"]       = luFactor-luFactorMin
    data["LU Factor Growth Rate"] = luFactorGrowthRate

    for f in range(Nf):
        try:
            if cmdLineArgs.problem != 'NULL':
                file.addField(Field(file.fileName, config.fieldNames[cmdLineArgs.problem]['field '+str(f)]))
            else:
                file.addField(Field(file.fileName, str(f)))
        except:
            sys.exit('The problem you specified on the command line: ' + cmdLineArgs.problem + ' \ncould not be found' \
                ' please check ' + config.__file__ + ' to ensure that you are using the correct name/have defined the fields for the problem.')


    file.fileData = data
    for f in range(Nf):
        print('fieldList[f]', file.fieldList)
        file.fieldList[f].fieldData["dofs"]   = dofs[f]
        file.fieldList[f].fieldData["Errors"] = errors[f]


    #results.append(file)
    file.printFile()


    return file

def dataProces(cmdLineArgs, fileName):
    """
    This function takes a data file, ASCII type, for supplied as command line arguments and parses it into a multi-level
    dictionary, whose top level key is the file name, followed by data type, i.e. dofs, times, flops, errors, and
    the finale value is a NumPy array of the data to plot.  This is the used to generate a tasClasses File object

        data[<file name>][<data type>]:<numpy array>

    :param cmdLineArgs: Contains the command line arguments.
    :param fileName: Contains the name of file to be processed
    :type string:

    :returns:   data a tasClasses File object containing the parsed data from the file specified on the command line.
    """

    data = {}
    files   = []
    results = []
    #if -file/-f was left blank then this will automatically add every .py and .pyc
    #file to the files[] list to be processed.

    module             = importlib.import_module(fileName)
    Nf                 = getNf(module.Stages["ConvEst Refinement Level 1"]["ConvEst Error"][0]["error"])
    nProcs             = module.size
    dofs               = []
    errors             = []

    times              = []
    timesMin           = []
    meanTime           = []
    timeGrowthRate     = []

    flops              = []
    flopsMax           = []
    flopsMin           = []
    meanFlop           = []
    flopGrowthRate     = []

    luFactor           = []
    luFactorMin        = []
    luFactorMean       = []
    luFactorGrowthRate = []

    file               = File(module.__name__)

    for f in range(Nf):
        try:
            if cmdLineArgs.problem != 'NULL':
                file.addField(Field(file.fileName, config.fieldNames[cmdLineArgs.problem]['field '+str(f)]))
            else:
                file.addField(Field(file.fileName, str(f)))
        except:
            sys.exit('The problem you specified on the command line: ' + cmdLineArgs.problem + ' \ncould not be found' \
            ' please check ' + config.__file__ + ' to ensure that you are using the correct name/have defined the fields for the problem.')

    for f in range(Nf): errors.append([])
    for f in range(Nf): dofs.append([])

    #level set to 1 due to problems with coarse grid effecting measurements
    level  = 1
    while level >= 1:
        stageName = "ConvEst Refinement Level "+str(level)
        if stageName in module.Stages:
            timeTempMax  = module.Stages[stageName]["SNESSolve"][0]["time"]
            timeTempMin  = module.Stages[stageName]["SNESSolve"][0]["time"]
            totalTime    = module.Stages[stageName]["SNESSolve"][0]["time"]


            flopsTempMax  = module.Stages[stageName]["SNESSolve"][0]["flop"]
            flopsTempMin  = module.Stages[stageName]["SNESSolve"][0]["flop"]
            totalFlop    = module.Stages[stageName]["SNESSolve"][0]["flop"]

            luFactorTempMax = module.Stages[stageName]["MatLUFactorNum"][0]["time"] + \
                module.Stages[stageName]["MatLUFactorSym"][0]["time"]
            luFactorTempMin = luFactorTempMax
            totalLuFactor   = luFactorTempMax

            # print("Proc number: {} flops: {} running sum: {}".format(0, module.Stages[stageName]["SNESSolve"][0]["flop"],totalFlop))
            # print("************Level {}************".format(level))

            #This loops is used to grab the greatest time and flop when run in parallel
            for n in range(1, nProcs):
                #Sum of MatLUFactorNum and MatLUFactorSym
                if module.Stages[stageName]["MatLUFactorNum"][n]["time"] != 0:
                    luFactorCur = module.Stages[stageName]["MatLUFactorNum"][n]["time"] + \
                        module.Stages[stageName]["MatLUFactorSym"][n]["time"]

                #Gather Time information
                timeTempMax = timeTempMax if timeTempMax >= module.Stages[stageName]["SNESSolve"][n]["time"] \
                        else module.Stages[stageName]["SNESSolve"][n]["time"]
                timeTempMin = timeTempMin if timeTempMin <= module.Stages[stageName]["SNESSolve"][n]["time"] \
                        else module.Stages[stageName]["SNESSolve"][n]["time"]
                totalTime = totalTime + module.Stages[stageName]["SNESSolve"][n]["time"]

                #Gather Flop information
                flopsTempMax = flopsTempMax if flopsTempMax >= module.Stages[stageName]["SNESSolve"][n]["flop"] \
                        else module.Stages[stageName]["SNESSolve"][n]["flop"]
                flopsTempMin = flopsTempMin if flopsTempMin <= module.Stages[stageName]["SNESSolve"][n]["flop"] \
                        else module.Stages[stageName]["SNESSolve"][n]["flop"]
                totalFlop = totalFlop + module.Stages[stageName]["SNESSolve"][n]["flop"]

                #Gather LU factor information
                if module.Stages[stageName]["MatLUFactorNum"][n]["time"] != 0:
                    luFactorTempMax = luFactorTempMax if luFactorTempMax >= luFactorCur \
                            else luFactorCur
                    luFactorTempMin = luFactorTempMin if luFactorTempMin <= luFactorCur \
                            else luFactorCur
                    totalLuFactor = totalLuFactor + luFactorCur

                #print("Proc number: {} flops: {} running sum: {}".format(n, module.Stages[stageName]["SNESSolve"][n]["flop"],totalFlop))



            #The information from level 0 is NOT included.
            meanTime.append(totalTime/nProcs)
            times.append(timeTempMax)
            timesMin.append(timeTempMin)

            meanFlop.append(totalFlop/nProcs)
            flops.append(totalFlop)
            flopsMax.append(flopsTempMax)
            flopsMin.append(timeTempMin)
            if module.Stages[stageName]["MatLUFactorNum"][n]["time"] != 0:
                luFactor.append(luFactorTempMax)
                luFactorMin.append(luFactorTempMin)
                luFactorMean.append(totalLuFactor/nProcs)

            #Calculats the growth rate of statistics between levels
            if level > 1:
                timeGrowthRate.append(meanTime[level-1]/meanTime[level-2])
                flopGrowthRate.append(meanFlop[level-1]/meanFlop[level-2])
                #if module.Stages[stageName]["MatLUFactorNum"][n]["time"] != 0:
                #    luFactorGrowthRate.append(luFactorMean[level-1]/luFactorMean[level-2])

            for f in range(Nf):
                dofs[f].append(module.Stages[stageName]["ConvEst Error"][0]["dof"][f])
                errors[f].append(module.Stages[stageName]["ConvEst Error"][0]["error"][f])

            level = level + 1
        else:
            level = -1

    dofs   = np.array(dofs)
    errors = np.array(errors)

    times              = np.array(times)
    meanTime           = np.array(meanTime)
    timesMin           = np.array(timesMin)
    timeGrowthRate     = np.array(timeGrowthRate)

    flops              = np.array(flops)
    meanFlop           = np.array(meanFlop)
    flopsMax           = np.array(flopsMax)
    flopsMin           = np.array(flopsMin)
    flopGrowthRate     = np.array(flopGrowthRate)

    luFactor           = np.array(luFactor)
    luFactorMin        = np.array(luFactorMin)
    luFactorMean       = np.array(luFactorMean)
    luFactorGrowthRate = np.array(luFactorGrowthRate)


    data["Times"]                = times
    data["Mean Time"]            = meanTime
    data["Times Range"]          = times-timesMin
    data["Time Growth Rate"]     = timeGrowthRate

    data["Flops"]                = flops
    data["Mean Flops"]            = meanFlop
    data["Flop Range"]           = flopsMax - flopsMin
    data["Flop Growth Rate"]     = flopGrowthRate

    data["LU Factor"]             = luFactor
    data["LU Factor Mean"]        = luFactorMean
    data["LU Factor Range"]       = luFactor-luFactorMin
    data["LU Factor Growth Rate"] = luFactorGrowthRate


    file.fileData = data
    for f in range(Nf):
        file.fieldList[f].fieldData["dofs"]   = dofs[f]
        file.fieldList[f].fieldData["Errors"] = errors[f]

    #results.append(file)
    file.printFile()


    return file

def getNf(errorList):
    """
    This simple function takes the supplied error list and loops through that list until it encounters -1.  The default
    convention is that each field from the problem has an entry in the error list with at most 8 fields.  If there are
    less than 8 fields those entries are set to -1.
    Example:
      A problem with 4 fields would have a list of the form [.01, .003, .2, .04, -1, -1, -1, -1]

    :param errorList: contains a list of floating point numbers with the errors from each level of refinement.
    :type errorList: List containing Floating point numbers.
    :returns: Nf an integer that represents the number of fields.
    """
    i  = 0
    Nf = 0
    while errorList[i] != -1:
         Nf = Nf + 1
         i += 1
    return Nf

def getNfCSV(df):
    """
    This simple function is the same as getNf, except it is for the CSV files. It loops through
    the values of the dofx columns, where x is an integer, from the row where
    Stage Name = ConvEst Refinement Level 0, Event Name = ConvEst Error, and Rank = 0 until it
    encounters -1.  The default convention is that each field from the problem has an entry in the error list with at most
    8 fields.  If there are less than 8 fields those entries are set to -1.
    Example:
      A problem with 4 fields would have a list of the form [.01, .003, .2, .04, -1, -1, -1, -1]

    :param df: Contains a Pandas Data Frame.
    :type df: A Pandas Data Frame object.
    :returns: Nf an integer that represents the number of fields.
    """
    #Get a single row from the Data Frame that contains the field information
    df = df.loc[(df['Event Name']=='ConvEst Error') & (df['Stage Name']=='ConvEst Refinement Level 0')\
        & (df['Rank']==0)].reset_index()
    level = 1
    while level >= 1:
        dof = 'dof' + str(level)
        if df.loc[0,dof] == -1:
            break
        else:
            level = level + 1
    return level

def graphGen(file, enable_graphs, graph_flops_scaling, dim):
    """
    This function takes the supplied dictionary and plots the data from each file on the Mesh Convergence, Static Scaling, and
    Efficacy graphs.

    :param file: Contains the data to be ploted on the graphs, assumes the format -- file[<file name>][<data type>]:<numpy array>
    :type file: Dictionary
    :param graph_flops_scaling: Controls creating the scaling graph that uses flops/second.  The default is not to.  This option
                                    is specified on the command line.
    :type graph_flops_scaling: Integer
    :param dim: Contains the number of dimension of the mesh.  This is specified on the command line.
    :type dim: Integer


    :returns: None
    """
    lstSqMeshConv = np.empty([2])

    counter = 0
    #Loop through each file and add the data/line for that file to the Mesh Convergence, Static Scaling, and Efficacy Graphs
    for field in file.fieldList:
        #Least squares solution for Mesh Convergence
        lstSqMeshConv[0], lstSqMeshConv[1] = leastSquares(field.fieldData['dofs'], field.fieldData['Errors'])


        print("Least Squares Data")
        print("==================")
        print("Mesh Convergence")
        print("Alpha: {} \n  {}".format(lstSqMeshConv[0], lstSqMeshConv[1]))

        convRate = lstSqMeshConv[0] * -dim
        print('convRate: {} of {} data'.format(convRate,file.fileName))

        field.setConvergeRate(convRate)
        field.setAlpha(lstSqMeshConv[0])
        field.setBeta(lstSqMeshConv[1])
    #file.writeCSV()

    if cmdLineArgs.enable_graphs == 1:
        #Uses the specified style sheet for generating the plots
        styleDir = os.path.join(os.environ.get('PETSC_DIR'), 'lib/petsc/bin')
        plt.style.use(os.path.join(styleDir, 'petsc_tas_style.mplstyle'))

        #Set up plots with labels
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
        axEffic.set_ylim(0,10)


        #Loop through each file and add the data/line for that file to the Mesh Convergence, Static Scaling, and Efficacy Graphs
        for field in file.fieldList:
            ##Start Mesh Convergence graph
            convRate = str(convRate)

            x, = axMeshConv.loglog(field.fieldData['dofs'], field.fieldData['Errors'],
                label = 'Field ' + field.fieldName + ' Orig Data', marker = "^")

            meshConvOrigHandles.append(x)

            y, = axMeshConv.loglog(field.fieldData['dofs'], ((field.fieldData['dofs']**lstSqMeshConv[0] * 10**lstSqMeshConv[1])),
                    label = field.fieldName + " Convergence rate =  " + convRate, marker = "x" )

            meshConvLstSqHandles.append(y)

            ##Start Static Scaling Graph, only if graph_flops_scaling equals 1.  Specified on the command line.
            if graph_flops_scaling == 1 :
                x, =axStatScale.loglog(file.fileData['Times'], file.fileData['Flops']/file.fileData['Times'],
                label = 'Field ' + field.fieldName, marker = "^")

            #statScaleHandles.append(x)
            ##Start Static Scaling with DoFs Graph
            x, =axStatScale.loglog(file.fileData['Times'], field.fieldData['dofs']/file.fileData['Times'],
                label = 'Field ' + field.fieldName, marker = "^")

            statScaleHandles.append(x)
            ##Start Efficacy graph
            x, = axEffic.semilogx(file.fileData['Times'], -np.log10(field.fieldData['Errors']*file.fileData['Times']),
                label = 'Field ' + field.fieldName, marker = "^")

            efficHandles.append(x)

            counter = counter + 1

            meshConvHandles = meshConvOrigHandles + meshConvLstSqHandles
            meshConvLabels = [h.get_label() for h in meshConvOrigHandles]
            meshConvLabels = meshConvLabels + [h.get_label() for h in meshConvLstSqHandles]
            meshConvFig.legend(handles = meshConvHandles, labels = meshConvLabels)

            statScaleLabels = [h.get_label() for h in statScaleHandles]
            statScaleFig.legend(handles = statScaleHandles, labels = statScaleLabels)

            efficLabels = [h.get_label() for h in efficHandles]
            efficFig.legend(handles = efficHandles, labels = efficLabels)

        meshConvFig.savefig(config.filePath['absoluteGraphs']+'meshConvergenceField_' + field.fileName + '.png')
        statScaleFig.savefig(config.filePath['absoluteGraphs']+'staticScalingField_' + field.fileName + '.png')
        efficFig.savefig(config.filePath['absoluteGraphs']+'efficacyField_' + field.fileName + '.png')

    return

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

    cmdLine.add_argument('-enable_graphs', '--enable_graphs', type = int, default = 1, choices = [0, 1],
    help = 'Enables graphing. Default: %(default)s  print the graphs. 0 to disable printing the graphs')

    cmdLine.add_argument('-view_variance', '--view_variance', type = int, default = 0, choices = [0, 1],
    help = 'Enables calculating and outputting the Variance. Default: %(default)s does not print the variance. 1 to enable \
        printing the graphs')

    cmdLine.add_argument('-problem', '--problem', default = 'NULL', help = 'Enables searching for the names of fields in \
        configureTAS.py. Default: %(default)s does not look for the names.  Instead identifies the fields using \
        a number, 0, 1, 2,...n')

    cmdLineArgs = cmdLine.parse_args()


    main(cmdLineArgs)
