#!/usr/bin/env python3
import numpy as np
import csv

class File(object):
    #Class constructor
    def __init__(self, fileName):
        self.fileName     = fileName
        self.numberFields = 0
        self.fieldList    = []
        self.fileData     = {}

    #Adds a field to the field list and increases the number by 1
    def addField(self, field):
        self.fieldList.append(field)
        self.numberFields = self.numberFields +1

    #Prints the content of the object
    def printFile(self):
        print('\t\t*******************Data for {}***************************'.format(self.fileName))
        np.set_printoptions(precision=3, linewidth=100)
        for k in self.fileData:
            print(" {: >18} : {}\n".format(k,self.fileData[k]))
        for field in self.fieldList:
            field.printField()

    #Writes its data as a series of CSV files.  One file for the
    #main body of data and one for each field.
    def writeCSV(self):
        with open(self.fileName + '.csv', mode='w') as csv_file:
            columnNames = ['Stage','Max Time', 'Mean Time', 'Max Giga Flops', 'Mean Giga Flops', 'LU Factor', 'LU Factor Mean']
            row = {}
            numStages = len(self.fileData['Times'])
            writer = csv.DictWriter(csv_file, fieldnames = columnNames, restval='N/A')
            writer.writeheader()
            for stage in range(numStages):
                row.clear()
                for item in columnNames:
                    if item == 'Stage':
                        row[item]=stage
                    elif item == 'Max Time':
                        row[item] =  '{:.3g}'.format((self.fileData['Times'][stage]))
                    elif item == 'Max Giga Flops':
                        row[item] =  '{:.3g}'.format((self.fileData['Flops'][stage])/1000000000)
                    elif item == 'Mean Giga Flops':
                        row[item] =  '{:.3g}'.format((self.fileData['Mean Flops'][stage])/1000000000)
                    else:
                        row[item]='{:.3g}'.format((self.fileData[item][stage]))
                writer.writerow(row)
        for field in self.fieldList:
            with open(self.fileName + '_' + field.fieldName+'.csv', mode='w') as csv_file:
                columnNames = ['Stage', 'dofs', 'Errors', 'Alpha', 'Beta', 'Convergence Rate']
                writer = csv.DictWriter(csv_file, fieldnames = columnNames, restval='N/A')
                writer.writeheader()
                for stage in range(numStages):
                    row.clear()
                    for item in columnNames:
                        if item == 'Stage':
                            row[item] = stage
                        elif item == 'Alpha':
                            row[item] =  '{:.3g}'.format(field.alpha)
                        elif item == 'Beta':
                            row[item] =  '{:.3g}'.format(field.beta)
                        elif item == 'Convergence Rate':
                            row[item] =  '{:.3g}'.format(field.cRate)
                        else:
                            row[item] = '{:.3g}'.format((field.fieldData[item][stage]))
                    writer.writerow(row)



class Field(File):
    #Class constructor
    def __init__(self, fileName, fieldName, alpha=0, cRate=0, beta=0):
        File.__init__(self, fileName)
        self.fieldName = fieldName
        self.fieldData = {}
        self.alpha     = alpha
        self.cRate     = cRate
        self.beta      = beta

    def setAlpha(self, alpha):
        self.alpha = alpha

    def setBeta(self, beta):
        self.beta = beta

    def setConvergeRate(self, cRate):
        self.cRate = cRate

    def printField(self):
        print('**********Data for Field {}************'.format(self.fieldName))
        for k in self.fieldData:
            print(" {: >18} : {}\n".format(k,self.fieldData[k]))
