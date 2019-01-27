#!/usr/bin/env python
"""
This module holds configuration information for the petsc_tas_analysis.py in
  a dictionary format.  This module takes no parameters.

  The key absoluteGraphs has contains the string of the absolute path to place
   the generated graphs.

   The key absoluteData has contains the string of the absolute path to look
    for the file containing the information to be analyzed.  Within petsc_tas_analysis
    the system.path variable is appended to include this path.

"""
filePath = {}
filePath['absoluteGraphs'] = '/home/arcowie/petscTas/graphs/'
filePath['absoluteData']   = '/home/arcowie/petscTas/data'

fieldNames = {}
fieldNames['ex62'] = {}
fieldNames['ex62']['field 0'] = 'Velocity'
fieldNames['ex62']['field 1'] = 'Pressure'
