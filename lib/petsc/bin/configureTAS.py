#!/usr/bin/env python3
"""
This module holds configuration information for the petsc_tas_analysis.py in
  a dictionary format.  This module takes no parameters.

The dict filePath holds key value pairs for file paths:
  The key absoluteGraphs has contains the string of the absolute path to place
  the generated graphs.

The key absoluteData has contains the string of the absolute path to look
  for the file containing the information to be analyzed.  Within petsc_tas_analysis
  the system.path variable is appended to include this path.

The dict fieldNames holds key value pairs for the fields of problems.  This is
  values for the level are also dicts since many problems have multiple fields.
  The key ex62 contains the name of fields for example 62, ie ex62.c
"""

filePath = {}
filePath['defaultData']=''
filePath['defaultGraphs']=''

fieldNames = {}
fieldNames['ex62'] = {}
fieldNames['ex62']['field 0'] = 'Velocity'
fieldNames['ex62']['field 1'] = 'Pressure'
