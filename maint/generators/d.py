#!/usr/bin/env python
#!/bin/env python
# $Id: adprocess.py,v 1.12 2001/08/24 18:26:15 bsmith Exp $ 
#
# change python to whatever is needed on your system to invoke python
#
#  Reads classes.data and prints the information out nicely
#
#  Crude as all hack!
#
#  Calling sequence: 
#      prettyprint.py
##
import os
import re
from exceptions import *
import sys
from string import *
import pickle

# list of classes found
classes = {}
enums = {}


def main(args):
  file = open('classes.data')
  enums   = pickle.load(file)
  classes = pickle.load(file)
  outfile = open('petsc.d','w')
  
  for i in enums:
    outfile.write("enum "+i+"\n")
    outfile.write("{\n")
    cnt = 0
    for j in enums[i]:
      outfile.write("    "+j)
      cnt = cnt + 1
      if not cnt == len(enums[i]): outfile.write(",")
      outfile.write("\n")
    outfile.write("};\n")      
  outfile.write("\n")

  for i in classes:
    outfile.write("class "+i+"{}\n")

  for i in classes:
    outfile.write("class "+i+"\n")
    outfile.write("{\n")
    for j in classes[i]:
      outfile.write("  int "+j+"(")
      cnt = 0
      for k in classes[i][j]:
        outfile.write(k.replace("const ",""))
        cnt = cnt + 1
        if not cnt == len(classes[i][j]): outfile.write(",")
      outfile.write(");\n")
    outfile.write("}\n")        
  
    
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
  main(sys.argv[1:])

