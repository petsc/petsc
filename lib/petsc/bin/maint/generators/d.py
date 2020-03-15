#!/usr/bin/env python
#!/bin/env python
# $Id: adprocess.py,v 1.12 2001/08/24 18:26:15 bsmith Exp $
#
# change python to whatever is needed on your system to invoke python
#
#  Reads classes.data and prints the language d classes
#
#  Crude as all hack!
#
#  Calling sequence:
#      d.py
##
import os
import re
import sys
from string import *
import pickle


def main(args):
  file = open('classes.data')
  enums   = pickle.load(file)
  senums  = pickle.load(file)
  structs = pickle.load(file)
  aliases = pickle.load(file)
  classes = pickle.load(file)
  outfile = open('petsc.d','w')

  for i in aliases:
    outfile.write("alias "+aliases[i]+" "+i+"; \n")
  outfile.write("\n")

  for i in senums:
    outfile.write("alias char* "+i+"; \n")
#    for j in senums[i]:
#      outfile.write("alias "+senums[i][j]+" "+j+"; \n")

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

  for i in structs:
    outfile.write("struct "+i+"\n")
    outfile.write("{\n")
    for j in structs[i]:
      outfile.write("    "+j+";\n")
    outfile.write("};\n")
  outfile.write("\n")

  for i in classes:
    outfile.write("class "+i+"\n")
    outfile.write("{\n")
    for j in classes[i]:
      outfile.write("  int "+j+"(")
      cnt = 0
      for k in classes[i][j]:
        if cnt > 0:
          outfile.write(k.replace("const ","").replace("unsigned long","ulong"))
          if cnt < len(classes[i][j])-1: outfile.write(",")
        cnt = cnt + 1
      outfile.write("){return 0;};\n")
    outfile.write("}\n")


#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__':
  main(sys.argv[1:])

