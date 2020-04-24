#!/usr/bin/env python
#!/bin/env python
# $Id: adprocess.py,v 1.12 2001/08/24 18:26:15 bsmith Exp $
#
# change python to whatever is needed on your system to invoke python
#
#  Reads classes.data and prints the C++ classes
#
#  Crude as all hack!
#
#  Calling sequence:
#      c++.py
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
  outfile = open('petsc.cpp','w')

  def ClassToPointer(a):
    if a in classes: return a+"_*"
    else: return a

  for i in aliases:
    outfile.write("typedef "+aliases[i]+" "+i+"; \n")
  outfile.write("\n")

  skeys = senums.keys()
  skeys.sort()
  for i in skeys:
    outfile.write("#define "+i+" char*\n")
  outfile.write("\n")

  skeys = enums.keys()
  skeys.sort()
  for i in skeys:
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

  skeys = classes.keys()
  skeys.sort()
  for i in skeys:
    outfile.write("class "+i+"_;\n")
  outfile.write("\n")

  skeys = structs.keys()
  skeys.sort()
  for i in skeys:
    outfile.write("struct "+i+"_\n")
    outfile.write("{\n")
    for j in structs[i]:
      l = j[:j.find(" ")]
      if l in classes:
        j = l+"* "+j[j.find(" "):]
      outfile.write("    "+ClassToPointer(j)+";\n")
    outfile.write("};\n")
  outfile.write("\n")

  skeys = classes.keys()
  skeys.sort()
  for i in skeys:
    outfile.write("class "+i+"\n")
    outfile.write("{\n")
    sskeys = classes[i].keys()
    sskeys.sort()
    for j in sskeys:
      if not classes[i][j] or not classes[i][j][0] == i:
        outfile.write("  static ")
        outfile.write("  int "+j+"(")
        cnt = 0
        for k in classes[i][j]:
          if cnt > 0 or not k == i:
            outfile.write(ClassToPointer(k))
            if cnt < len(classes[i][j])-1: outfile.write(",")
          cnt = cnt + 1
        outfile.write("){return 0;};\n")
    for j in sskeys:
      if classes[i][j] and (classes[i][j][0] == i and not j == 'Destroy'):
        outfile.write("  int "+j+"(")
        cnt = 0
        for k in classes[i][j]:
          if cnt > 0 or not k == i:
            outfile.write(ClassToPointer(k))
            if cnt < len(classes[i][j])-1: outfile.write(",")
          cnt = cnt + 1
        outfile.write("){return 0;};\n")
    outfile.write("};\n")


#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__':
  main(sys.argv[1:])

