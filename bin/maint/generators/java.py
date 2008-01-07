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
from exceptions import *
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
  outfile = open('petsc.java','w')

  for i in enums:
    outfile.write("enum "+i+"\n")
    outfile.write("{\n")
    cnt = 0
    for j in enums[i]:
      outfile.write("    "+j.replace("=","(")+")")
      cnt = cnt + 1
      if not cnt == len(enums[i]): outfile.write(",")
      outfile.write("\n")
    outfile.write(";private final int value;")
    outfile.write(i+"(int v) { this.value = v; }")
    outfile.write("}\n")      
  outfile.write("\n")

  for i in structs:
    outfile.write("class "+i+"\n")
    outfile.write("{\n")
    for k in structs[i]:
      k = k.replace("*","")
      for l in aliases: k = k.replace(l,aliases[l])
      k = k.replace("char[]","String").replace("char*","String").replace("char","String");
      for l in senums: k = k.replace(l,"String")
      k = k.replace("void","int")
      k = k.replace("PetscTruth","boolean")
      k = k.replace("*","")
      k = k.replace("unsigned","")
      k = k.replace("const","")
      k = k.replace("ushort","short")                    
      outfile.write("    "+k+";\n")
    outfile.write("};\n")      
  outfile.write("\n")

  for i in classes:
    outfile.write("class "+i+"\n")
    outfile.write("{\n")
    for j in classes[i]:
      outfile.write(" void "+j+"(")
      cnt = 0
      for k in classes[i][j]:
        k = k.replace("*","")
        for l in aliases: k = k.replace(l,aliases[l])
        k = k.replace("char[]","String").replace("char*","String").replace("char","String");
        for l in senums: k = k.replace(l,"String")
        k = k.replace("void","int")
        k = k.replace("PetscTruth","boolean")
        k = k.replace("*","")
        k = k.replace("unsigned","")
        k = k.replace("ushort","short")                            
        if cnt > 0:
          outfile.write(k.replace("const ","").replace("unsigned long","ulong"))
          outfile.write(" a"+str(cnt))
          if cnt < len(classes[i][j])-1: outfile.write(",")
        cnt = cnt + 1
      outfile.write("){;}\n")
    outfile.write("}\n")        
  
    
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
  main(sys.argv[1:])

