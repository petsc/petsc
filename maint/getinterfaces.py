#!/usr/bin/env python
#!/bin/env python
# $Id: adprocess.py,v 1.12 2001/08/24 18:26:15 bsmith Exp $ 
#
# change python to whatever is needed on your system to invoke python
#
#  Processes PETSc's include/petsc*.h files to determine
#  the PETSc enums, functions and classes
#
#  Crude as all hack!
#
#  Calling sequence: 
#      getinterfaces *.h
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

def getenums(filename):
  import re
  regtypedef  = re.compile('typedef [ ]*enum')
  regcomment  = re.compile('/\* [A-Za-z _(),<>|^\*]* \*/')
  reg         = re.compile('}')
  regblank    = re.compile(' [ ]*')
  regname     = re.compile('}[ A-Za-z]*')
  f = open(filename)
  line = f.readline()
  while line:
    fl = regtypedef.search(line)
    if fl:
      struct = line
      while line:
        fl = reg.search(line)
        if fl:
	  struct = struct.replace("\\","")
  	  struct = struct.replace("\n","")
  	  struct = struct.replace(";","")
  	  struct = struct.replace("typedef enum","")	  
          struct = regcomment.sub("",struct)
          struct = regblank.sub(" ",struct)

          name = regname.search(struct)
          name = name.group(0)
          name = name.replace("} ","")

          values = struct[struct.find("{")+1:struct.find("}")]
          values = values.split(",")

          if struct.find("=") == -1:
            for i in range(len(values)):
              values[i] = values[i] + " = " + str(i)
            
	  enums[name] = values
          break
        line = f.readline()
        struct = struct + line
    line = f.readline()
  f.close()

def getclasses(filename):
  import re
  regclass    = re.compile('typedef struct _p_[A-Za-z_]*[ ]*\*')
  regcomment  = re.compile('/\* [A-Za-z _(),<>|^\*]* \*/')
  regblank    = re.compile(' [ ]*')
  regsemi     = re.compile(';')  
  f = open(filename)
  line = f.readline()
  while line:
    fl = regclass.search(line)
    if fl:
      struct = line
      struct = regclass.sub("",struct)
      struct = regcomment.sub("",struct)      
      struct = regblank.sub("",struct)
      struct = regsemi.sub("",struct)      
      struct = struct.replace("\n","")
      classes[struct] = {}
    line = f.readline()
  
  f.close()

def getfunctions(filename):
  import re
  regfun      = re.compile('EXTERN PetscErrorCode PETSC[A-Z]*_DLLEXPORT ')
  regcomment  = re.compile('/\* [A-Za-z _(),<>|^\*]* \*/')
  regblank    = re.compile(' [ ]*')
  regarg      = re.compile('\([A-Za-z]*[,\)]')
  regerror    = re.compile('PetscErrorCode')
  f = open(filename)
  line = f.readline()
  while line:
    fl = regfun.search(line)
    if fl:
      struct = line
      struct = regfun.sub("",struct)
      struct = regcomment.sub("",struct)      
      struct = regblank.sub(" ",struct)
      struct = struct.replace("\n","")
      struct = struct.replace(";","")      
      fl = regarg.search(struct)
      if fl:
        arg = fl.group(0)
        arg = arg[1:-1]
	if struct.find("PetscErrorCode") == -1 and struct.find("...") == -1 and struct.find("<") == -1 and arg in classes:
          args = struct[struct.find("(")+1:struct.find(")")]
          args = args.split(",")
          name = struct[:struct.find("(")]
	  classes[arg][name] = args
      
    line = f.readline()
  
  f.close()

def main(args):
  for i in args:
    getenums(i)
  for i in args:
    getclasses(i)
  for i in args:
    getfunctions(i)
  file = open('classes.data','w')
  pickle.dump(enums,file)
  pickle.dump(classes,file)  

  
    
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
  main(sys.argv[1:])

