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
import sys
import pickle

# list of classes found
classes = {}
enums = {}
aliases = {}
senums = {} # like enums except strings instead of integer values
structs = {}


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

          ivalues = []
          for i in values:
            if i[0] == " ": i = i[1:]
            ivalues.append(i)

          enums[name] = ivalues
          break
        line = f.readline()
        struct = struct + line
    line = f.readline()
  f.close()

def getsenums(filename):
  import re
  regdefine   = re.compile('#define [A-Za-z]*Type ')
  regblank    = re.compile(' [ ]*')
  f = open(filename)
  line = f.readline()
  while line:
    fl = regdefine.search(line)
    if fl:
      senum = fl.group(0)[8:-1]
      senums[senum] = {}
      line = regblank.sub(" ",f.readline().strip())
      while line:
        values = line.split(" ")
        senums[senum][values[1]] = values[2]
        line = regblank.sub(" ",f.readline().strip())
    line = f.readline()
  f.close()

def getstructs(filename):
  import re
  regtypedef  = re.compile('^typedef [ ]*struct {')
  regcomment  = re.compile('/\* [A-Za-z _(),<>|^\*/0-9.]* \*/')
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
          struct = struct.replace("typedef struct {","")
          struct = regblank.sub(" ",struct)
          struct = struct.replace("; ",";")
          struct = regcomment.sub("",struct)

          name = regname.search(struct)
          name = name.group(0)
          name = name.replace("} ","")

          values = struct[struct.find("{")+1:struct.find(";}")]
          if not values.find('#') == -1: break
          values = values.split(";")
          ivalues = []
          for i in values:
            if i[0] == " ": i = i[1:]
            ivalues.append(i)
          structs[name] = ivalues
          break
        line = f.readline()
        struct = struct + line
    line = f.readline()
  f.close()

def getclasses(filename):
  import re
  regclass    = re.compile('typedef struct _[pn]_[A-Za-z_]*[ ]*\*')
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
  regarg      = re.compile('\([A-Za-z*_\[\]]*[,\)]')
  regerror    = re.compile('PetscErrorCode')

  rejects     = ['PetscErrorCode','DALocalFunction','...','<','(*)','(**)','off_t','MPI_Datatype','va_list','size_t','PetscStack']
  #
  # search through list BACKWARDS to get the longest match
  #
  classlist   = classes.keys()
  classlist.sort()
  classlist.reverse()
  f = open(filename)
  line = f.readline()
  while line:
    fl = regfun.search(line)
    if fl:
      struct = line
      struct = regfun.sub("",struct)
      struct = regcomment.sub("",struct)
      struct = struct.replace("unsigned ","u")
      struct = regblank.sub("",struct)
      struct = struct.replace("\n","")
      struct = struct.replace("const","")
      struct = struct.replace(";","")
      struct = struct.strip()
      fl = regarg.search(struct)
      if fl:
        arg = fl.group(0)
        arg = arg[1:-1]
        reject = 0
        for i in rejects:
          if struct.find(i) > -1:
            reject = 1
        if  not reject:
          args = struct[struct.find("(")+1:struct.find(")")]
          args = args.split(",")
          if args == ['void']: args = []
          name = struct[:struct.find("(")]
          for i in classlist:
            if name.startswith(i):
              classes[i][name[len(i):]] = args
              break


    line = f.readline()

  f.close()
#
#  For now, hardwire aliases
#
def getaliases():
  aliases['ulong']              = 'unsigned long'
  aliases['ushort']             = 'unsigned short'
  aliases['uchar']              = 'unsigned char'
  aliases['PetscInt']           = 'int'
  aliases['PetscScalar']        = 'double'
  aliases['PetscReal']          = 'double'
  aliases['MPI_Comm']           = 'int'
  aliases['MPI_Request']        = 'int'
  aliases['FILE']               = 'int'
  aliases['PetscMPIInt']        = 'int'
  aliases['PetscClassId']        = 'int'
  aliases['PetscLogDouble']     = 'double'
  aliases['PetscTablePosition'] = 'int*'
  aliases['ISColoringValue']    = 'ushort'
  aliases['PetscLogEvent']      = 'int'
  # for HDF5
  aliases['hid_t']              = 'int'

def main(args):
  for i in args:
    getenums(i)
  for i in args:
    getsenums(i)
  getaliases()
  for i in args:
    getstructs(i)
  # this classes ONLY have static methods
  classes['Petsc'] = {}
  classes['PetscLog'] = {}
  classes['PetscSort'] = {}
  classes['PetscStr'] = {}
  classes['PetscBinary'] = {}
  classes['PetscOptions'] = {}
  classes['PetscMalloc'] = {}
  classes['PetscToken'] = {}
  for i in args:
    getclasses(i)
  for i in args:
    getfunctions(i)
  file = open('classes.data','w')
  pickle.dump(enums,file)
  pickle.dump(senums,file)
  pickle.dump(structs,file)
  pickle.dump(aliases,file)
  pickle.dump(classes,file)



#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__':
  main(sys.argv[1:])

