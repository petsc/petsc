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
  outfile = open('petsc.hh','w')

  def ClassToPointer(a):
    if a in classes: return a+"*"
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
    outfile.write("class "+i+";\n")
  outfile.write("\n")
  
  skeys = structs.keys()
  skeys.sort()
  for i in skeys:
    outfile.write("struct "+i+"\n")
    outfile.write("{\n")
    for j in structs[i]:
      l = j[:j.find(" ")]
      if l in classes: j = l+"* "+j[j.find(" "):]
      outfile.write("    "+ClassToPointer(j)+";\n")
    outfile.write("};\n")      
  outfile.write("\n")

  if not os.path.isdir('matlab'): os.mkdir('matlab')
  skeys = classes.keys()
  skeys.sort()
  for i in skeys:
    if not os.path.isdir('matlab/@'+i): os.mkdir('matlab/@'+i)
    fd = open('matlab/@'+i+'/'+i+'.m','w')
    fd.write('function OUT = '+i+'()\n')
    fd.write("S = struct('id',PetscMex('mexFunction"+i+"'));\n")
    fd.write("OUT = class(S,'"+i+"');\n")
    fd.close()
    fd = open('matlab/@'+i+'/GetId.m','w')
    fd.write('function OUT = GetId(i0)\n')
    fd.write("OUT = i0.id;\n")
    fd.close()
    fd = open('matlab/'+i+'Createmex.c','w')
    fd.write('#include "petscts.h"\n')
    fd.write('#include "petscdmmg.h"\n')                
    fd.write('#include "mex.h"\n')
    fd.write('void mexFunction'+i+'(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])\n{\n')
    fd.write('  plhs[0]  = mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);')
    fd.write('  '+i+'Create(PETSC_COMM_WORLD,('+i+'*)mxGetPr(plhs[0]));')
    fd.write('  return;\n}\n')
    fd.close()
        
    sskeys = classes[i].keys()
    sskeys.sort()
    for j in sskeys:
      if len(classes[i][j]) < 1 or not classes[i][j][0] == i and not j == 'Create':
        fd = open('matlab/'+i+j+'.m','w')
        fd.write('function '+i+j+'(')        
        cnt = 0
        for k in classes[i][j]:
          fd.write('i'+str(cnt))
          if cnt < len(classes[i][j])-1: fd.write(",")
          cnt = cnt + 1
        fd.write(')\n')
        fd.write("PetscMex('mexFunction"+i+j+"'")
        if classes[i][j]: fd.write(",")
        cnt = 0
        for k in classes[i][j]:
          if k in classes:
            fd.write('GetId(i'+str(cnt)+')')            
          else:
            fd.write('i'+str(cnt))
          if cnt < len(classes[i][j])-1: fd.write(",")
          cnt = cnt + 1
        fd.write(');\n')
        fd.close()
        buildmex(i,j,classes)
    for j in sskeys:
      if len(classes[i][j]) > 0 and classes[i][j][0] == i and not j == 'Destroy' and not j == 'Create':
        fd = open('matlab/@'+i+'/'+j+'.m','w')
        fd.write('function '+j+'(')        
        cnt = 0
        for k in classes[i][j]:
          fd.write('i'+str(cnt))
          if cnt < len(classes[i][j])-1: fd.write(",")
          cnt = cnt + 1
        fd.write(')\n')
        fd.write("PetscMex('mexFunction"+i+j+"'")
        if classes[i][j]: fd.write(",")
        cnt = 0
        for k in classes[i][j]:
          if k in classes:
            fd.write('GetId(i'+str(cnt)+')')            
          else:
            fd.write('i'+str(cnt))
          if cnt < len(classes[i][j])-1: fd.write(",")
          cnt = cnt + 1
        fd.write(');\n')
        fd.close()
        buildmex(i,j,classes)

    fd = open('matlab/makefile','w')
    fd.write('LOCDIR   = 0\n')
    fd.write('include ${PETSC_DIR}/conf/base\n')
    fd.write('include ${PETSC_DIR}/conf/test\n')
    fd.write("mexs:\n\t${MATLAB_MEX} -output PetscMex *.c -g CC=${CC} CFLAGS='${COPTFLAGS} ${CFLAGS} ${CCPPFLAGS}' ${PETSC_TS_LIB}\n")
    fd.close()

    # universal Matlab mex function called by all functions/methods
    fd = open('matlab/PetscMex.c','w')
    fd.write('#include "mex.h"\n')
    fd.write('#include "petscsys.h"\n')
    fd.write('#include "petscfix.h"\n')    
    fd.write('#if defined(PETSC_HAVE_PWD_H)\n')
    fd.write('#include <pwd.h>\n')
    fd.write('#endif\n')
    fd.write('#include <ctype.h>\n')
    fd.write('#include <sys/types.h>\n')
    fd.write('#include <sys/stat.h>\n')
    fd.write('#if defined(PETSC_HAVE_UNISTD_H)\n')
    fd.write('#include <unistd.h>\n')
    fd.write('#endif\n')
    fd.write('#if defined(PETSC_HAVE_STDLIB_H)\n')
    fd.write('#include <stdlib.h>\n')
    fd.write('#endif\n')
    fd.write('#if defined(PETSC_HAVE_SYS_UTSNAME_H)\n')
    fd.write('#include <sys/utsname.h>\n')
    fd.write('#endif\n')
    fd.write('#if defined(PETSC_HAVE_WINDOWS_H)\n')
    fd.write('#include <windows.h>\n')
    fd.write('#endif\n')
    fd.write('#include <fcntl.h>\n')
    fd.write('#include <time.h>\n')
    fd.write('#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)\n')
    fd.write('#include <sys/systeminfo.h>\n')
    fd.write('#endif\n')
    fd.write('#if defined(PETSC_HAVE_DLFCN_H)\n')
    fd.write('#include <dlfcn.h>\n')
    fd.write('#endif\n')
    
    fd.write('void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])\n{\n')
    fd.write('  void (*f)(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]);\n')
    fd.write('  void *handle;\n')
    fd.write('  char buffer[256];\n\n')
    
    fd.write('#if defined(PETSC_HAVE_LOADLIBRARY)\n')
    fd.write('  handle = LoadLibrary(0);\n')
    fd.write('#elif defined(PETSC_HAVE_RTLD_GLOBAL)\n')
    fd.write('  handle = dlopen(0,RTLD_LAZY | RTLD_GLOBAL);\n')
    fd.write('#else\n')
    fd.write('  handle = dlopen(0,RTLD_LAZY);\n')
    fd.write('#endif\n\n')

    fd.write('  if (!mxIsChar(prhs[0])) return;\n')
    fd.write('  mxGetNChars(prhs[0], buffer, 256);\n\n')
  
    fd.write('#if defined(PETSC_HAVE_GETPROCADDRESS)\n')
    fd.write('  f = (void (*)(int,mxArray *[],int,const mxArray *[]))GetProcAddress((HMODULE)handle,buffer);\n')
    fd.write('#else\n')
    fd.write('  f = (void (*)(int,mxArray *[],int,const mxArray *[]))dlsym(handle,buffer);\n')
    fd.write('#endif\n\n')

    fd.write('  (*f)(nlhs,plhs,nrhs-1,prhs+1);\n')    
    fd.write('  return;\n}\n')
    fd.close()
  
        
def buildmex(i,j,classes):
  fd = open('matlab/'+i+j+'mex.c','w')
  fd.write('#include "petscts.h"\n')
  fd.write('#include "petscdmmg.h"\n')                
  fd.write('#include "mex.h"\n')
  fd.write('void mexFunction'+i+j+'(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])\n{\n')
  
  cnt = 0
  for k in classes[i][j]:
    if k in classes:
      fd.write('  '+k+' i'+str(cnt)+' = ('+k+') (int) mxGetScalar(prhs['+str(cnt)+']);\n')
    elif k in ['PetscInt','PetscScalar']:
      fd.write('  '+k+' i'+str(cnt)+' = ('+k+') mxGetScalar(prhs['+str(cnt)+']);\n')            
    elif k in ['char*']:
      fd.write('  char i'+str(cnt)+'[256]; mxGetNChars(prhs['+str(cnt)+'], i'+str(cnt)+', 256);\n')
    elif k.endswith('Type'):
      fd.write('  const char i'+str(cnt)+'[256]; mxGetNChars(prhs['+str(cnt)+'], (char*)i'+str(cnt)+', 256);\n')
    else:
      fd.write('  '+k+' i'+str(cnt)+' = ('+k+') 0;\n')
      pass
    cnt = cnt + 1
  fd.write('  '+i+j+'(')        
  cnt = 0
  for k in classes[i][j]:
    fd.write('i'+str(cnt))
    if cnt < len(classes[i][j])-1: fd.write(",")
    cnt = cnt + 1
  fd.write(');\n')
  fd.write('  return;\n}\n')
    
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
  main(sys.argv[1:])

