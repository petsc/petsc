#!/usr/bin/env python
#!/bin/env python
#
#    Generates fortran stubs for PETSc using Sowings bfort program
#
#
# 
import os
import re
from exceptions import *
import sys
from string import *
import commands

#
#  Opens all generated files and fixes them; also generates list in makefile.src

def FixFile(filename):
	  ff = open(filename)
	  data = ff.read()
          ff.close()

          # gotta be a better way to do this
	  data = re.subn('\nvoid ','\nvoid PETSC_STDCALL ',data)[0]
	  data = re.subn('\nPetscErrorCode ','\nvoid PETSC_STDCALL ',data)[0]
	  data = re.subn('Petsc([ToRm]*)Pointer\(int\)','Petsc\\1Pointer(void*)',data)[0]	  
	  data = re.subn('PetscToPointer\(a\) \(a\)','PetscToPointer(a) (*(long *)(a))',data)[0]
	  data = re.subn('PetscFromPointer\(a\) \(int\)\(a\)','PetscFromPointer(a) (long)(a)',data)[0]
  	  data = re.subn('PetscToPointer\( \*\(int\*\)','PetscToPointer(',data)[0]
  	  data = re.subn('MPI_Comm comm','MPI_Comm *comm',data)[0]
  	  data = re.subn('\(MPI_Comm\)PetscToPointer\( \(comm\) \)','(MPI_Comm)MPI_Comm_f2c(*(MPI_Fint*)(comm))',data)[0]
  	  data = re.subn('\(PetscInt\* \)PetscToPointer','',data)[0]
          match = re.compile(r"""\b(PETSC)(_DLL|VEC_DLL|MAT_DLL|DM_DLL|KSP_DLL|SNES_DLL|TS_DLL|FORTRAN_DLL)(EXPORT)""")
          data = match.sub(r'',data)

  	  ff = open(filename,'w')
	  ff.write('#include "petsc.h"\n#include "petscfix.h"\n'+data)
          ff.close()
    
def FixDir(petscdir):
	dir =os.path.join(petscdir,'src','fortran','auto') 
	files = os.listdir(dir)
	names = []
        for f in files:
          if os.path.splitext(f)[1]=='.c':
            FixFile(os.path.join(dir,f))
	    names.append(f)
	ff = open(os.path.join(dir,'makefile.src'),'w')
	ff.write('SOURCEC = '+' '.join(names) + '\n')
	ff.close()


def processDir(arg,dirname,names):
	petscdir = arg[0]
	bfort    = arg[1]
	newls = []
	for l in names:
          if os.path.splitext(l)[1]=='.c' or os.path.splitext(l)[1]=='.h':
	    newls.append(l)
        if newls:
          (status,output) = commands.getstatusoutput('cd '+dirname+';'+bfort+' -dir '+os.path.join(petscdir,'src','fortran','auto')+' \
	  -mnative -ansi -nomsgs -anyname -mapptr -mpi -ferr -ptrprefix Petsc \
	  -ptr64 PETSC_USE_POINTER_CONVERSION  -fcaps PETSC_HAVE_FORTRAN_CAPS  \
          -fuscore PETSC_HAVE_FORTRAN_UNDERSCORE  '+' '.join(newls)) 
  	  if status:
	    raise RuntimeError("Error running bfort "+output)
	if 'SCCS' in names: del names[names.index('SCCS')]
	if 'output' in names: del names[names.index('output')]
	if 'BitKeeper' in names: del names[names.index('BitKeeper')]
	if 'examples' in names: del names[names.index('examples')]
	if 'externalpackages' in names: del names[names.index('externalpackages')]
       	if 'bilinear' in names: del names[names.index('bilinear')]				
	
def main():
        petscdir = os.getcwd()
	dir      = os.path.join(petscdir,'src','fortran','auto') 
	files    = os.listdir(dir)
        for f in files:
          if os.path.splitext(f)[1]=='.c':
            try: os.unlink(os.path.join(dir,f))
            except: pass
	os.path.walk(os.getcwd(),processDir,[petscdir,sys.argv[1]])
        FixDir(os.getcwd())
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
    main()

