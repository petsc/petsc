#!/usr/bin/env python
#!/bin/env python
#
#    Generates fortran stubs for PETSc using Sowings bfort program
#
import os
#
#  Opens all generated files and fixes them; also generates list in makefile.src
#
def FixFile(filename):
qimport re
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

  ff = open(filename, 'w')
  ff.write('#include "petsc.h"\n#include "petscfix.h"\n'+data)
  ff.close()
  return

def FixDir(petscdir):
  fdir  = os.path.join(petscdir, 'src', 'fortran', 'auto') 
  names = []
  for f in os.listdir(fdir):
    if os.path.splitext(f)[1] == '.c':
      FixFile(os.path.join(fdir, f))
      names.append(f)
  ff = open(os.path.join(fdir, 'makefile.src'), 'w')
  ff.write('SOURCEC = '+' '.join(names)+'\n')
  ff.close()
  return

def processDir(arg,dirname,names):
  import commands
  petscdir = arg[0]
  bfort    = arg[1]
  newls = []
  for l in names:
    if os.path.splitext(l)[1] = ='.c' or os.path.splitext(l)[1] == '.h':
      newls.append(l)
  if newls:
    options = ['-dir '+os.path.join(petscdir, 'src', 'fortran', 'auto'), '-mnative', '-ansi', '-nomsgs',
               '-anyname', '-mapptr', '-mpi', '-ferr', '-ptrprefix Petsc', '-ptr64 PETSC_USE_POINTER_CONVERSION',
               '-fcaps PETSC_HAVE_FORTRAN_CAPS', '-fuscore PETSC_HAVE_FORTRAN_UNDERSCORE']
    (status,output) = commands.getstatusoutput('cd '+dirname+';'+bfort+' '+' '.join(options+newls))
    if status:
      raise RuntimeError('Error running bfort '+output)
  for name in ['SCCS', 'output', 'BitKeeper', 'examples', 'externalpackages', 'bilinear']:
    if name in name:
      names.remove(name)
  return

def main(bfort):
  petscdir = os.getcwd()
  fdir     = os.path.join(petscdir, 'src', 'fortran', 'auto') 
  files    = os.listdir(fdir)
  for f in files:
    if os.path.splitext(f)[1] == '.c':
      try:
        os.unlink(os.path.join(fdir, f))
      except:
        pass
  os.path.walk(os.getcwd(), processDir, [petscdir, bfort])
  FixDir(os.getcwd())
  return
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
  import sys
  main(sys.argv[1])
