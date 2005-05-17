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
  import re
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

def FixDir(dir):
  names = []
  for f in os.listdir(dir):
    if os.path.splitext(f)[1] == '.c':
      FixFile(os.path.join(dir, f))
      names.append(f)
  if not names == []:
    mfile=os.path.abspath(os.path.join(dir,'..','makefile'))
    try:
      fd=open(mfile,'r')
    except:
      print 'Error! missing file:', mfile
      return
    inbuf = fd.read()
    fd.close()
    libbase = ""
    locdir = ""
    for line in inbuf.splitlines():
      if line.find('LIBBASE') >=0:
        libbase = line
      elif line.find('LOCDIR') >=0:
        locdir = line.rstrip() + 'ftn-auto/'

    # now assemble the makefile
    outbuf  =  '\n'
    outbuf +=  "#requirespackage   'PETSC_HAVE_FORTRAN'\n"
    outbuf +=  'ALL: lib\n'
    outbuf +=  'CFLAGS   =\n'
    outbuf +=  'FFLAGS   =\n'
    outbuf +=  'SOURCEC  = ' +' '.join(names)+ '\n'
    outbuf +=  'OBJSC    = ' +' '.join(names).replace('.c','.o')+ '\n'    
    outbuf +=  'SOURCEF  =\n'
    outbuf +=  'SOURCEH  =\n'
    outbuf +=  'DIRS     =\n'
    outbuf +=  libbase + '\n'
    outbuf +=  locdir + '\n'
    outbuf +=  'include ${PETSC_DIR}/bmake/common/base\n'
    outbuf +=  'include ${PETSC_DIR}/bmake/common/test\n'
    
    ff = open(os.path.join(dir, 'makefile'), 'w')
    ff.write(outbuf)
    ff.close()

  # if dir is empty - remove it
  if os.path.exists(dir) and os.path.isdir(dir) and os.listdir(dir) == []:
    os.rmdir(dir)
  return

def PrepFtnDir(dir):
  if os.path.exists(dir) and not os.path.isdir(dir):
    raise RuntimeError('Error - specified path is not a dir: ' + dir)
  elif not os.path.exists(dir):
    os.mkdir(dir)
  else:
    files = os.listdir(dir)
    for file in files:
      os.remove(os.path.join(dir,file))
  return

def processDir(arg,dirname,names):
  import commands
  petscdir = arg[0]
  bfort    = arg[1]
  newls = []
  for l in names:
    if os.path.splitext(l)[1] =='.c' or os.path.splitext(l)[1] == '.h':
      newls.append(l)
  if newls:
    outdir = os.path.join(dirname,'ftn-auto')
    PrepFtnDir(outdir)
    options = ['-dir '+outdir, '-mnative', '-ansi', '-nomsgs',
               '-anyname', '-mapptr', '-mpi', '-ferr', '-ptrprefix Petsc', '-ptr64 PETSC_USE_POINTER_CONVERSION',
               '-fcaps PETSC_HAVE_FORTRAN_CAPS', '-fuscore PETSC_HAVE_FORTRAN_UNDERSCORE']
    (status,output) = commands.getstatusoutput('cd '+dirname+';'+bfort+' '+' '.join(options+newls))
    if status:
      raise RuntimeError('Error running bfort '+output)
    FixDir(outdir)
  for name in ['SCCS', 'output', 'BitKeeper', 'examples', 'externalpackages', 'bilinear', 'ftn-auto','fortran']:
    if name in names:
      names.remove(name)
  return

def main(bfort):
  petscdir = os.getcwd()
  tmpdir = os.path
  os.path.walk(petscdir, processDir, [petscdir, bfort])
  return
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
  import sys
  main(sys.argv[1])
