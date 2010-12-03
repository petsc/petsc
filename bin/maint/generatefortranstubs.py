#!/usr/bin/env python
#!/bin/env python
#
#    Generates fortran stubs for PETSc using Sowings bfort program
#
import os
#
#  Opens all generated files and fixes them
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
  data = re.subn('\(MPI_Comm\)PetscToPointer\( \(comm\) \)','MPI_Comm_f2c(*(MPI_Fint*)(comm))',data)[0]
  data = re.subn('\(PetscInt\* \)PetscToPointer','',data)[0]
  match = re.compile(r"""\b(PETSC)(_DLL|VEC_DLL|MAT_DLL|DM_DLL|KSP_DLL|SNES_DLL|TS_DLL|FORTRAN_DLL)(EXPORT)""")
  data = match.sub(r'',data)

  ff = open(filename, 'w')
  ff.write('#include "petscsys.h"\n#include "petscfix.h"\n'+data)
  ff.close()

  def FindSource(filename):
    import os.path
    gendir, fname = os.path.split(filename)
    base, ext = os.path.splitext(fname)
    sdir, ftn_auto = os.path.split(gendir)
    if ftn_auto != 'ftn-auto': return None # Something is wrong, skip
    sfname = os.path.join(sdir, base[:-1] + ext)
    return sfname
  sourcefile = FindSource(filename)
  if sourcefile and os.path.isfile(sourcefile):
    import shutil
    shutil.copystat(sourcefile, filename)
  return

def FixDir(petscdir,dir):
  mansec = 'unknown'
  cnames = []
  hnames = []
  parentdir =os.path.abspath(os.path.join(dir,'..'))
  for f in os.listdir(dir):
    ext = os.path.splitext(f)[1]
    if ext == '.c':
      FixFile(os.path.join(dir, f))
      cnames.append(f)
    elif ext == '.h90':
      hnames.append(f)
  if (cnames != [] or hnames != []):
    mfile=os.path.abspath(os.path.join(parentdir,'makefile'))
    try:
      fd=open(mfile,'r')
    except:
      print 'Error! missing file:', mfile
      return
    inbuf = fd.read()
    fd.close()
    cppflags = ""
    libbase = ""
    locdir = ""
    for line in inbuf.splitlines():
      if line.find('CPPFLAGS') >=0:
        cppflags = line
      if line.find('LIBBASE') >=0:
        libbase = line
      elif line.find('LOCDIR') >=0:
        locdir = line.rstrip() + 'ftn-auto/'
      elif line.find('MANSEC') >=0:
        mansec = line.split('=')[1].lower().strip()

    # now assemble the makefile
    outbuf  =  '\n'
    outbuf +=  "#requirespackage   'PETSC_HAVE_FORTRAN'\n"
    outbuf +=  'ALL: lib\n'
    outbuf +=   cppflags + '\n'
    outbuf +=  'CFLAGS   =\n'
    outbuf +=  'FFLAGS   =\n'
    outbuf +=  'SOURCEC  = ' +' '.join(cnames)+ '\n'
    outbuf +=  'OBJSC    = ' +' '.join(cnames).replace('.c','.o')+ '\n'    
    outbuf +=  'SOURCEF  =\n'
    outbuf +=  'SOURCEH  = ' +' '.join(hnames)+ '\n'
    outbuf +=  'DIRS     =\n'
    outbuf +=  libbase + '\n'
    outbuf +=  locdir + '\n'
    outbuf +=  'include ${PETSC_DIR}/conf/variables\n'
    outbuf +=  'include ${PETSC_DIR}/conf/rules\n'
    outbuf +=  'include ${PETSC_DIR}/conf/test\n'
    
    ff = open(os.path.join(dir, 'makefile'), 'w')
    ff.write(outbuf)
    ff.close()

  # if dir is empty - remove it
  if os.path.exists(dir) and os.path.isdir(dir) and os.listdir(dir) == []:
    os.rmdir(dir)

  # Now process f90module.f90 file - and update include/finclude/ftn-auto
  modfile = os.path.join(parentdir,'f90module.f90')
  if os.path.exists(modfile):
    fd = open(modfile)
    txt = fd.read()
    fd.close()

    if txt and mansec == 'unknown':
      print 'makefile has missing MANSEC',parentdir
    elif txt:
      ftype = 'w'
      f90inc = os.path.join(petscdir,'include','finclude','ftn-auto','petsc'+mansec+'.h90')
      if os.path.exists(f90inc): ftype = 'a'
      fd = open(f90inc,ftype)
      fd.write(txt)
      fd.close()
    os.remove(modfile)
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
  outdir = os.path.join(dirname,'ftn-auto')

  # skip include/finclude/ftn-auto - as this is processed separately
  if os.path.realpath(os.path.join(petscdir,'include','finclude','ftn-auto')) == os.path.realpath(outdir): return

  for l in names:
    if os.path.splitext(l)[1] =='.c' or os.path.splitext(l)[1] == '.h':
      newls.append(l)
  if newls:
    PrepFtnDir(outdir)
    options = ['-dir '+outdir, '-mnative', '-ansi', '-nomsgs', '-noprofile', '-anyname', '-mapptr',
               '-mpi', '-mpi2', '-ferr', '-ptrprefix Petsc', '-ptr64 PETSC_USE_POINTER_CONVERSION',
               '-fcaps PETSC_HAVE_FORTRAN_CAPS', '-fuscore PETSC_HAVE_FORTRAN_UNDERSCORE',
               '-f90mod_skip_header','-f90modfile','f90module.f90']
    cmd = 'cd '+dirname+';'+bfort+' '+' '.join(options+newls)
    (status,output) = commands.getstatusoutput(cmd)
    if status:
      raise RuntimeError('Error running bfort\n'+cmd+'\n'+output)
    FixDir(petscdir,outdir)

  # remove from list of subdirectories all directories without source code
  rmnames=[]
  for name in names:
    if name in ['.hg','SCCS', 'output', 'BitKeeper', 'examples', 'externalpackages', 'bilinear', 'ftn-auto','fortran','bin','maint','ftn-custom','config','f90-custom']:
      rmnames.append(name)
    # skip for ./configure generated $PETSC_ARCH directories
    if os.path.isdir(os.path.join(name,'conf')):
      rmnames.append(name)
    # skip include/finclude directory 
    if name == 'finclude':
      rmnames.append(name)     
  for rmname in rmnames:
    names.remove(rmname)
  return

def main(bfort):
  petscdir = os.getcwd()
  tmpdir = os.path
  ftnautoinc = os.path.join(petscdir,'include','finclude','ftn-auto')
  PrepFtnDir(ftnautoinc)
  os.path.walk(petscdir, processDir, [petscdir, bfort])
  FixDir(petscdir,ftnautoinc)
  return
#
# The classes in this file can also be used in other python-programs by using 'import'
#
if __name__ ==  '__main__': 
  import sys
  if len(sys.argv) < 2:
    sys.exit('Must give the BFORT program as the first argument')
  main(sys.argv[1])
