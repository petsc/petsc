#!/usr/bin/env python
#
#   Generates a subdirectory containing all the .c files needed for a build by Apple's Xcode GUI
#
#   Before using removed /usr/include/mpi.h and /Developer/SDKs/MacOSX10.5.sdk/usr/include/mpi.h or
#      Xcode will use those instead of the MPIuni one we point to
#
#   Run ./configure with the options --with-valgrind=0 [--with-mpi=0 --with-x=0 --with-cc="gcc -m32" --download-c-blas-lapack ](when building for iPhone)
#
#   Remove mention of xmm*.h in $PETSC_ARCH/include/petscconf.h and change PETSc_Prefetch() to do nothing (when building for iPhone)
#
#   After running xcodebuilder.py
#      In Project->Add to Project put in the directory $PETSC_DIR/PETSC_ARCH/xcode-links
#      In Project->Edit Project Settings->Search Paths->Header Search Paths add
#         $PETSC_DIR/include $PETSC_DIR/include/mpiuni $PETSC_DIR/$PETSC_ARCH/include  replacing the variables with their values, for example
#         /Users/barrysmith/Src/petsc-dev/include /Users/barrysmith/Src/petsc-dev/include/mpiuni /Users/barrysmith/Src/petsc-dev/arch-uni/include
#      Press control mouse on Frameworks bullet->Add Existing Frameworks then select libblas and liblapack (when building for Mac)
#
#  Notes - if you skip the --with-mpi=0 and let it use the NATIVE Apple MPI that may work, I have not tried it (When building for Mac)
#        - have not tried anything with Fortran or C++
#        - if you link against the X11 libraries you can probably skip the --with-x=0 (When building for Mac)
#
import os, sys

sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config'))
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))

import script

class PETScMaker(script.Script):
 def __init__(self):
   import RDict
   import os

   argDB = RDict.RDict(None, None, 0, 0, readonly = True)
   argDB.saveFilename = os.path.join(os.environ['PETSC_DIR'], os.environ['PETSC_ARCH'], 'conf', 'RDict.db')
   argDB.load()
   script.Script.__init__(self, argDB = argDB)
   self.log = sys.stdout
   return

 def setupModules(self):
   self.mpi           = self.framework.require('config.packages.MPI',         None)
   self.base          = self.framework.require('config.base',                 None)
   self.setCompilers  = self.framework.require('config.setCompilers',         None)   
   self.arch          = self.framework.require('PETSc.utilities.arch',        None)
   self.petscdir      = self.framework.require('PETSc.utilities.petscdir',    None)
   self.languages     = self.framework.require('PETSc.utilities.languages',   None)
   self.debugging     = self.framework.require('PETSc.utilities.debugging',   None)
   self.make          = self.framework.require('PETSc.utilities.Make',        None)
   self.CHUD          = self.framework.require('PETSc.utilities.CHUD',        None)
   self.compilers     = self.framework.require('config.compilers',            None)
   self.types         = self.framework.require('config.types',                None)
   self.headers       = self.framework.require('config.headers',              None)
   self.functions     = self.framework.require('config.functions',            None)
   self.libraries     = self.framework.require('config.libraries',            None)
   self.scalarType    = self.framework.require('PETSc.utilities.scalarTypes', None)
   self.memAlign      = self.framework.require('PETSc.utilities.memAlign',    None)
   self.libraryOptions= self.framework.require('PETSc.utilities.libraryOptions', None)      
   return

 def setupHelp(self, help):
   import nargs

   help = script.Script.setupHelp(self, help)
   help.addArgument('RepManager', '-rootDir', nargs.ArgDir(None, os.environ['PETSC_DIR'], 'The root directory for this build', isTemporary = 1))
   help.addArgument('RepManager', '-dryRun',  nargs.ArgBool(None, False, 'Only output what would be run', isTemporary = 1))
   help.addArgument('RepManager', '-verbose', nargs.ArgInt(None, 0, 'The verbosity level', min = 0, isTemporary = 1))
   return help

 def setup(self):
   script.Script.setup(self)
   self.framework = self.loadConfigure()
   self.setupModules()
   return

 @property
 def verbose(self):
   '''The verbosity level'''
   return self.argDB['verbose']

 @property
 def dryRun(self):
   '''Flag for only output of what would be run'''
   return self.argDB['dryRun']

 def getPackageInfo(self):
   packageIncludes = []
   packageLibs     = []
   for p in self.framework.packages:
     # Could put on compile line, self.addDefine('HAVE_'+i.PACKAGE, 1)
     if hasattr(p, 'lib'):
       if not isinstance(p.lib, list):
         packageLibs.append(p.lib)
       else:
         packageLibs.extend(p.lib)
     if hasattr(p, 'include'):
       if not isinstance(p.include, list):
         packageIncludes.append(p.include)
       else:
         packageIncludes.extend(p.include)
   packageLibs     = self.libraries.toStringNoDupes(packageLibs+self.libraries.math)
   packageIncludes = self.headers.toStringNoDupes(packageIncludes)
   return packageIncludes, packageLibs

 def buildDir(self, dirname):
   ''' This is run in a PETSc source directory'''
   if self.verbose: print 'Entering '+dirname
   os.chdir(dirname)
   l = len(os.environ['PETSC_DIR'])
   basedir = os.path.join(os.environ['PETSC_DIR'],os.environ['PETSC_ARCH'],'xcode-links')
   #newdirname = os.path.join(basedir,dirname[l+1:])
   #os.mkdir(newdirname)


   # Get list of source files in the directory 
   cnames = []
   onames = []
   fnames = []
   hnames = []
   for f in os.listdir(dirname):
     ext = os.path.splitext(f)[1]
     if ext == '.c':
       cnames.append(f)
       onames.append(f.replace('.c', '.o'))
     if ext == '.h':
       hnames.append(f)
   if cnames:
     if self.verbose: print 'Linking C files',cnames
     for i in cnames:
       j = i[l+1:]
       print os.path.join(dirname,i)
       print os.path.join(basedir,i)
       if not os.path.islink(os.path.join(basedir,i)):
         os.symlink(os.path.join(dirname,i),os.path.join(basedir,i))
   # do not need to link these because xcode project points to original source code directory
   #if hnames:
   #  if self.verbose: print 'Linking h files',hnames
   #  for i in hnames:
   #    if not os.path.islink(os.path.join(basedir,i)):
   #      os.symlink(os.path.join(dirname,i),os.path.join(basedir,i))
   return

 def checkDir(self, dirname):
   '''Checks whether we should recurse into this directory
   - Excludes projects directory
   - Excludes examples directory
   - Excludes contrib directory
   - Excludes tutorials directory
   - Excludes benchmarks directory
   - Checks whether fortran bindings are necessary
   - Checks makefile to see if compiler is allowed to visit this directory for this configuration'''
   base = os.path.basename(dirname)

   if base == 'examples': return False
   if base == 'projects': return False
   if not hasattr(self.compilers, 'FC'):
     if base.startswith('ftn-') or base.startswith('f90-'): return False
   if base == 'contrib':  return False
   if base == 'tutorials':  return False
   if base == 'benchmarks':  return False
   if base == 'xcode':  return False
   if base.startswith('arch-'):  return False     

   import re
   reg   = re.compile(' [ ]*')
   fname = os.path.join(dirname, 'makefile')
   if not os.path.isfile(fname):
     if os.path.isfile(os.path.join(dirname, 'Makefile')): print 'ERROR: Change Makefile to makefile in',dirname
     return False
   fd = open(fname)
   text = fd.readline()
   while text:
     if text.startswith('#requires'):
       text = text[9:-1].strip()
       text = reg.sub(' ',text)
       rtype = text.split(' ')[0]
       rvalue = text.split(' ')[1]
       if rtype == 'scalar' and not self.scalarType.scalartype == rvalue:
         if self.verbose: print 'Rejecting',dirname,'because scalar type '+self.scalarType.scalartype+' is not '+rvalue
         return 0
       if rtype == 'language':
         if rvalue == 'CXXONLY' and self.languages.clanguage == 'C':
           if self.verbose: print 'Rejecting',dirname,'because language is '+self.languages.clanguage+' is not C++'
           return 0
       if rtype == 'precision' and not rvalue == self.scalarType.precision:
         if self.verbose: print 'Rejecting',dirname,'because precision '+self.scalarType.precision+' is not '+rvalue
         return 0
       # handles both missing packages and other random stuff that is treated as a package, that should be changed
       if rtype == 'package':
         if rvalue == "'"+'PETSC_HAVE_FORTRAN'+"'" or rvalue == "'"+'PETSC_USING_F90'+"'":
           if not hasattr(self.compilers, 'FC'):
             if self.verbose: print 'Rejecting',dirname,'because fortran is not being used'
             return 0
         elif rvalue == "'"+'PETSC_USE_LOG'+"'":
           if not self.libraryOptions.useLog:
             if self.verbose: print 'Rejecting',dirname,'because logging is turned off'
             return 0
         elif rvalue == "'"+'PETSC_USE_FORTRAN_KERNELS'+"'":
           if not self.libraryOptions.useFortranKernels:
             if self.verbose: print 'Rejecting',dirname,'because fortran kernels are turned off'
             return 0
         else:    
           found = 0
           if self.mpi.usingMPIUni:
             pname = 'PETSC_HAVE_MPIUNI'
             pname = "'"+pname+"'"
             if pname == rvalue: found = 1
           for i in self.framework.packages:
             pname = 'PETSC_HAVE_'+i.PACKAGE
             pname = "'"+pname+"'"
             if pname == rvalue: found = 1
           for i in self.base.defines:
             pname = 'PETSC_'+i
             pname = "'"+pname+"'"
             if pname == rvalue: found = 1
           for i in self.functions.defines:
             pname = 'PETSC_'+i
             pname = "'"+pname+"'"
             if pname == rvalue: found = 1
           if not found:
             if self.verbose: print 'Rejecting',dirname,'because package '+rvalue+' is not installed or function does not exist'
             return 0
         
     text = fd.readline()
   fd.close()
   return True

 def buildAll(self, rootDir = None):
   import shutil
   self.setup()
   if rootDir is None:
     rootDir = self.argDB['rootDir']
   if not self.checkDir(rootDir):
     print 'Nothing to be done'
   if rootDir == os.environ['PETSC_DIR']:
     basedir = os.path.join(self.petscdir.dir, self.arch.arch, 'xcode-links')
     if os.path.isdir(basedir):
       if self.verbose: print 'Removing '+basedir
       shutil.rmtree(basedir)
   os.mkdir(basedir)       
   for root, dirs, files in os.walk(rootDir):
     self.buildDir(root)
     for badDir in [d for d in dirs if not self.checkDir(os.path.join(root, d))]:
       dirs.remove(badDir)
   #self.buildDir(os.path.join(os.environ['PETSC_DIR'],'externalpackages','f2cblaslapack-3.1.1','blas'))
   #self.buildDir(os.path.join(os.environ['PETSC_DIR'],'externalpackages','f2cblaslapack-3.1.1','lapack'))
   # manually link f2c include file
   #if not os.path.islink(os.path.join(self.petscdir.dir, self.arch.arch, 'include','f2c.h')):   
   #os.symlink(os.path.join(self.petscdir.dir, 'externalpackages', 'f2cblaslapack-3.1.1','blas','f2c.h'),os.path.join(self.petscdir.dir, self.arch.arch, 'include','f2c.h'))   
   # do not need to link these because xcode project points to original source code directory
   #os.symlink(os.path.join(self.petscdir.dir, self.arch.arch, 'petscconf.h'),os.path.join(self.petscdir.dir, self.arch.arch, 'xcode-links','include','petscconf.h'))
   #os.symlink(os.path.join(self.petscdir.dir, self.arch.arch, 'petscfix.h'),os.path.join(self.petscdir.dir, self.arch.arch, 'xcode-links','include','petscfix.h'))     
   return

def noCheckCommand(command, status, output, error):
  ''' Do no check result'''
  return 
  noCheckCommand = staticmethod(noCheckCommand)
  
if __name__ == '__main__':
  PETScMaker().buildAll()
