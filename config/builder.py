#!/usr/bin/env python
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
#    self.mpi           = self.framework.require('config.packages.MPI',      None)
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
   help.addArgument('RepManager', '-verbose', nargs.ArgInt(None, 1, 'The verbosity level', min = 0, isTemporary = 1))
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

 def compileC(self, source):
   '''PETSC_INCLUDE         = -I${PETSC_DIR}/${PETSC_ARCH}/include -I${PETSC_DIR}/include
                               ${PACKAGES_INCLUDES} ${TAU_DEFS} ${TAU_INCLUDE}
      PETSC_CC_INCLUDES     = ${PETSC_INCLUDE}
      PETSC_CCPPFLAGS	    = ${PETSC_CC_INCLUDES} ${PETSCFLAGS} ${CPP_FLAGS} ${CPPFLAGS}  -D__SDIR__='"${LOCDIR}"'
      CCPPFLAGS	            = ${PETSC_CCPPFLAGS}
      PETSC_COMPILE         = ${PCC} -c ${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS}  ${SOURCEC} ${SSOURCE}
      PETSC_COMPILE_SINGLE  = ${PCC} -o $*.o -c ${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS}'''
   # PETSCFLAGS, CFLAGS and CPPFLAGS are taken from user input (or empty)
   includes = ['-I'+inc for inc in [os.path.join(self.petscdir.dir, self.arch.arch, 'include'), os.path.join(self.petscdir.dir, 'include')]]
   self.setCompilers.pushLanguage(self.languages.clanguage)
   compiler = self.setCompilers.getCompiler()
   flags = []
   flags.append(self.setCompilers.getCompilerFlags())             # PCC_FLAGS
   flags.extend([self.setCompilers.CPPFLAGS, self.CHUD.CPPFLAGS]) # CPP_FLAGS
   flags.append('-D__INSDIR__='+os.getcwd().replace(self.petscdir.dir, ''))
   packageIncludes, packageLibs = self.getPackageInfo()
   cmd = ' '.join([compiler]+['-c']+includes+[packageIncludes]+flags+source)
   if self.dryRun or self.verbose: print cmd
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd,checkCommand = noCheckCommand,log=self.log)
     if status:
       print "ERROR IN COMPILE ******************************"
       print output+error
   self.setCompilers.popLanguage()
   return

 def compileF(self, source):
   '''PETSC_INCLUDE	        = -I${PETSC_DIR}/${PETSC_ARCH}/include -I${PETSC_DIR}/include
                              ${PACKAGES_INCLUDES} ${TAU_DEFS} ${TAU_INCLUDE}
      PETSC_CC_INCLUDES     = ${PETSC_INCLUDE}
      PETSC_CCPPFLAGS	    = ${PETSC_CC_INCLUDES} ${PETSCFLAGS} ${CPP_FLAGS} ${CPPFLAGS}  -D__SDIR__='"${LOCDIR}"'
      CCPPFLAGS	            = ${PETSC_CCPPFLAGS}
      PETSC_COMPILE         = ${PCC} -c ${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS}  ${SOURCEC} ${SSOURCE}
      PETSC_COMPILE_SINGLE  = ${PCC} -o $*.o -c ${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS}'''
   # PETSCFLAGS, CFLAGS and CPPFLAGS are taken from user input (or empty)
   flags           = []

   includes = ['-I'+inc for inc in [os.path.join(self.petscdir.dir, self.arch.arch, 'include'), os.path.join(self.petscdir.dir, 'include')]]
   self.setCompilers.pushLanguage('FC')
   compiler      = self.setCompilers.getCompiler()
   flags.append(self.setCompilers.getCompilerFlags())             # PCC_FLAGS
   flags.extend([self.setCompilers.CPPFLAGS, self.CHUD.CPPFLAGS]) # CPP_FLAGS
   cmd = ' '.join([compiler]+['-c']+includes+flags+source)
   if self.dryRun or self.verbose: print cmd
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd,checkCommand = noCheckCommand,log=self.log)
     if status:
       print "ERROR IN COMPILE ******************************"
       print output+error
   self.setCompilers.popLanguage()
   return

 def archive(self, library, objects):
   '''${AR} ${AR_FLAGS} ${LIBNAME} $*.o'''
   lib = os.path.splitext(library)[0]+'.'+self.setCompilers.AR_LIB_SUFFIX
   cmd = ' '.join([self.setCompilers.AR, self.setCompilers.FAST_AR_FLAGS, lib]+objects)
   if self.dryRun or self.verbose: print cmd
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd,checkCommand = noCheckCommand,log=self.log)
     if status:
       print "ERROR IN ARCHIVE ******************************"
       print output+error
   return

 def ranlib(self, library):
   '''${ranlib} ${LIBNAME} '''
   library = os.path.join(self.petscdir.dir, self.arch.arch, 'lib', library)   
   lib = os.path.splitext(library)[0]+'.'+self.setCompilers.AR_LIB_SUFFIX
   cmd = ' '.join([self.setCompilers.RANLIB, lib])
   if self.dryRun or self.verbose: print cmd
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd,checkCommand = noCheckCommand,log=self.log)
     if status:
       print "ERROR IN RANLIB ******************************"
       print output+error
   return
 
 def buildDir(self, libname, dirname, fnames):
   ''' This is run in a PETSc source directory'''
   if self.verbose: print 'Entering '+dirname
   os.chdir(dirname)

   # Get list of source files in the directory 
   cnames = []
   onames = []
   fnames = []
   for f in os.listdir(dirname):
     ext = os.path.splitext(f)[1]
     if ext == '.c':
       cnames.append(f)
       onames.append(f.replace('.c', '.o'))
     if hasattr(self.compilers, 'FC'):
       if ext == '.F':
         fnames.append(f)
         onames.append(f.replace('.F', '.o'))
       if self.compilers.fortranIsF90:
         if ext == '.F90':
           fnames.append(f)
           onames.append(f.replace('.F90', '.o'))
   if cnames:
     if self.verbose: print 'Compiling C files',cnames
     self.compileC(cnames)
   if fnames:
     if self.verbose: print 'Compiling Fortran files',fnames
     self.compileF(fnames)
   if onames:
     if self.verbose: print 'Archiving files',onames,'into',libname
     self.archive(os.path.join(self.petscdir.dir, self.arch.arch, 'lib', libname), onames)
   return

 def checkDir(self, dirname):
   '''Checks whether we should recurse into this directory
   - Excludes examples directory
   - Excludes contrib directory
   - Excludes tutorials directory
   - Excludes benchmarks directory
   - Checks whether fortran bindings are necessary
   - Checks makefile to see if compiler is allowed to visit this directory for this configuration'''
   base = os.path.basename(dirname)

   if base == 'examples': return False
   if not hasattr(self.compilers, 'FC'):
     if base.startswith('ftn-') or base.startswith('f90-'): return False
   if base == 'contrib':  return False
   if base == 'tutorials':  return False
   if base == 'benchmarks':  return False     

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
   self.setup()
   if rootDir is None:
     rootDir = self.argDB['rootDir']
   if not self.checkDir(rootDir):
     print 'Nothing to be done'
   if rootDir == os.environ['PETSC_DIR']:
     library = os.path.join(self.petscdir.dir, self.arch.arch, 'lib', 'libpetsc')   
     lib = os.path.splitext(library)[0]+'.'+self.setCompilers.AR_LIB_SUFFIX
     if os.path.isfile(lib):
       if self.verbose: print 'Removing '+lib
       os.unlink(lib)
   for root, dirs, files in os.walk(rootDir):
     print 'Processing',root
     self.buildDir('libpetsc', root, files)
     for badDir in [d for d in dirs if not self.checkDir(os.path.join(root, d))]:
       dirs.remove(badDir)
   self.ranlib('libpetsc')
     
   return

def noCheckCommand(command, status, output, error):
  ''' Do no check result'''
  return 
  noCheckCommand = staticmethod(noCheckCommand)
  
if __name__ == '__main__':
  PETScMaker().buildAll()
