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
   self.logName = 'make.log'
   #self.log = sys.stdout
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
   #help.addArgument('RepManager', '-rootDir', nargs.ArgDir(None, os.environ['PETSC_DIR'], 'The root directory for this build', isTemporary = 1))
   help.addArgument('RepManager', '-rootDir', nargs.ArgDir(None, os.getcwd(), 'The root directory for this build', isTemporary = 1))
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

 def compileC(self, source):
   '''PETSC_INCLUDE         = -I${PETSC_DIR}/${PETSC_ARCH}/include -I${PETSC_DIR}/include
                               ${PACKAGES_INCLUDES} ${TAU_DEFS} ${TAU_INCLUDE}
      PETSC_CC_INCLUDES     = ${PETSC_INCLUDE}
      PETSC_CCPPFLAGS	    = ${PETSC_CC_INCLUDES} ${PETSCFLAGS} ${CPP_FLAGS} ${CPPFLAGS}  -D__SDIR__='"${LOCDIR}"'
      CCPPFLAGS	            = ${PETSC_CCPPFLAGS}
      PETSC_COMPILE         = ${PCC} -c ${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS}  ${SOURCEC} ${SSOURCE}
      PETSC_COMPILE_SINGLE  = ${PCC} -o $*.o -c ${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS}'''
   import shutil
   # PETSCFLAGS, CFLAGS and CPPFLAGS are taken from user input (or empty)
   includes = ['-I'+inc for inc in [os.path.join(self.petscdir.dir, self.arch.arch, 'include'), os.path.join(self.petscdir.dir, 'include')]]
   self.setCompilers.pushLanguage(self.languages.clanguage)
   compiler = self.setCompilers.getCompiler()
   flags = []
   flags.append(self.setCompilers.getCompilerFlags())             # PCC_FLAGS
   flags.extend([self.setCompilers.CPPFLAGS, self.CHUD.CPPFLAGS]) # CPP_FLAGS
   flags.append('-D__INSDIR__='+os.getcwd().replace(self.petscdir.dir, ''))
   packageIncludes, packageLibs = self.getPackageInfo()
   sources = source
   cmd = ' '.join([compiler]+['-c']+includes+[packageIncludes]+flags+sources)
   if self.dryRun or self.verbose:
     section = 'screen'
   else:
     section = None
   self.logWrite(cmd+'\n', debugSection = section, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd,checkCommand = noCheckCommand,log=self.log)
     if status:
       self.logPrint("ERROR IN COMPILE ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
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
   if self.dryRun or self.verbose:
     section = 'screen'
   else:
     section = None
   self.logWrite(cmd+'\n', debugSection = section, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd,checkCommand = noCheckCommand,log=self.log)
     if status:
       self.logPrint("ERROR IN COMPILE ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   self.setCompilers.popLanguage()
   return

 def archive(self, library, objects):
   '''${AR} ${AR_FLAGS} ${LIBNAME} $*.o'''
   lib = os.path.splitext(library)[0]+'.'+self.setCompilers.AR_LIB_SUFFIX
   if self.argDB['rootDir'] == os.environ['PETSC_DIR']:
     cmd = ' '.join([self.setCompilers.AR, self.setCompilers.FAST_AR_FLAGS, lib]+objects)
   else:
     cmd = ' '.join([self.setCompilers.AR, self.setCompilers.AR_FLAGS, lib]+objects)
   if self.dryRun or self.verbose:
     section = 'screen'
   else:
     section = None
   self.logWrite(cmd+'\n', debugSection = section, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd,checkCommand = noCheckCommand,log=self.log)
     if status:
       self.logPrint("ERROR IN ARCHIVE ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   return

 def ranlib(self, library):
   '''${ranlib} ${LIBNAME} '''
   library = os.path.join(self.petscdir.dir, self.arch.arch, 'lib', library)   
   lib = os.path.splitext(library)[0]+'.'+self.setCompilers.AR_LIB_SUFFIX
   cmd = ' '.join([self.setCompilers.RANLIB, lib])
   if self.dryRun or self.verbose:
     section = 'screen'
   else:
     section = None
   self.logWrite(cmd+'\n', debugSection = section, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd,checkCommand = noCheckCommand,log=self.log)
     if status:
       self.logPrint("ERROR IN RANLIB ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   return
 
 def buildDir(self, libname, dirname, fnames):
   ''' This is run in a PETSc source directory'''
   self.logWrite('Entering '+dirname+'\n', debugSection = 'screen', forceScroll = True)
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
     self.logPrint('Compiling C files '+str(cnames))
     self.compileC(cnames)
   if fnames:
     self.logPrint('Compiling Fortran files '+str(fnames))
     self.compileF(fnames)
   if onames:
     self.logPrint('Archiving files '+str(onames)+' into '+libname)
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
     if os.path.isfile(os.path.join(dirname, 'Makefile')): self.logPrint('ERROR: Change Makefile to makefile in '+dirname, debugSection = 'screen')
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
         self.logPrint('Rejecting '+dirname+' because scalar type '+self.scalarType.scalartype+' is not '+rvalue)
         return 0
       if rtype == 'language':
         if rvalue == 'CXXONLY' and self.languages.clanguage == 'C':
           self.logPrint('Rejecting '+dirname+' because language is '+self.languages.clanguage+' is not C++')
           return 0
       if rtype == 'precision' and not rvalue == self.scalarType.precision:
         self.logPrint('Rejecting '+dirname+' because precision '+self.scalarType.precision+' is not '+rvalue)
         return 0
       # handles both missing packages and other random stuff that is treated as a package, that should be changed
       if rtype == 'package':
         if rvalue == "'"+'PETSC_HAVE_FORTRAN'+"'" or rvalue == "'"+'PETSC_USING_F90'+"'":
           if not hasattr(self.compilers, 'FC'):
             self.logPrint('Rejecting '+dirname+' because fortran is not being used')
             return 0
         elif rvalue == "'"+'PETSC_USE_LOG'+"'":
           if not self.libraryOptions.useLog:
             self.logPrint('Rejecting '+dirname+' because logging is turned off')
             return 0
         elif rvalue == "'"+'PETSC_USE_FORTRAN_KERNELS'+"'":
           if not self.libraryOptions.useFortranKernels:
             self.logPrint('Rejecting '+dirname+' because fortran kernels are turned off')
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
             self.logPrint('Rejecting '+dirname+' because package '+rvalue+' is not installed or function does not exist')
             return 0
         
     text = fd.readline()
   fd.close()
   return True

  def cleanupLog(self, framework):
    '''Move configure.log to PROJECT_ARCH/conf - and update configure.log.bkp in both locations appropriately'''
    arch    = self.arch.arch
    logFile = 'make.log'
    if hasattr(framework, 'logName'): logFile = framework.logName

    if arch:
      import shutil
      import os

      confDir = os.path.join(arch, 'conf')
      if not os.path.isdir(arch):    os.mkdir(arch)
      if not os.path.isdir(confDir): os.mkdir(confDir)

      logFileBkp        = logFile + '.bkp'
      logFileArchive    = os.path.join(confDir, logFile)
      logFileArchiveBkp = logFileArchive + '.bkp'

      # Keep backup in $PROJECT_ARCH/conf location
      if os.path.isfile(logFileArchiveBkp): os.remove(logFileArchiveBkp)
      if os.path.isfile(logFileArchive):    os.rename(logFileArchive, logFileArchiveBkp)
      if os.path.isfile(logFile):
        shutil.copyfile(logFile, logFileArchive)
        os.remove(logFile)
      if os.path.isfile(logFileArchive):    os.symlink(logFileArchive, logFile)
      # If the old bkp is using the same $PROJECT_ARCH/conf, then update bkp link
      if os.path.realpath(logFileBkp) == os.path.realpath(logFileArchive):
        if os.path.isfile(logFileBkp):        os.remove(logFileBkp)
        if os.path.isfile(logFileArchiveBkp): os.symlink(logFileArchiveBkp, logFileBkp)
    return

 def buildAll(self, rootDir = None):
   self.setup()
   if rootDir is None:
     rootDir = self.argDB['rootDir']
   if not self.checkDir(rootDir):
     self.logPrint('Nothing to be done')
   if rootDir == os.environ['PETSC_DIR']:
     library = os.path.join(self.petscdir.dir, self.arch.arch, 'lib', 'libpetsc')   
     lib = os.path.splitext(library)[0]+'.'+self.setCompilers.AR_LIB_SUFFIX
     if os.path.isfile(lib):
       self.logPrint('Removing '+lib)
       os.unlink(lib)
   for root, dirs, files in os.walk(rootDir):
     self.logPrint('Processing '+root)
     self.buildDir('libpetsc', root, files)
     for badDir in [d for d in dirs if not self.checkDir(os.path.join(root, d))]:
       dirs.remove(badDir)
   self.ranlib('libpetsc')
   self.cleanupLog()
   return

def noCheckCommand(command, status, output, error):
  ''' Do no check result'''
  return 
  noCheckCommand = staticmethod(noCheckCommand)
  
if __name__ == '__main__':
  PETScMaker().buildAll()
