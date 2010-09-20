#!/usr/bin/env python

from __future__ import with_statement  # For python-2.5

import os, sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config'))
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))

import script

regressionRequirements = {'src/vec/vec/examples/tests/ex31':  set(['Matlab'])
                          }

regressionParameters = {'src/vec/vec/examples/tests/ex1_2':  {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex3':    {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex4':    {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex5':    {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex9':    {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex10':   {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex11':   {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex12':   {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex13':   {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex14':   {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex16':   {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex17':   {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex17f':  {'numProcs': 3},
                        'src/vec/vec/examples/tests/ex21_2': {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex22':   {'numProcs': 4},
                        'src/vec/vec/examples/tests/ex23':   {'numProcs': 2},
                        'src/vec/vec/examples/tests/ex24':   {'numProcs': 3},
                        'src/vec/vec/examples/tests/ex25':   {'numProcs': 3},
                        'src/vec/vec/examples/tests/ex26':   {'numProcs': 4},
                        'src/vec/vec/examples/tests/ex28':   {'numProcs': 3},
                        'src/vec/vec/examples/tests/ex29':   {'numProcs': 3, 'args': '-n 126'},
                        'src/vec/vec/examples/tests/ex30f':  {'numProcs': 4},
                        'src/vec/vec/examples/tests/ex33':   {'numProcs': 4},
                        'src/vec/vec/examples/tests/ex36':   {'numProcs': 2, 'args': '-set_option_negidx -set_values_negidx -get_values_negidx'}
                        }

def noCheckCommand(command, status, output, error):
  ''' Do no check result'''
  return 

class NullSourceDatabase(object):
  def __init__(self, verbose = 0):
    return

  def setNode(self, vertex, deps):
    return

  def updateNode(self, vertex):
    return

  def rebuild(self, vertex):
    return True

class SourceDatabase(object):
  '''This can be replaced by the favorite software of Jed'''
  def __init__(self, verbose = 0):
    # Vertices are filenames
    #   Arcs indicate a dependence and are decorated with consistency markers
    self.dependencyGraph = {}
    self.verbose         = verbose
    return

  def __str__(self):
    return str(self.dependencyGraph)

  @staticmethod
  def marker(dep):
    import hashlib
    with file(dep) as f:
      mark = hashlib.sha1(f.read()).digest()
    return mark

  def setNode(self, vertex, deps):
    self.dependencyGraph[vertex] = [(dep, SourceDatabase.marker(dep)) for dep in deps]
    return

  def updateNode(self, vertex):
    self.dependencyGraph[vertex] = [(dep, SourceDatabase.marker(dep)) for dep,mark in self.dependencyGraph[vertex]]
    return

  def rebuildArc(self, vertex, dep, mark):
    import hashlib
    with file(dep) as f:
      newMark = hashlib.sha1(f.read()).digest()
    return not mark == newMark

  def rebuild(self, vertex):
    if self.verbose: print 'Checking for rebuild of',vertex
    try:
      for dep,mark in self.dependencyGraph[vertex]:
        if self.rebuildArc(vertex, dep, mark):
          if self.verbose: print '    dep',dep,'is changed'
          return True
    except KeyError:
      return True
    return False

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
   self.fortrancpp    = self.framework.require('PETSc.utilities.fortranCPP', None)
   self.debuggers     = self.framework.require('PETSc.utilities.debuggers', None)
   self.sharedLibraries= self.framework.require('PETSc.utilities.sharedLibraries', None)      
   return

 def setupHelp(self, help):
   import nargs

   help = script.Script.setupHelp(self, help)
   #help.addArgument('PETScMaker', '-rootDir', nargs.ArgDir(None, os.environ['PETSC_DIR'], 'The root directory for this build', isTemporary = 1))
   help.addArgument('PETScMaker', '-rootDir', nargs.ArgDir(None, os.getcwd(), 'The root directory for this build', isTemporary = 1))
   help.addArgument('PETScMaker', '-dryRun',  nargs.ArgBool(None, False, 'Only output what would be run', isTemporary = 1))
   help.addArgument('PETScMaker', '-dependencies',  nargs.ArgBool(None, True, 'Use dependencies to control build', isTemporary = 1))
   help.addArgument('PETScMaker', '-buildLibraries', nargs.ArgBool(None, True, 'Build the PETSc libraries', isTemporary = 1))
   help.addArgument('PETScMaker', '-regressionTests', nargs.ArgBool(None, False, 'Only run regression tests', isTemporary = 1))
   help.addArgument('PETScMaker', '-rebuildDependencies', nargs.ArgBool(None, False, 'Rebuild dependency information', isTemporary = 1))
   help.addArgument('PETScMaker', '-verbose', nargs.ArgInt(None, 0, 'The verbosity level', min = 0, isTemporary = 1))

   help.addArgument('PETScMaker', '-maxSources', nargs.ArgInt(None, -1, 'The maximum number of source files in a directory', min = -1, isTemporary = 1))
   return help

 def setup(self):
   script.Script.setup(self)
   if self.dryRun or self.verbose:
     self.debugSection = 'screen'
   else:
     self.debugSection = None
   self.argDB['rootDir'] = os.path.abspath(self.argDB['rootDir'])
   self.framework = self.loadConfigure()
   self.setupModules()
   if self.argDB['dependencies']:
     confDir = os.path.join(self.petscdir.dir, self.arch.arch, 'conf')
     if not self.argDB['rebuildDependencies'] and os.path.isfile(os.path.join(confDir, 'source.db')):
       import cPickle

       with file(os.path.join(confDir, 'source.db'), 'rb') as f:
         self.sourceDatabase = cPickle.load(f)
       self.sourceDatabase.verbose = self.verbose
     else:
       self.sourceDatabase = SourceDatabase(self.verbose)
   else:
     self.sourceDatabase = NullSourceDatabase(self.verbose)
   return

 def cleanupLog(self, framework, confDir):
   '''Move configure.log to PROJECT_ARCH/conf - and update configure.log.bkp in both locations appropriately'''
   import os

   self.log.flush()
   if hasattr(framework, 'logName'):
     logName         = framework.logName
   else:
     logName         = 'make.log'
   logFile           = os.path.join(self.petscdir.dir, logName)
   logFileBkp        = logFile + '.bkp'
   logFileArchive    = os.path.join(confDir, logName)
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

 def cleanup(self):
   root    = self.petscdir.dir
   arch    = self.arch.arch
   archDir = os.path.join(root, arch)
   confDir = os.path.join(archDir, 'conf')
   if not os.path.isdir(archDir): os.mkdir(archDir)
   if not os.path.isdir(confDir): os.mkdir(confDir)

   self.cleanupLog(self, confDir)
   if self.argDB['dependencies']:
     import cPickle
     with file(os.path.join(confDir, 'source.db'), 'wb') as f:
       cPickle.dump(self.sourceDatabase, f)
   return

 @property
 def verbose(self):
   '''The verbosity level'''
   return self.argDB['verbose']

 @property
 def dryRun(self):
   '''Flag for only output of what would be run'''
   return self.argDB['dryRun']

 def readDependencyFile(self, depFile):
   with file(depFile) as f:
     target,deps = f.read().split(':')
   self.sourceDatabase.setNode(target, deps.replace('\\','').split())
   return

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

 def getObjectName(self, source, objDir = None):
   if objDir is None:
     return os.path.splitext(source)[0]+'.o'
   return os.path.join(objDir, os.path.splitext(os.path.basename(source))[0]+'.o')

 def sortSourceFiles(self, dirname, objDir = None):
   '''Sorts source files by language (returns dictionary with language keys)'''
   cnames    = []
   cxxnames  = []
   cudanames = []
   f77names  = []
   f90names  = []
   for f in os.listdir(dirname):
     ext = os.path.splitext(f)[1]
     if ext == '.c':
       cnames.append(f)
     elif ext in ['.cxx', '.cpp', '.cc']:
       if self.languages.clanguage == 'Cxx':
         cxxnames.append(f)
     elif ext == '.cu':
       cudanames.append(f)
     elif ext == '.F':
       if hasattr(self.compilers, 'FC'):
         f77names.append(f)
     elif ext == '.F90':
       if hasattr(self.compilers, 'FC') and self.compilers.fortranIsF90:
         f90names.append(f)
   source = cnames+cxxnames+cudanames+f77names+f90names
   if self.argDB['maxSources'] >= 0:
     cnames    = cnames[:self.argDB['maxSources']]
     cxxnames  = cxxnames[:self.argDB['maxSources']]
     cudanames = cudanames[:self.argDB['maxSources']]
     f77names  = f77names[:self.argDB['maxSources']]
     f90names  = f90names[:self.argDB['maxSources']]
     source    = source[:self.argDB['maxSources']]
   return {'C': cnames, 'Cxx': cxxnames, 'Cuda': cudanames, 'F77': f77names, 'F90': f90names, 'Fortran': f77names+f90names, 'Objects': [self.getObjectName(s, objDir) for s in source]}

 def compileC(self, source, objDir = None):
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
   # TODO: Move this up to configure
   if self.argDB['dependencies']: flags.append('-MMD')
   sources = [s for s in source if not os.path.isfile(self.getObjectName(s, objDir)) or self.sourceDatabase.rebuild(self.getObjectName(s, objDir))]
   objects = [self.getObjectName(s, objDir) for s in sources]
   packageIncludes, packageLibs = self.getPackageInfo()
   cmd = ' '.join([compiler]+['-c']+includes+[packageIncludes]+flags+sources)
   if len(sources):
     self.logWrite(cmd+'\n', debugSection = self.debugSection, forceScroll = True)
     if not self.dryRun:
       (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
       if status:
         self.logPrint("ERROR IN COMPILE ******************************", debugSection='screen')
         self.logPrint(output+error, debugSection='screen')
       else:
         self.buildDependenciesFiles(sources)
   else:
     self.logPrint('Nothing to build', debugSection = self.debugSection)
   self.setCompilers.popLanguage()
   for o in objects:
     locObj = os.path.basename(o)
     if not os.path.isfile(locObj):
       print 'ERROR: Missing object file',locObj
     else:
       shutil.move(locObj, o)
   return objects

 def compileF(self, source, objDir = None):
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
   objects  = [self.getObjectName(s, objDir) for s in source]
   self.setCompilers.pushLanguage('FC')
   compiler      = self.setCompilers.getCompiler()
   flags.append(self.setCompilers.getCompilerFlags())             # PCC_FLAGS
   flags.extend([self.setCompilers.CPPFLAGS, self.CHUD.CPPFLAGS]) # CPP_FLAGS
   cmd = ' '.join([compiler]+['-c']+includes+flags+source)
   self.logWrite(cmd+'\n', debugSection = self.debugSection, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.logPrint("ERROR IN COMPILE ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   self.setCompilers.popLanguage()
   for o in objects:
     if not os.path.isfile(o):
       print 'ERROR: Missing object file',o
   return objects

 def archive(self, library, objects):
   '''${AR} ${AR_FLAGS} ${LIBNAME} $*.o'''
   lib = os.path.splitext(library)[0]+'.'+self.setCompilers.AR_LIB_SUFFIX
   if self.argDB['rootDir'] == os.environ['PETSC_DIR']:
     cmd = ' '.join([self.setCompilers.AR, self.setCompilers.FAST_AR_FLAGS, lib]+objects)
   else:
     cmd = ' '.join([self.setCompilers.AR, self.setCompilers.AR_FLAGS, lib]+objects)
   self.logWrite(cmd+'\n', debugSection = self.debugSection, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.logPrint("ERROR IN ARCHIVE ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   return [library]

 def ranlib(self, library):
   '''${ranlib} ${LIBNAME} '''
   library = os.path.join(self.petscdir.dir, self.arch.arch, 'lib', library)   
   lib = os.path.splitext(library)[0]+'.'+self.setCompilers.AR_LIB_SUFFIX
   cmd = ' '.join([self.setCompilers.RANLIB, lib])
   self.logWrite(cmd+'\n', debugSection = self.debugSection, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.logPrint("ERROR IN RANLIB ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   return

 def linkShared(self, sharedLib, libDir, tmpDir):
   '''
   CLINKER                  = ${PCC_LINKER} ${PCC_LINKER_FLAGS}
   PETSC_EXTERNAL_LIB_BASIC = ${EXTERNAL_LIB} ${PACKAGES_LIBS} ${PCC_LINKER_LIBS}
   SYS_LIB                  = ???
   '''
   osName = self.arch.hostOsBase
   # PCC_LINKER PCC_LINKER_FLAGS
   linker      = self.setCompilers.getLinker()
   linkerFlags = self.setCompilers.getLinkerFlags()
   # PACKAGES_LIBS PCC_LINKER_LIBS
   packageIncludes, packageLibs = self.getPackageInfo()
   extraLibs = self.libraries.toStringNoDupes(self.compilers.flibs+self.compilers.cxxlibs+self.compilers.LIBS.split(' '))+self.CHUD.LIBS
   sysLib      = ''
   sysLib.replace('-Wl,-rpath', '-L')
   externalLib = packageLibs+' '+extraLibs
   externalLib.replace('-Wl,-rpath', '-L')
   # Move this switch into the sharedLibrary module
   if self.setCompilers.isSolaris() and self.setCompilers.isGNU(self.framework.getCompiler()):
     cmd = self.setCompilers.LD+' -G -h '+os.path.basename(sharedLib)+' *.o -o '+sharedLib+' '+sysLib+' '+externalLib
     oldDir = os.getcwd()
     os.chdir(tmpDir)
     self.executeShellCommand(cmd, log=self.log)
     os.chdir(oldDir)
   elif '-qmkshrobj' in self.setCompilers.sharedLibraryFlags:
     cmd = linker+' '+linkerFlags+' -qmkshrobj -o '+sharedLib+' *.o '+externalLib
     oldDir = os.getcwd()
     os.chdir(tmpDir)
     self.executeShellCommand(cmd, log=self.log)
     os.chdir(oldDir)
   else:
     if osName == 'linux':
       cmd = linker+' -shared -Wl,-soname,'+os.path.basename(sharedLib)+' -o '+sharedLib+' *.o '+externalLib
     elif osName.startswith('darwin'):
       cmd   = ''
       flags = ''
       if not 'MACOSX_DEPLOYMENT_TARGET' in os.environ:
         cmd += 'MACOSX_DEPLOYMENT_TARGET=10.5 '
       if self.setCompilers.getLinkerFlags().find('-Wl,-commons,use_dylibs') > -1:
         flags += '-Wl,-commons,use_dylibs'
       cmd += self.setCompilers.getSharedLinker()+' -g  -dynamiclib -single_module -multiply_defined suppress -undefined dynamic_lookup '+flags+' -o '+sharedLib+' *.o -L'+libDir+' '+packageLibs+' '+sysLib+' '+extraLibs+' -lm -lc'
     elif osName == 'cygwin':
       cmd = linker+' '+linkerFlags+' -shared -o '+sharedLib+' *.o '+externalLib
     else:
       raise RuntimeError('Do not know how to make shared library for your crappy '+osName+' OS')
     oldDir = os.getcwd()
     os.chdir(tmpDir)
     self.executeShellCommand(cmd, log=self.log)
     os.chdir(oldDir)
     if hasattr(self.debuggers, 'dsymutil'):
       cmd = self.debuggers.dsymutil+' '+sharedLib
       self.executeShellCommand(cmd, log=self.log)
   return

 def expandArchive(self, archive, objDir):
   [shutil.rmtree(p) for p in os.listdir(objDir)]
   oldDir = os.getcwd()
   os.chdir(objDir)
   self.executeShellCommand(self.setCompilers.AR+' x '+archive, log = self.log)
   os.chdir(oldDir)
   return

 def buildSharedLibrary(self, libname):
   '''
   PETSC_LIB_DIR        = ${PETSC_DIR}/${PETSC_ARCH}/lib
   INSTALL_LIB_DIR	= ${PETSC_LIB_DIR}
   '''
   if self.sharedLibraries.useShared:
     libDir = os.path.join(self.petscdir.dir, self.arch.arch, 'lib')
     objDir = os.path.join(self.petscdir.dir, self.arch.arch, 'lib', libname+'-obj')
     self.logPrint('Making shared libraries in '+libDir)
     sharedLib = os.path.join(libDir, os.path.splitext(libname)[0]+'.'+self.setCompilers.sharedLibraryExt)
     archive   = os.path.join(libDir, os.path.splitext(libname)[0]+'.'+self.setCompilers.AR_LIB_SUFFIX)
     # Should we rebuild?
     rebuild = False
     if os.path.isfile(archive):
       if os.path.isfile(sharedLib):
         if os.path.getmtime(archive) >= os.path.getmtime(sharedLib):
           rebuild = True
       else:
         rebuild = True
     if rebuild:
       self.logPrint('Building '+sharedLib)
       #self.expandArchive(archive, objDir)
       self.linkShared(sharedLib, libDir, objDir)
     else:
       self.logPrint('Nothing to rebuild for shared library '+libname)
   else:
     self.logPrint('Shared libraries disabled')
   return

 def link(self, executable, objects, language):
   '''${CLINKER} -o $@ $^ ${PETSC_LIB}
      ${DSYMUTIL} $@'''
   self.compilers.pushLanguage(language)
   cmd = self.compilers.getFullLinkerCmd(objects+' -lpetsc', executable)
   self.logWrite(cmd+'\n', debugSection = self.debugSection, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.logPrint("ERROR IN LINK ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   # TODO: Move dsymutil stuff from PETSc.utilities.debuggers to config.compilers
   self.compilers.popLanguage()
   return [executable]

 def checkDir(self, dirname, allowExamples = False):
   '''Checks whether we should recurse into this directory
   - Excludes examples directory
   - Excludes contrib directory
   - Excludes tutorials directory
   - Excludes benchmarks directory
   - Checks whether fortran bindings are necessary
   - Checks makefile to see if compiler is allowed to visit this directory for this configuration'''
   base = os.path.basename(dirname)

   if base == 'examples' and not allowExamples: return False
   if not hasattr(self.compilers, 'FC'):
     if base.startswith('ftn-') or base.startswith('f90-'): return False
   if base == 'contrib':  return False
   #if base == 'tutorials' and not allowExamples:  return False
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
 
 def buildDir(self, dirname, dummy, objDir):
   ''' This is run in a PETSc source directory'''
   self.logWrite('Entering '+dirname+'\n', debugSection = 'screen', forceScroll = True)
   os.chdir(dirname)
   sourceMap = self.sortSourceFiles(dirname, objDir)
   objects   = []
   if sourceMap['C']:
     self.logPrint('Compiling C files '+str(sourceMap['C']))
     objects.extend(self.compileC(sourceMap['C'], objDir))
   if sourceMap['Fortran']:
     self.logPrint('Compiling Fortran files '+str(sourceMap['Fortran']))
     objects.extend(self.compileF(sourceMap['Fortran'], objDir))
   return objects

 def buildAll(self, libname, rootDir = None):
   self.setup()
   if rootDir is None:
     rootDir = self.argDB['rootDir']
   if rootDir == self.petscdir.dir:
     srcdirs = [os.path.join(rootDir, 'include'), os.path.join(rootDir, 'src')]
   else:
     srcdirs = [rootDir]
   if not any(map(self.checkDir, srcdirs)):
     self.logPrint('Nothing to be done')
   library = os.path.join(self.petscdir.dir, self.arch.arch, 'lib', libname)
   objDir  = os.path.join(self.petscdir.dir, self.arch.arch, 'lib', libname+'-obj')
   if not os.path.isdir(objDir): os.mkdir(objDir)
   if rootDir == self.petscdir.dir and not self.argDB['dependencies']:
     # Remove old library by default when rebuilding the entire package
     lib = os.path.splitext(library)[0]+'.'+self.setCompilers.AR_LIB_SUFFIX
     if os.path.isfile(lib):
       self.logPrint('Removing '+lib)
       os.unlink(lib)
   objects = []
   for srcdir in srcdirs:
     for root, dirs, files in os.walk(srcdir):
       self.logPrint('Processing '+root)
       objects += self.buildDir(root, files, objDir)
       for badDir in [d for d in dirs if not self.checkDir(os.path.join(root, d))]:
         dirs.remove(badDir)
   if len(objects):
     self.logPrint('Archiving files '+str(objects)+' into '+libname)
     self.archive(library, objects)
   self.ranlib(libname)
   self.buildSharedLibrary(libname)
   return

 def buildDependenciesFiles(self, names):
   if names:
     self.logPrint('Rebuilding dependency info for files '+str(names))
     for source in names:
       depFile = os.path.splitext(source)[0]+'.d'
       if os.path.isfile(depFile):
         self.logWrite('FOUND DEPENDENCY FILE '+depFile+'\n', debugSection = self.debugSection, forceScroll = True)
         if not self.dryRun:
           self.readDependencyFile(depFile)
   return
 
 def buildDependenciesDir(self, dirname, fnames, objDir):
   ''' This is run in a PETSc source directory'''
   self.logWrite('Entering '+dirname+'\n', debugSection = 'screen', forceScroll = True)
   os.chdir(dirname)
   sourceMap = self.sortSourceFiles(dirname, objDir)
   self.buildDependenciesFiles(sourceMap['Objects'])
   return

 def rebuildDependencies(self, libname, rootDir = None):
   if rootDir is None:
     rootDir = self.argDB['rootDir']
   if not self.checkDir(rootDir):
     self.logPrint('Nothing to be done')
   objDir = os.path.join(self.petscdir.dir, self.arch.arch, 'lib', libname+'-obj')
   for root, dirs, files in os.walk(rootDir):
     self.logPrint('Processing '+root)
     self.buildDependenciesDir(root, files, objDir)
     for badDir in [d for d in dirs if not self.checkDir(os.path.join(root, d))]:
       dirs.remove(badDir)
   return
 
 def cleanupTest(self, dirname, execname):
   # ${RM} $* *.o $*.mon.* gmon.out mon.out *.exe *.ilk *.pdb *.tds
   import re
   trash = re.compile('^('+execname+'(\.o|\.mon\.\w+|\.exe|\.ilk|\.pdb|\.tds)?|g?mon.out)$')
   for fname in os.listdir(dirname):
     if trash.match(fname):
       os.remove(fname)
   return

 def checkTestOutput(self, executable, output, testNum):
   outputName = os.path.abspath(os.path.join('output', executable+'_'+str(testNum)+'.out'))
   if not os.path.isfile(outputName):
     self.logPrint("MISCONFIGURATION: Regression output file %s (test %d) is missing" % (outputName, testNum), debugSection='screen')
   else:
     with file(outputName) as f:
       validOutput = f.read()
       if not validOutput == output:
         self.logPrint("TEST ERROR: Regression output for %s (test %d) does not match" % (executable, testNum), debugSection='screen')
         self.logPrint(validOutput, debugSection='screen')
         self.logPrint(output, debugSection='screen')
       else:
         self.logPrint("TEST SUCCESS: Regression output for %s (test %d) matches" % (executable, testNum), debugSection='screen')
   return

 def runTest(self, executable, testNum, **params):
   numProcs = params.get('numProcs', 1)
   args     = params.get('args', '')
   # TODO: Take this line out when configure is fixed
   # mpiexec = self.mpi.mpiexec.replace(' -n 1','').replace(' ', '\\ ')
   cmd = ' '.join([self.mpi.mpiexec, '-n', str(numProcs), os.path.abspath(executable), args])
   self.logWrite('Running test for '+executable+'\n'+cmd+'\n', debugSection = self.debugSection, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.logPrint("TEST ERROR: Failed to execute %s\n" % executable, debugSection = 'screen', forceScroll = True)
       self.logPrint(output+error, debugSection='screen', indent = 0, forceScroll = True)
     else:
       self.checkTestOutput(executable, output+error, testNum)
   return

 def regressionTestsDir(self, dirname, dummy):
   ''' This is run in a PETSc source directory'''
   self.logWrite('Entering '+dirname+'\n', debugSection = 'screen', forceScroll = True)
   os.chdir(dirname)
   sourceMap = self.sortSourceFiles(dirname)
   objects   = []
   if sourceMap['C']:
     self.logPrint('Compiling C files '+str(sourceMap['C']))
     self.compileC(sourceMap['C'])
   if sourceMap['Fortran']:
     if not self.fortrancpp.fortranDatatypes:
       self.logPrint('Compiling Fortran files '+str(sourceMap['Fortran']))
       self.compileF(sourceMap['Fortran'])
   if sourceMap['Objects']:
     packageNames = set([p.name for p in self.framework.packages])
     for obj in sourceMap['Objects']:
       # TESTEXAMPLES_C_X11 = ex3.PETSc runex3 ex3.rm
       # .PETSc: filters out messages from build
       # .rm: cleans up test
       executable = os.path.splitext(obj)[0]
       paramKey   = os.path.relpath(os.path.abspath(executable), self.petscdir.dir)
       testNum    = 1
       if paramKey in regressionRequirements:
         if not regressionRequirements[paramKey].issubset(packageNames):
           continue
       self.logPrint('Linking object '+obj+' into '+executable)
       # TODO: Fix this hack
       if executable[-1] == 'f':
         self.link(executable, obj, 'FC')
       else:
         self.link(executable, obj, self.languages.clanguage)
       self.runTest(executable, testNum, **regressionParameters.get(paramKey, {}))
       testNum += 1
       while '%s_%d' % (paramKey, testNum) in regressionParameters:
         self.runTest(executable, testNum, **regressionParameters.get('%s_%d' % (paramKey, testNum), {}))
         testNum += 1
       self.cleanupTest(dirname, executable)
   return

 def regressionTests(self, rootDir = None):
   if rootDir is None:
     rootDir = self.argDB['rootDir']
   if not self.checkDir(rootDir, allowExamples = True):
     self.logPrint('Nothing to be done')
   for root, dirs, files in os.walk(rootDir):
     self.logPrint('Processing '+root)
     if 'examples' in dirs:
       for exroot, exdirs, exfiles in os.walk(os.path.join(root, 'examples')):
         self.logPrint('Processing '+exroot)
         print '  Testing in root',root
         self.regressionTestsDir(exroot, exfiles)
         for badDir in [d for d in exdirs if not self.checkDir(os.path.join(exroot, d), allowExamples = True)]:
           exdirs.remove(badDir)
     for badDir in [d for d in dirs if not self.checkDir(os.path.join(root, d))]:
       dirs.remove(badDir)
   return

 def run(self):
   self.setup()
   if self.argDB['rebuildDependencies']:
     self.rebuildDependencies('libpetsc')
   if self.argDB['buildLibraries']:
     self.buildAll('libpetsc')
   if self.argDB['regressionTests']:
     self.regressionTests()
   self.cleanup()
   return

if __name__ == '__main__':
  PETScMaker().run()
