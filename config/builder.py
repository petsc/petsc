#!/usr/bin/env python

from __future__ import with_statement  # For python-2.5

import os, sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config'))
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))

import logger, script

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

class SourceDatabaseDict(object):
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
    self.dependencyGraph[vertex] = [(dep, SourceDatabaseDict.marker(dep)) for dep in deps]
    return

  def updateNode(self, vertex):
    self.dependencyGraph[vertex] = [(dep, SourceDatabaseDict.marker(dep)) for dep,mark in self.dependencyGraph[vertex]]
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

class SourceNode:
  def __init__(self, filename, marker):
    self.filename = filename
    self.marker   = marker
    return

  def __str__(self):
    #return self.filename+' ('+str(self.marker)+')'
    return self.filename

  def __repr__(self):
    return self.__str__()

  def __getitem__(self, pos):
    if   pos == 0: return self.filename
    elif pos == 1: return self.marker
    raise IndexError

  def __eq__(self, n):
    return n.filename == self.filename and n.marker == self.marker

  def __hash__(self):
    return self.filename.__hash__()

class SourceDatabase(logger.Logger):
  '''This can be replaced by the favorite software of Jed'''
  def __init__(self, argDB, log):
    logger.Logger.__init__(self, argDB = argDB, log = log)
    self.setup()
    # Vertices are (filename, consistency marker) pairs
    #   Arcs indicate a dependence
    import graph
    self.dependencyGraph = graph.DirectedGraph()
    return

  def __len__(self):
    return len(self.dependencyGraph)

  def __str__(self):
    return str(self.dependencyGraph)

  @staticmethod
  def marker(dep):
    import hashlib
    if not os.path.isfile(dep):
      return 0
    with file(dep) as f:
      mark = hashlib.sha1(f.read()).digest()
    return mark

  @staticmethod
  def vertex(filename):
    return SourceNode(filename, SourceDatabase.marker(filename))

  def setNode(self, vertex, deps):
    self.dependencyGraph.addEdges(SourceDatabase.vertex(vertex), [SourceDatabase.vertex(dep) for dep in deps])
    return

  def updateNode(self, vertex):
    v = SourceDatabase.vertex(vertex)
    self.dependencyGraph.clearEdges(v, inOnly = True)
    self.dependencyGraph.addEdges([SourceDatabase.vertex(dep) for dep,mark in self.dependencyGraph.getEdges(v)[0]])
    return

  def rebuildArc(self, vertex, dep, mark):
    import hashlib
    with file(dep) as f:
      newMark = hashlib.sha1(f.read()).digest()
    return not mark == newMark

  def rebuild(self, vertex):
    self.logPrint('Checking for rebuild of '+str(vertex))
    v = SourceDatabase.vertex(vertex)
    try:
      if not os.path.isfile(vertex):
        self.logPrint('    %s does not exist' % vertex)
      for dep,mark in self.dependencyGraph.getEdges(v)[0]:
        if self.rebuildArc(vertex, dep, mark):
          self.logPrint('    dep '+str(dep)+' is changed')
          return True
    except KeyError, e:
      self.logPrint('    %s not in database' % vertex)
      return True
    return False

  def topologicalSort(self, predicate):
    import graph
    for vertex,marker in graph.DirectedGraph.topologicalSort(self.dependencyGraph):
      if predicate(vertex):
        yield vertex
    return

class DirectoryTreeWalker(logger.Logger):
  def __init__(self, argDB, log, configInfo, allowFortran = None, allowExamples = False):
    logger.Logger.__init__(self, argDB = argDB, log = log)
    self.configInfo = configInfo
    if allowFortran is None:
      self.allowFortran  = hasattr(self.configInfo.compilers, 'FC')
    else:
      self.allowFortran  = allowFortran
    self.allowExamples   = allowExamples
    self.setup()
    self.collectDefines()
    return

  def collectDefines(self):
    self.defines = {}
    self.defines.update(self.configInfo.base.defines)
    self.defines.update(self.configInfo.compilers.defines)
    self.defines.update(self.configInfo.libraryOptions.defines)
    for p in self.configInfo.framework.packages:
      self.defines.update(p.defines)
    return

  def checkSourceDir(self, dirname):
    '''Checks makefile to see if compiler is allowed to visit this directory for this configuration'''
    # Require makefile
    makename = os.path.join(dirname, 'makefile')
    if not os.path.isfile(makename):
      if os.path.isfile(os.path.join(dirname, 'Makefile')): self.logPrint('ERROR: Change Makefile to makefile in '+dirname, debugSection = 'screen')
      return False
    # Parse makefile
    import re
    reg = re.compile(' [ ]*')
    with file(makename) as f:
      for line in f.readlines():
        if not line.startswith('#requires'): continue
        # Remove leader and redundant spaces and split into names
        reqType, reqValue = reg.sub(' ', line[9:-1].strip()).split(' ')[0:2]
        # Check requirements
        if reqType == 'scalar':
          if not self.configInfo.scalarType.scalartype == reqValue:
            self.logPrint('Rejecting '+dirname+' because scalar type '+self.configInfo.scalarType.scalartype+' is not '+reqValue)
            return False
        elif reqType == 'language':
          if reqValue == 'CXXONLY' and self.configInfo.languages.clanguage == 'C':
            self.logPrint('Rejecting '+dirname+' because language is '+self.configInfo.languages.clanguage+' is not C++')
            return False
        elif reqType == 'precision':
          if not self.configInfo.scalarType.precision == reqValue:
            self.logPrint('Rejecting '+dirname+' because precision '+self.configInfo.scalarType.precision+' is not '+reqValue)
            return False
        elif reqType == 'function':
          if not reqValue in ['\'PETSC_'+f+'\'' for f in self.configInfo.functions.defines]:
            self.logPrint('Rejecting '+dirname+' because function '+reqValue+' does not exist')
            return False
        elif reqType == 'define':
          if not reqValue in ['\'PETSC_'+d+'\'' for d in self.defines]:
            self.logPrint('Rejecting '+dirname+' because define '+reqValue+' does not exist')
            return False
        elif reqType == 'package':
          if not self.allowFortran and reqValue in ['\'PETSC_HAVE_FORTRAN\'', '\'PETSC_USING_F90\'']:
            self.logPrint('Rejecting '+dirname+' because fortran is not being used')
            return False
          elif not self.configInfo.libraryOptions.useLog and reqValue == '\'PETSC_USE_LOG\'':
            self.logPrint('Rejecting '+dirname+' because logging is turned off')
            return False
          elif not self.configInfo.libraryOptions.useFortranKernels and reqValue == '\'PETSC_USE_FORTRAN_KERNELS\'':
            self.logPrint('Rejecting '+dirname+' because fortran kernels are turned off')
            return False
          elif not self.configInfo.mpi.usingMPIUni and reqValue == '\'PETSC_HAVE_MPIUNI\'':
            self.logPrint('Rejecting '+dirname+' because we are not using MPIUNI')
            return False
          elif not reqValue in ['\'PETSC_HAVE_'+p.PACKAGE+'\'' for p in self.configInfo.framework.packages]:
            self.logPrint('Rejecting '+dirname+' because package '+reqValue+' is not installed')
            return False
        else:
          self.logPrint('ERROR: Invalid requirement type %s in %s' % (reqType, makename), debugSection = 'screen')
          return False
    return True

  def checkDir(self, dirname):
    '''Checks whether we should recurse into this directory
    - Excludes ftn-* and f90-* if self.allowFortran is False
    - Excludes examples directory if self.allowExamples is False
    - Excludes contrib, tutorials, and benchmarks directory
    - Otherwise calls checkSourceDir()'''
    base = os.path.basename(dirname)

    if base == 'examples':
      return self.allowExamples
    elif base in ['externalpackages', 'projects', 'tutorials', 'benchmarks', 'contrib']:
      return False
    elif base.startswith('ftn-') or base.startswith('f90-'):
      return self.allowFortran
    return self.checkSourceDir(dirname)

  def walk(self, rootDir):
    if not self.checkDir(rootDir):
      self.logPrint('Nothing to be done in '+self.rootDir)
    for root, dirs, files in os.walk(rootDir):
      self.logPrint('Processing '+root)
      yield root, files
      for badDir in [d for d in dirs if not self.checkDir(os.path.join(root, d))]:
        dirs.remove(badDir)
    return

class SourceFileManager(logger.Logger):
  def __init__(self, argDB, log, configInfo):
    logger.Logger.__init__(self, argDB = argDB, log = log)
    self.configInfo = configInfo
    self.setup()
    return

  def getObjectName(self, source, objDir = None):
    '''Get object file name corresponding to a source file'''
    if objDir is None:
      compilerObj = self.configInfo.compiler['C'].getTarget(source)
    else:
      compilerObj = os.path.join(objDir, self.configInfo.compiler['C'].getTarget(os.path.basename(source)))
    return compilerObj

  def sortSourceFiles(self, fnames, objDir = None):
    '''Sorts source files by language (returns dictionary with language keys)'''
    cnames    = []
    cxxnames  = []
    cudanames = []
    f77names  = []
    f90names  = []
    for f in fnames:
      ext = os.path.splitext(f)[1]
      if ext == '.c':
        cnames.append(f)
      elif ext in ['.cxx', '.cpp', '.cc']:
        if self.configInfo.languages.clanguage == 'Cxx':
          cxxnames.append(f)
      elif ext == '.cu':
        cudanames.append(f)
      elif ext == '.F':
        if hasattr(self.configInfo.compilers, 'FC'):
          f77names.append(f)
      elif ext == '.F90':
        if hasattr(self.configInfo.compilers, 'FC') and self.configInfo.compilers.fortranIsF90:
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

class DependencyBuilder(logger.Logger):
  def __init__(self, argDB, log, sourceManager, sourceDatabase, objDir):
    logger.Logger.__init__(self, argDB = argDB, log = log)
    self.sourceManager  = sourceManager
    self.sourceDatabase = sourceDatabase
    self.objDir         = objDir
    self.setup()
    return

  def readDependencyFile(self, dirname, source, depFile):
    '''Read *.d file with dependency information and store it in the source database'''
    with file(depFile) as f:
      target, deps = f.read().split(':')
      assert(target == self.sourceManager.getObjectName(source))
    deps = deps.replace('\\','').split()
    assert(deps[0] == source)
    self.sourceDatabase.setNode(os.path.join(dirname, source), [os.path.join(dirname, d) for d in deps[1:]])
    return

  def buildDependency(self, dirname, source):
    self.logPrint('Rebuilding dependency info for '+os.path.join(dirname, source))
    depFile = os.path.splitext(source)[0]+'.d'
    if os.path.isfile(depFile):
      self.logPrint('Found dependency file '+depFile)
      self.readDependencyFile(dirname, source, depFile)
    return

  def buildDependencies(self, dirname, fnames):
    ''' This is run in a PETSc source directory'''
    self.logWrite('Building dependencies in '+dirname+'\n', debugSection = 'screen', forceScroll = True)
    oldDir = os.getcwd()
    os.chdir(dirname)
    sourceMap = self.sourceManager.sortSourceFiles(fnames, self.objDir)
    for language in sourceMap.keys():
      if language == 'Objects': continue
      for source in sourceMap[language]:
        self.buildDependency(dirname, source)
    os.chdir(oldDir)
    return

  def buildDependenciesF90(self):
    '''We have to hardcode F90 module dependencies (shaking fist)'''
    pdir = self.sourceManager.configInfo.petscdir.dir
    if self.sourceManager.configInfo.mpi.usingMPIUni:
      self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'sys', 'f90-mod', 'petscsysmod.F'), [os.path.join(pdir, 'src', 'sys', 'mpiuni', 'f90-mod', 'mpiunimod.F')])
    self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'vec',  'f90-mod', 'petscvecmod.F'),  [os.path.join(pdir, 'src', 'sys',  'f90-mod', 'petscsysmod.F')])
    self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'mat',  'f90-mod', 'petscmatmod.F'),  [os.path.join(pdir, 'src', 'vec',  'f90-mod', 'petscvecmod.F')])
    self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'dm',   'f90-mod', 'petscdmmod.F'),   [os.path.join(pdir, 'src', 'mat',  'f90-mod', 'petscmatmod.F')])
    self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'ksp',  'f90-mod', 'petsckspmod.F'),  [os.path.join(pdir, 'src', 'dm',   'f90-mod', 'petscdmmod.F')])
    self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'snes', 'f90-mod', 'petscsnesmod.F'), [os.path.join(pdir, 'src', 'ksp',  'f90-mod', 'petsckspmod.F')])
    self.sourceDatabase.setNode(os.path.join(pdir, 'src', 'ts',   'f90-mod', 'petsctsmod.F'),   [os.path.join(pdir, 'src', 'snes', 'f90-mod', 'petscsnesmod.F')])
    return

class PETScConfigureInfo(object):
  def __init__(self, framework):
    self.framework = framework
    self.setupModules()
    self.compiler = {}
    self.compiler['C'] = self.framework.getCompilerObject(self.languages.clanguage)
    self.compiler['C'].checkSetup()
    return

  def setupModules(self):
    self.mpi             = self.framework.require('config.packages.MPI',         None)
    self.base            = self.framework.require('config.base',                 None)
    self.setCompilers    = self.framework.require('config.setCompilers',         None)
    self.arch            = self.framework.require('PETSc.utilities.arch',        None)
    self.petscdir        = self.framework.require('PETSc.utilities.petscdir',    None)
    self.languages       = self.framework.require('PETSc.utilities.languages',   None)
    self.debugging       = self.framework.require('PETSc.utilities.debugging',   None)
    self.debuggers       = self.framework.require('PETSc.utilities.debuggers',   None)
    self.make            = self.framework.require('config.programs',        None)
    self.CHUD            = self.framework.require('PETSc.utilities.CHUD',        None)
    self.compilers       = self.framework.require('config.compilers',            None)
    self.types           = self.framework.require('config.types',                None)
    self.headers         = self.framework.require('config.headers',              None)
    self.functions       = self.framework.require('config.functions',            None)
    self.libraries       = self.framework.require('config.libraries',            None)
    self.scalarType      = self.framework.require('PETSc.utilities.scalarTypes', None)
    self.memAlign        = self.framework.require('PETSc.utilities.memAlign',    None)
    self.libraryOptions  = self.framework.require('PETSc.utilities.libraryOptions', None)
    self.fortrancpp      = self.framework.require('PETSc.utilities.fortranCPP', None)
    self.debuggers       = self.framework.require('PETSc.utilities.debuggers', None)
    self.sharedLibraries = self.framework.require('PETSc.utilities.sharedLibraries', None)
    return

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

 def setupHelp(self, help):
   import nargs

   help = script.Script.setupHelp(self, help)
   #help.addArgument('PETScMaker', '-rootDir', nargs.ArgDir(None, os.environ['PETSC_DIR'], 'The root directory for this build', isTemporary = 1))
   help.addArgument('PETScMaker', '-rootDir', nargs.ArgDir(None, os.getcwd(), 'The root directory for this build', isTemporary = 1))
   help.addArgument('PETScMaker', '-dryRun',  nargs.ArgBool(None, False, 'Only output what would be run', isTemporary = 1))
   help.addArgument('PETScMaker', '-dependencies',  nargs.ArgBool(None, True, 'Use dependencies to control build', isTemporary = 1))
   help.addArgument('PETScMaker', '-buildLibraries', nargs.ArgBool(None, True, 'Build the PETSc libraries', isTemporary = 1))
   help.addArgument('PETScMaker', '-regressionTests', nargs.ArgBool(None, False, 'Only run regression tests', isTemporary = 1))
   help.addArgument('PETScMaker', '-rebuildDependencies', nargs.ArgBool(None, False, 'Force dependency information to be recalculated', isTemporary = 1))
   help.addArgument('PETScMaker', '-verbose', nargs.ArgInt(None, 0, 'The verbosity level', min = 0, isTemporary = 1))

   help.addArgument('PETScMaker', '-maxSources', nargs.ArgInt(None, -1, 'The maximum number of source files in a directory', min = -1, isTemporary = 1))
   return help

 def setup(self):
   '''
   - Loads configure information
   - Loads dependency information (unless it will be recalculated)
   '''
   script.Script.setup(self)
   if self.dryRun or self.verbose:
     self.debugSection = 'screen'
   else:
     self.debugSection = None
   self.rootDir = os.path.abspath(self.argDB['rootDir'])
   # Load configure information
   self.framework  = self.loadConfigure()
   self.configInfo = PETScConfigureInfo(self.framework)
   # Setup directories
   self.petscDir     = self.configInfo.petscdir.dir
   self.petscArch    = self.configInfo.arch.arch
   self.petscConfDir = os.path.join(self.petscDir, self.petscArch, 'conf')
   self.petscLibDir  = os.path.join(self.petscDir, self.petscArch, 'lib')
   # Setup subobjects
   self.sourceManager = SourceFileManager(self.argDB, self.log, self.configInfo)
   return

 def cleanupLog(self, framework, confDir):
   '''Move configure.log to PROJECT_ARCH/conf - and update configure.log.bkp in both locations appropriately'''
   import os

   self.log.flush()
   if hasattr(framework, 'logName'):
     logName         = framework.logName
   else:
     logName         = 'make.log'
   logFile           = os.path.join(self.petscDir, logName)
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
   '''
   - Move logs to proper location
   - Save dependency information
   '''
   root    = self.petscDir
   arch    = self.petscArch
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

 def getObjDir(self, libname):
   return os.path.join(self.petscDir, self.petscArch, 'lib', libname+'-obj')

 def getPackageInfo(self):
   '''Get package include and library information from configure data'''
   packageIncludes = []
   packageLibs     = []
   for p in self.configInfo.framework.packages:
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
   packageLibs     = self.configInfo.libraries.toStringNoDupes(packageLibs+self.configInfo.libraries.math)
   packageIncludes = self.configInfo.headers.toStringNoDupes(packageIncludes)
   return packageIncludes, packageLibs

 def storeObjects(self, objects):
   for o in objects:
     locObj = os.path.basename(o)
     self.logPrint('Moving %s to %s' % (locObj, o))
     if not self.dryRun:
       if not os.path.isfile(locObj):
         print 'ERROR: Missing object file',locObj
       else:
         shutil.move(locObj, o)
   return

 def compile(self, language, source, objDir = None):
   if not len(source):
     self.logPrint('Nothing to build', debugSection = self.debugSection)
     return
   self.configInfo.setCompilers.pushLanguage(language)
   packageIncludes, packageLibs = self.getPackageInfo()
   compiler = self.configInfo.setCompilers.getCompiler()
   objects  = [self.sourceManager.getObjectName(s, objDir) for s in source]
   includes = ['-I'+inc for inc in [os.path.join(self.petscDir, self.petscArch, 'include'), os.path.join(self.petscDir, 'include')]]
   flags    = []
   flags.append(self.configInfo.setCompilers.getCompilerFlags())                        # Add PCC_FLAGS
   flags.extend([self.configInfo.setCompilers.CPPFLAGS, self.configInfo.CHUD.CPPFLAGS]) # Add CPP_FLAGS
   flags.append('-D__INSDIR__='+os.getcwd().replace(self.petscDir, ''))                 # Define __INSDIR__
   # TODO: Move this up to configure
   if self.argDB['dependencies']: flags.append('-MMD')
   cmd      = ' '.join([compiler]+['-c']+includes+[packageIncludes]+flags+source)
   self.logWrite(cmd+'\n', debugSection = self.debugSection, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.logPrint("ERROR IN C COMPILE ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   self.configInfo.setCompilers.popLanguage()
   self.storeObjects(objects)
   return objects

 def compileC(self, source, objDir = None):
   return self.compile(self.configInfo.languages.clanguage, source, objDir)

 def compileFortran(self, source, objDir = None):
   objects = self.compile('FC', source, objDir)
   # Copy any module files produced into the include directory
   for locMod in os.listdir(os.getcwd()):
     if os.path.splitext(locMod)[1] == '.mod':
       mod = os.path.join(self.petscDir, self.petscArch, 'include', locMod)
       self.logPrint('Moving F90 module %s to %s' % (locMod, mod))
       shutil.move(locMod, mod)
   return objects

 def archive(self, library, objects):
   '''${AR} ${AR_FLAGS} ${LIBNAME} $*.o'''
   lib = os.path.splitext(library)[0]+'.'+self.configInfo.setCompilers.AR_LIB_SUFFIX
   if self.rootDir == self.petscDir:
     cmd = ' '.join([self.configInfo.setCompilers.AR, self.configInfo.setCompilers.FAST_AR_FLAGS, lib]+objects)
   else:
     cmd = ' '.join([self.configInfo.setCompilers.AR, self.configInfo.setCompilers.AR_FLAGS, lib]+objects)
   self.logWrite(cmd+'\n', debugSection = self.debugSection, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.logPrint("ERROR IN ARCHIVE ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   return [library]

 def ranlib(self, library):
   '''${ranlib} ${LIBNAME} '''
   library = os.path.join(self.petscLibDir, library)
   lib     = os.path.splitext(library)[0]+'.'+self.configInfo.setCompilers.AR_LIB_SUFFIX
   cmd     = ' '.join([self.configInfo.setCompilers.RANLIB, lib])
   self.logPrint('Running ranlib on '+lib)
   self.logWrite(cmd+'\n', debugSection = self.debugSection, forceScroll = True)
   if not self.dryRun:
     (output, error, status) = self.executeShellCommand(cmd, checkCommand = noCheckCommand, log=self.log)
     if status:
       self.logPrint("ERROR IN RANLIB ******************************", debugSection='screen')
       self.logPrint(output+error, debugSection='screen')
   return

 def linkShared(self, sharedLib, libDir, tmpDir):
   osName = sys.platform
   # PCC_LINKER PCC_LINKER_FLAGS
   linker      = self.configInfo.setCompilers.getLinker()
   linkerFlags = self.configInfo.setCompilers.getLinkerFlags()
   packageIncludes, packageLibs = self.getPackageInfo()
   extraLibs = self.configInfo.libraries.toStringNoDupes(self.configInfo.compilers.flibs+self.configInfo.compilers.cxxlibs+self.configInfo.compilers.LIBS.split(' '))+self.configInfo.CHUD.LIBS
   sysLib      = ''
   sysLib.replace('-Wl,-rpath', '-L')
   externalLib = packageLibs+' '+extraLibs
   externalLib.replace('-Wl,-rpath', '-L')
   # Move this switch into the sharedLibrary module
   if self.configInfo.setCompilers.isSolaris() and self.configInfo.setCompilers.isGNU(self.configInfo.framework.getCompiler()):
     cmd = self.configInfo.setCompilers.LD+' -G -h '+os.path.basename(sharedLib)+' *.o -o '+sharedLib+' '+sysLib+' '+externalLib
     oldDir = os.getcwd()
     os.chdir(tmpDir)
     self.executeShellCommand(cmd, log=self.log)
     os.chdir(oldDir)
   elif '-qmkshrobj' in self.configInfo.setCompilers.sharedLibraryFlags:
     cmd = linker+' '+linkerFlags+' -qmkshrobj -o '+sharedLib+' *.o '+externalLib
     oldDir = os.getcwd()
     os.chdir(tmpDir)
     self.executeShellCommand(cmd, log=self.log)
     os.chdir(oldDir)
   else:
     if osName == 'linux2':
       cmd = linker+' -shared -Wl,-soname,'+os.path.basename(sharedLib)+' -o '+sharedLib+' *.o '+externalLib
     elif osName.startswith('darwin'):
       cmd   = ''
       flags = ''
       if not 'MACOSX_DEPLOYMENT_TARGET' in os.environ:
         cmd += 'MACOSX_DEPLOYMENT_TARGET=10.5 '
       if self.configInfo.setCompilers.getLinkerFlags().find('-Wl,-commons,use_dylibs') > -1:
         flags += '-Wl,-commons,use_dylibs'
       cmd += self.configInfo.setCompilers.getSharedLinker()+' -g  -dynamiclib -single_module -multiply_defined suppress -undefined dynamic_lookup '+flags+' -o '+sharedLib+' *.o -L'+libDir+' '+packageLibs+' '+sysLib+' '+extraLibs+' -lm -lc'
     elif osName == 'cygwin':
       cmd = linker+' '+linkerFlags+' -shared -o '+sharedLib+' *.o '+externalLib
     else:
       raise RuntimeError('Do not know how to make shared library for your crappy '+osName+' OS')
     oldDir = os.getcwd()
     os.chdir(tmpDir)
     self.executeShellCommand(cmd, log=self.log)
     os.chdir(oldDir)
     if hasattr(self.configInfo.debuggers, 'dsymutil'):
       cmd = self.configInfo.debuggers.dsymutil+' '+sharedLib
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
   if self.configInfo.sharedLibraries.useShared:
     libDir = self.petscLibDir
     objDir = self.getObjDir(libname)
     self.logPrint('Making shared libraries in '+libDir)
     sharedLib = os.path.join(libDir, os.path.splitext(libname)[0]+'.'+self.configInfo.setCompilers.sharedLibraryExt)
     archive   = os.path.join(libDir, os.path.splitext(libname)[0]+'.'+self.configInfo.setCompilers.AR_LIB_SUFFIX)
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
     if hasattr(self.debuggers, 'dsymutil'):
       (output, error, status) = self.executeShellCommand(self.debuggers.dsymutil+' '+executable, checkCommand = noCheckCommand, log=self.log)
       if status:
         self.logPrint("ERROR IN LINK ******************************", debugSection='screen')
         self.logPrint(output+error, debugSection='screen')
   self.compilers.popLanguage()
   return [executable]

 def buildDir(self, dirname, files, objDir):
   ''' This is run in a PETSc source directory'''
   self.logWrite('Building in '+dirname+'\n', debugSection = 'screen', forceScroll = True)
   oldDir = os.getcwd()
   os.chdir(dirname)
   sourceMap = self.sourceManager.sortSourceFiles(files, objDir)
   objects   = []
   for language in ['C', 'Fortran', 'Cuda']:
     if sourceMap[language]:
       self.logPrint('Compiling %s files %s' % (language, str(sourceMap['C'])))
       objects.extend(getattr(self, 'compile'+language)(sourceMap[language], objDir))
   os.chdir(oldDir)
   return objects

 def buildFile(self, filename, objDir):
   ''' This is run in a PETSc source directory'''
   self.logWrite('Building '+filename+'\n', debugSection = 'screen', forceScroll = True)
   sourceMap = self.sourceManager.sortSourceFiles([filename], objDir)
   objects   = []
   for language in ['C', 'Fortran', 'Cuda']:
     if sourceMap[language]:
       self.logPrint('Compiling %s files %s' % (language, str(sourceMap['C'])))
       objects.extend(getattr(self, 'compile'+language)(sourceMap[language], objDir))
   return objects

 def buildLibraries(self, libname, rootDir):
   if not self.argDB['buildLibraries']: return
   self.logPrint('Building Libraries')
   library = os.path.join(self.petscDir, self.petscArch, 'lib', libname)
   objDir  = self.getObjDir(libname)
   if not os.path.isdir(objDir): os.mkdir(objDir)
   # Remove old library by default when rebuilding the entire package
   if rootDir == self.petscDir and not self.argDB['dependencies']:
     lib = os.path.splitext(library)[0]+'.'+self.configInfo.setCompilers.AR_LIB_SUFFIX
     if os.path.isfile(lib):
       self.logPrint('Removing '+lib)
       os.unlink(lib)

   objects = []
   if len(self.sourceDatabase):
     def check(filename):
       if self.sourceDatabase.rebuild(filename): return True
       for obj in self.sourceManager.sortSourceFiles([filename], objDir)['Objects']:
         if not os.path.isfile(obj):
           self.logPrint('    object file '+obj+' is missing')
           return True
       return False
     import graph
     for filename in self.sourceDatabase.topologicalSort(check):
       objects += self.buildFile(filename, objDir)
   else:
     walker  = DirectoryTreeWalker(self.argDB, self.log, self.configInfo)
     for root, files in walker.walk(rootDir):
       self.logPrint('Building directory '+root)
       objects += self.buildDir(root, files, objDir)

   if len(objects):
     self.logPrint('Archiving files '+str(objects)+' into '+libname)
     self.archive(library, objects)
   #self.ranlib(libname)
   #self.buildSharedLibrary(libname)
   return

 def rebuildDependencies(self, libname, rootDir):
   '''Calculates build dependencies and stores them in a database
   - If --dependencies is False, ignore them
   '''
   if self.argDB['dependencies']:
     dbFilename = os.path.join(self.petscConfDir, 'source.db')

     if not self.argDB['rebuildDependencies'] and os.path.isfile(dbFilename):
       self.logPrint('Loading Dependencies')
       import cPickle

       with file(dbFilename, 'rb') as f:
         self.sourceDatabase = cPickle.load(f)
       self.sourceDatabase.verbose = self.verbose
     else:
       self.logPrint('Rebuilding Dependencies')
       self.sourceDatabase = SourceDatabase(self.argDB, self.log)
       depBuilder          = DependencyBuilder(self.argDB, self.log, self.sourceManager, self.sourceDatabase, self.getObjDir(libname))
       walker              = DirectoryTreeWalker(self.argDB, self.log, self.configInfo)

       for root, files in walker.walk(rootDir):
         depBuilder.buildDependencies(root, files)
       depBuilder.buildDependenciesF90()
   else:
     self.logPrint('Disabling dependency tracking')
     self.sourceDatabase = NullSourceDatabase()
   if self.verbose > 3:
     import graph
     print 'Source database:'
     for filename in self.sourceDatabase.topologicalSort(lambda x: True):
       print '  ',filename
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
   sourceMap = self.sourceManager.sortSourceFiles(dirname)
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
       paramKey   = os.path.relpath(os.path.abspath(executable), self.petscDir)
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

 def regressionTests(self, rootDir):
   if not self.argDB['regressionTests']: return
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
   self.rebuildDependencies('libpetsc', self.rootDir)
   self.buildLibraries('libpetsc', self.rootDir)
   self.regressionTests(self.rootDir)
   self.cleanup()
   return

if __name__ == '__main__':
  PETScMaker().run()
