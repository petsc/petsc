import bs
import compile
import fileset
import link
import nargs
import transform
import BSTemplates.sidlStructs

import distutils.sysconfig
import os
import sys

def guessProject(dir):
  for project in bs.argDB['installedprojects']:
    if project.getRoot() == dir:
      return project
  return bs.Project(os.path.basename(dir).lower(), '')

class UsingCompiler:
  '''This class handles all interaction specific to a compiled language'''
  def __init__(self, usingSIDL):
    self.usingSIDL      = usingSIDL
    self.defines        = ['PIC']
    self.includeDirs    = BSTemplates.sidlStructs.SIDLPackageDict(usingSIDL)
    self.extraLibraries = BSTemplates.sidlStructs.SIDLPackageDict(usingSIDL)
    return

  def getClientLanguages(self):
    '''Returns all languages involved in the client library'''
    return [self.getLanguage()]

  def getDefines(self):
    return self.defines

  def getClientLibrary(self, project, lang, isArchive = 1, root = None):
    '''Client libraries following the naming scheme: lib<project>-<lang>-client.a'''
    if not root:  root = os.path.join(project.getRoot(), 'lib')
    if isArchive:  ext = '.a'
    else:          ext = '.so'
    return fileset.FileSet([os.path.join(root, 'lib'+project.getName()+'-'+lang.lower()+'-client'+ext)])

  def getServerLibrary(self, project, lang, package, isArchive = 1, root = None):
    '''Server libraries following the naming scheme: lib<project>-<lang>-<package>-server.a'''
    if not root:  root = os.path.join(project.getRoot(), 'lib')
    if isArchive:  ext = '.a'
    else:          ext = '.so'
    return fileset.FileSet([os.path.join(root, 'lib'+project.getName()+'-'+lang.lower()+'-'+package+'-server'+ext)])

  def getClientCompileTarget(self, project):
    sourceDir = self.usingSIDL.getClientRootDir(self.getLanguage())
    # Client filter
    clientFilter = []
    for language in self.getClientLanguages():
      tag = language.lower().replace('+', 'x')
      clientFilter.append(transform.FileFilter(lambda source: self.usingSIDL.compilerDefaults.isClient(source, sourceDir), tags = [tag, 'old '+tag]))
    if len(clientFilter) == 1: clientFilter = clientFilter[0]
    # Client compiler
    compilers = self.getCompiler(self.getClientLibrary(project, self.getLanguage()))
    if not isinstance(compilers, list): compilers = [compilers]
    for compiler in compilers:
      compiler.defines.extend(self.getDefines())
      compiler.includeDirs.append(sourceDir)
      compiler.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
      compiler.includeDirs.extend(self.includeDirs[self.getLanguage()])
      for dir in self.usingSIDL.repositoryDirs:
        includeDir = self.usingSIDL.getClientRootDir(self.getLanguage(), root = dir)
        if os.path.isdir(includeDir):
          compiler.includeDirs.append(includeDir)
    if len(compilers) == 1: compilers = compilers[0]
    return [self.getTagger(sourceDir), clientFilter, compilers]

  def getClientLinkTarget(self, project, doLibraryCheck = 1):
    libraries = fileset.FileSet([])
    libraries.extend(self.usingSIDL.extraLibraries[self.getLanguage()])
    libraries.extend(self.extraLibraries[self.getLanguage()])
    for dir in self.usingSIDL.repositoryDirs:
      for lib in self.getClientLibrary(guessProject(dir), self.getLanguage(), isArchive = 0, root = os.path.join(dir, 'lib')):
        if os.path.isfile(lib):
          libraries.append(lib)
    linker    = link.LinkSharedLibrary(self.usingSIDL.sourceDB, extraLibraries = libraries)
    linker.doLibraryCheck = doLibraryCheck
    return [link.TagLibrary(self.usingSIDL.sourceDB), linker]

  def getServerLinkTarget(self, project, package, doLibraryCheck = 1):
    libraries = fileset.FileSet([])
    libraries.extend(self.usingSIDL.extraLibraries[package])
    if not self.getLanguage() in self.usingSIDL.internalClientLanguages[package]:
      libraries.extend(self.getClientLibrary(project, self.getLanguage()))
    libraries.extend(self.extraLibraries[package])
    for dir in self.usingSIDL.repositoryDirs:
      for lib in self.getClientLibrary(guessProject(dir), self.getLanguage(), isArchive = 0, root = os.path.join(dir, 'lib')):
        if os.path.isfile(lib):
          libraries.append(lib)
    linker    = link.LinkSharedLibrary(self.usingSIDL.sourceDB, extraLibraries = libraries)
    linker.doLibraryCheck = doLibraryCheck
    return [link.TagLibrary(self.usingSIDL.sourceDB), linker]

  def getExecutableCompileTarget(self, project, sources, executable):
    baseName = os.path.splitext(os.path.basename(executable[0]))[0] 
    library  = fileset.FileSet([os.path.join(project.getRoot(), 'lib', 'lib'+baseName+'.a')])
    compiler = self.getCompiler(library)
    # Might be all internal clients
    if os.path.isdir(self.usingSIDL.getClientRootDir(self.getLanguage())):
      compiler.includeDirs.append(self.usingSIDL.getClientRootDir(self.getLanguage()))
    compiler.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compiler.includeDirs.extend(self.includeDirs['executable'])
    for dir in self.usingSIDL.repositoryDirs:
      includeDir = self.usingSIDL.getClientRootDir(self.getLanguage(), root = dir)
      if os.path.isdir(includeDir):
        compiler.includeDirs.append(includeDir)
    return [self.getTagger(None), compiler]

  def getExecutableLinkTarget(self, project):
    libraries = fileset.FileSet()
    # Might be all internal clients
    if os.path.isfile(self.getClientLibrary(project, self.getLanguage())[0]):
      libraries.extend(self.getClientLibrary(project, self.getLanguage()))
    libraries.extend(self.extraLibraries['executable'])
    libraries.extend(self.usingSIDL.extraLibraries['executable'])
    for dir in self.usingSIDL.repositoryDirs:
      for lib in self.getClientLibrary(guessProject(dir), self.getLanguage(), isArchive = 0, root = os.path.join(dir, 'lib')):
        if os.path.isfile(lib):
          libraries.append(lib)
    return [link.TagLibrary(self.usingSIDL.sourceDB), link.LinkSharedLibrary(self.usingSIDL.sourceDB, extraLibraries = libraries)]

class UsingC (UsingCompiler):
  '''This class handles all interaction specific to the C language'''
  def __init__(self, usingSIDL):
    UsingCompiler.__init__(self, usingSIDL)

  def getLanguage(self):
    '''The language name'''
    return 'C'

  def getCompileSuffixes(self):
    '''The suffix for C files'''
    return ['.c', '.h']

  def getTagger(self, rootDir):
    return compile.TagC(self.usingSIDL.sourceDB, root = rootDir)

  def getCompiler(self, library):
    return compile.CompileC(self.usingSIDL.sourceDB, library)

  def getServerCompileTarget(self, project, package):
    rootDir = self.usingSIDL.getServerRootDir(self.getLanguage(), package)
    stubDir = self.usingSIDL.getStubDir(self.getLanguage(), package)
    library = self.getServerLibrary(project, self.getLanguage(), package)
    # IOR and server compile are both C
    compiler = compile.CompileC(self.usingSIDL.sourceDB, library)
    compiler.defines.extend(self.getDefines())
    compiler.includeDirs.append(rootDir)
    compiler.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    # Server specific flags
    compiler.includeDirs.append(stubDir)
    compiler.includeDirs.extend(self.includeDirs[package])
    compiler.includeDirs.extend(self.includeDirs[self.getLanguage()])
    for dir in self.usingSIDL.repositoryDirs:
      includeDir = self.usingSIDL.getClientRootDir(self.getLanguage(), root = dir)
      if os.path.isdir(includeDir):
        compiler.includeDirs.append(includeDir)
    return [compile.TagC(self.usingSIDL.sourceDB, root = rootDir), compiler]

class UsingCxx (UsingCompiler):
  '''This class handles all interaction specific to the C++ language'''
  def __init__(self, usingSIDL):
    UsingCompiler.__init__(self, usingSIDL)

  def getLanguage(self):
    '''The language name'''
    return 'C++'

  def getCompileSuffixes(self):
    '''The suffix for C++ files'''
    return ['.cc', '.hh']

  def getTagger(self, rootDir):
    return compile.TagCxx(self.usingSIDL.sourceDB, root = rootDir)

  def getCompiler(self, library):
    return compile.CompileCxx(self.usingSIDL.sourceDB, library)

  def getServerCompileTarget(self, project, package):
    rootDir = self.usingSIDL.getServerRootDir(self.getLanguage(), package)
    stubDir = self.usingSIDL.getStubDir(self.getLanguage(), package)
    library = self.getServerLibrary(project, self.getLanguage(), package)
    # IOR Filter
    iorFilter = transform.FileFilter(self.usingSIDL.compilerDefaults.isIOR, tags = ['c', 'old c'])
    # IOR compiler
    compileC = compile.CompileC(self.usingSIDL.sourceDB, library)
    compileC.defines.extend(self.getDefines())
    compileC.includeDirs.append(rootDir)
    compileC.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    # Server Filter
    serverFilter = transform.FileFilter(lambda source: self.usingSIDL.compilerDefaults.isServer(source, rootDir), tags = ['cxx', 'old cxx'])
    # Server compiler
    compileCxx = compile.CompileCxx(self.usingSIDL.sourceDB, library)
    compileCxx.defines.extend(self.getDefines())
    compileCxx.includeDirs.append(rootDir)
    compileCxx.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileCxx.includeDirs.append(stubDir)
    compileCxx.includeDirs.extend(self.includeDirs[package])
    compileCxx.includeDirs.extend(self.includeDirs[self.getLanguage()])
    for dir in self.usingSIDL.repositoryDirs:
      includeDir = self.usingSIDL.getClientRootDir(self.getLanguage(), root = dir)
      if os.path.isdir(includeDir):
        compileCxx.includeDirs.append(includeDir)
    targets = [compile.TagC(self.usingSIDL.sourceDB, root = rootDir), iorFilter, compileC,
               compile.TagCxx(self.usingSIDL.sourceDB, root = rootDir), serverFilter, compileCxx]
    return targets

class UsingPython(UsingCompiler):
  '''This class handles all interaction specific to the Python language'''
  def __init__(self, usingSIDL):
    UsingCompiler.__init__(self, usingSIDL)
    bs.argDB.setType('PYTHON_INCLUDE', nargs.ArgDir(1,'The directory containing Python.h'))
    #TODO: bs.argDB.setType('PYTHON_LIB', nargs.ArgLibrary(1, 'The library containing PyInitialize()'))
    self.setupIncludeDirectories()
    self.setupExtraLibraries()
    try:
      import Numeric
    except ImportError, e:
      raise RuntimeError("BS requires Numeric Python (http://www.pfdubois.com/numpy) to be installed: "+str(e))
    if not hasattr(sys,"version_info") or float(sys.version_info[0]) < 2 or float(sys.version_info[1]) < 2:
      raise RuntimeError("Requires Python version 2.2 or higher. Get Python at python.org")
    return

  def setupIncludeDirectories(self):
    try:
      if not bs.argDB.has_key('PYTHON_INCLUDE'):
        bs.argDB['PYTHON_INCLUDE'] = distutils.sysconfig.get_python_inc()
    except: pass
    includeDir = bs.argDB['PYTHON_INCLUDE']
    if isinstance(includeDir, list):
      self.includeDirs[self.getLanguage()].extend(includeDir)
    else:
      self.includeDirs[self.getLanguage()].append(includeDir)
    return self.includeDirs

  def setupExtraLibraries(self):
    try:
      if not bs.argDB.has_key('PYTHON_LIB'):
        lib = os.path.join(distutils.sysconfig.get_config_var('LIBPL'), distutils.sysconfig.get_config_var('LDLIBRARY'))
        # if .so was not built then need to strip .a off of end
        if lib[-2:] == '.a': lib = lib[0:-2]
        # may be stuff after .so like .0, so cannot use splitext()
        lib = lib.split('.so')[0]+'.so'
        bs.argDB['PYTHON_LIB'] = lib
    except: pass

    extraLibraries = [bs.argDB['PYTHON_LIB']]
    for lib in distutils.sysconfig.get_config_var('LIBS').split():
      # Change -l<lib> to lib<lib>.so
      extraLibraries.append('lib'+lib[2:]+'.so')
    self.extraLibraries[self.getLanguage()].extend(extraLibraries)
    for package in self.usingSIDL.getPackages():
      self.extraLibraries[package].extend(extraLibraries)
    return self.extraLibraries

  def getLanguage(self):
    '''The language name'''
    return 'Python'

  def getCompileSuffixes(self):
    '''The suffix for Python files'''
    return ['.py']

  def getTagger(self, rootDir):
    return compile.TagC(self.usingSIDL.sourceDB, root = rootDir)

  def getCompiler(self, library):
    return compile.CompilePythonC(self.usingSIDL.sourceDB)

  def getClientLibrary(self, project, lang, isArchive = 1, root = None):
    '''Need to return empty fileset for Python client library'''
    if lang == self.getLanguage():
      return fileset.FileSet()
    else:
      return UsingCompiler.getClientLibrary(self, project, lang, isArchive, root)

  def getServerCompileTarget(self, project, package):
    rootDir = self.usingSIDL.getServerRootDir(self.getLanguage(), package)
    stubDir = self.usingSIDL.getStubDir(self.getLanguage(), package)
    library = self.getServerLibrary(project, self.getLanguage(), package)
    # Server Filter
    serverFilter = transform.FileFilter(lambda source: self.usingSIDL.compilerDefaults.isServer(source, rootDir), tags = ['c', 'old c'])
    # IOR and server compile are both C
    compiler = compile.CompileC(self.usingSIDL.sourceDB, library)
    compiler.defines.extend(self.getDefines())
    compiler.includeDirs.append(rootDir)
    compiler.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    # Server specific flags
    compiler.includeDirs.append(stubDir)
    compiler.includeDirs.extend(self.includeDirs[package])
    compiler.includeDirs.extend(self.includeDirs[self.getLanguage()])
    for dir in self.usingSIDL.repositoryDirs:
      includeDir = self.usingSIDL.getClientRootDir(self.getLanguage(), root = dir)
      if os.path.isdir(includeDir):
        compiler.includeDirs.append(includeDir)
    return [compile.TagC(self.usingSIDL.sourceDB, root = rootDir), serverFilter, compiler]

  def getExecutableCompileTarget(self, project, sources, executable):
    return []

  def getExecutableLinkTarget(self, project):
    return []

class UsingMathematica(UsingCompiler):
  '''This class handles all interaction specific to the Mathematica language'''
  def __init__(self, usingSIDL):
    UsingCompiler.__init__(self, usingSIDL)
    bs.argDB.setType('MATHEMATICA_INCLUDE', nargs.ArgDir(1,'The directory containing mathlink.h'))
    #TODO: bs.argDB.setType('MATHEMATICA_LIB', nargs.ArgLibrary(1, 'The library containing MathOpenEnv()'))
    self.setupIncludeDirectories()
    self.setupExtraLibraries()
    return

  def setupIncludeDirectories(self):
    includeDir = bs.argDB['MATHEMATICA_INCLUDE']
    if isinstance(includeDir, list):
      self.includeDirs[self.getLanguage()].extend(includeDir)
    else:
      self.includeDirs[self.getLanguage()].append(includeDir)
    return self.includeDirs

  def setupExtraLibraries(self):
    for package in self.usingSIDL.getPackages():
      self.extraLibraries[package].append(bs.argDB['MATHEMATICA_LIB'])
    return self.extraLibraries

  def getLanguage(self):
    '''The language name'''
    return 'Mathematica'

  def getCompileSuffixes(self):
    '''The suffix for Mathematica files'''
    return ['.m', '.cc', '.hh']

  def getTagger(self, rootDir):
    return compile.TagCxx(self.usingSIDL.sourceDB, root = rootDir)

  def getCompiler(self, library):
    return compile.CompileCxx(self.usingSIDL.sourceDB, library)

  def getServerCompileTarget(self, project, package):
    rootDir = self.usingSIDL.getServerRootDir(self.getLanguage(), package)
    stubDir = self.usingSIDL.getStubDir(self.getLanguage(), package)
    library = self.getServerLibrary(project, self.getLanguage(), package)
    # IOR Filter
    iorFilter = transform.FileFilter(self.usingSIDL.compilerDefaults.isIOR, tags = ['c', 'old c'])
    # IOR compiler
    compileC = compile.CompileC(self.usingSIDL.sourceDB, library)
    compileC.defines.extend(self.getDefines())
    compileC.includeDirs.append(rootDir)
    compileC.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    # Server Filter
    serverFilter = transform.FileFilter(lambda source: self.usingSIDL.compilerDefaults.isServer(source, rootDir), tags = ['cxx', 'old cxx'])
    # Server compiler
    compileCxx = compile.CompileCxx(self.usingSIDL.sourceDB, library)
    compileCxx.defines.extend(self.getDefines())
    compileCxx.includeDirs.append(rootDir)
    compileCxx.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileCxx.includeDirs.append(stubDir)
    compileCxx.includeDirs.extend(self.includeDirs[package])
    compileCxx.includeDirs.extend(self.includeDirs[self.getLanguage()])
    for dir in self.usingSIDL.repositoryDirs:
      includeDir = self.usingSIDL.getClientRootDir(self.getLanguage(), root = dir)
      if os.path.isdir(includeDir):
        compileCxx.includeDirs.append(includeDir)
    targets = [compile.TagC(self.usingSIDL.sourceDB, root = rootDir), iorFilter, compileC,
               compile.TagCxx(self.usingSIDL.sourceDB, root = rootDir), serverFilter, compileCxx]
    return targets

class UsingF77 (UsingCompiler):
  '''This class handles all interaction specific to the Fortran 77 language'''
  def __init__(self, usingSIDL):
    UsingCompiler.__init__(self, usingSIDL)

  def getLanguage(self):
    '''The language name'''
    return 'F77'

  def getCompileSuffixes(self):
    '''The suffix for Fortran 77 files'''
    return ['.f', '.f90']

  def getTagger(self, rootDir):
    return compile.TagC(self.usingSIDL.sourceDB, root = rootDir)

  def getCompiler(self, library):
    return compile.CompileC(self.usingSIDL.sourceDB, library)

  def getServerCompileTarget(self, project, package):
    rootDir = self.usingSIDL.getServerRootDir(self.getLanguage(), package)
    stubDir = self.usingSIDL.getStubDir(self.getLanguage(), package)
    library = self.getServerLibrary(project, self.getLanguage(), package)
    # IOR compiler
    compileC = compile.CompileC(self.usingSIDL.sourceDB, library)
    compileC.defines.extend(self.getDefines())
    compileC.includeDirs.append(rootDir)
    compileC.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    # Server compiler
    compileF77 = compile.CompileF77(self.usingSIDL.sourceDB, library)
    compileF77.defines.extend(self.getDefines())
    compileF77.includeDirs.append(rootDir)
    compileF77.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileF77.includeDirs.append(stubDir)
    compileF77.includeDirs.extend(self.includeDirs[package])
    compileF77.includeDirs.extend(self.includeDirs[self.getLanguage()])
    return [compile.TagC(self.usingSIDL.sourceDB, root = rootDir), compileC, compile.TagF77(self.usingSIDL.sourceDB, root = rootDir), compileF77]

  def getExecutableCompileTarget(self, project, sources, executable):
    baseName = os.path.splitext(os.path.basename(executable[0]))[0] 
    library  = fileset.FileSet([os.path.join(project.getRoot(), 'lib', 'lib'+baseName+'.a')])
    compileC = compile.CompileC(self.usingSIDL.sourceDB, library)
    compileC.includeDirs.append(self.usingSIDL.getClientRootDir(self.getLanguage()))
    compileC.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileC.includeDirs.extend(self.includeDirs['executable'])
    return [compile.TagC(self.usingSIDL.sourceDB), compileC, compile.TagF77(self.usingSIDL.sourceDB), compile.CompileF77(self.usingSIDL.sourceDB, library)]

class UsingF90 (UsingCompiler):
  '''This class handles all interaction specific to the Fortran 90 language'''
  def __init__(self, usingSIDL):
    UsingCompiler.__init__(self, usingSIDL)
    #TODO: bs.argDB.setType('F90_LIB', nargs.ArgLibrary(1, 'The libraries containing F90 intrinsics'))
    self.setupExtraLibraries()
    return

  def setupExtraLibraries(self):
    if bs.argDB.has_key('F90_LIB'):
      extraLibraries = bs.argDB['F90_LIB']
      if not isinstance(extraLibraries, list):
        extraLibraries = [extraLibraries]
      self.extraLibraries[self.getLanguage()].extend(extraLibraries)
      for package in self.usingSIDL.getPackages():
        self.extraLibraries[package].extend(extraLibraries)
    return self.extraLibraries

  def getLanguage(self):
    '''The language name'''
    return 'F90'

  def getClientLanguages(self):
    '''Returns all languages involved in the client library'''
    return [self.getLanguage(), 'C++']

  def getCompileSuffixes(self):
    '''The suffix for Fortran 90 files'''
    return ['.f90', '.cc', '.hh']

  def getTagger(self, rootDir):
    return [compile.TagF90(self.usingSIDL.sourceDB, root = rootDir), compile.TagCxx(self.usingSIDL.sourceDB, root = rootDir)]

  def getCompiler(self, library):
    return [compile.CompileF90(self.usingSIDL.sourceDB, library), compile.CompileCxx(self.usingSIDL.sourceDB, library)]

  def getServerCompileTarget(self, project, package):
    rootDir = self.usingSIDL.getServerRootDir(self.getLanguage(), package)
    stubDir = self.usingSIDL.getStubDir(self.getLanguage(), package)
    library = self.getServerLibrary(project, self.getLanguage(), package)
    # IOR compiler
    compileC = compile.CompileC(self.usingSIDL.sourceDB, library)
    compileC.defines.extend(self.getDefines())
    compileC.includeDirs.append(rootDir)
    compileC.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    # Server compiler
    compilers = self.getCompiler(library)
    if not isinstance(compilers, list): compilers = [compilers]
    for compiler in compilers:
      compiler.defines.extend(self.getDefines())
      compiler.includeDirs.append(rootDir)
      compiler.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
      compiler.includeDirs.append(stubDir)
      compiler.includeDirs.extend(self.includeDirs[package])
      compiler.includeDirs.extend(self.includeDirs[self.getLanguage()])
    if len(compilers) == 1: compilers = compilers[0]
    return [compile.TagC(self.usingSIDL.sourceDB, root = rootDir), compileC, self.getTagger(rootDir), compilers]

  def getExecutableCompileTarget(self, project, sources, executable):
    baseName = os.path.splitext(os.path.basename(executable[0]))[0] 
    library  = fileset.FileSet([os.path.join(project.getRoot(), 'lib', 'lib'+baseName+'.a')])
    compileCxx = compile.CompileCxx(self.usingSIDL.sourceDB, library)
    compileCxx.includeDirs.append(self.usingSIDL.getClientRootDir(self.getLanguage()))
    compileCxx.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileCxx.includeDirs.extend(self.includeDirs['executable'])
    compileF90 = compile.CompileF90(self.usingSIDL.sourceDB, library)
    compileF90.includeDirs.append(self.usingSIDL.getClientRootDir(self.getLanguage()))
    compileF90.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileF90.includeDirs.extend(self.includeDirs['executable'])
    return [compile.TagCxx(self.usingSIDL.sourceDB), compileCxx, compile.TagF90(self.usingSIDL.sourceDB), compileF90]

class UsingJava (UsingCompiler):
  '''This class handles all interaction specific to the Java language'''
  def __init__(self, usingSIDL):
    UsingCompiler.__init__(self, usingSIDL)
    bs.argDB.setType('JAVA_INCLUDE', nargs.ArgDir(1, 'The directory containing jni.h'))
    #TODO: bs.argDB.setType('JAVA_RUNTIME_LIB', nargs.ArgLibrary(1, 'The library containing holders for Java builtin types'))
    self.setupIncludeDirectories()

  def setupIncludeDirectories(self):
    '''The directory containing jni.h'''
    includeDir = bs.argDB['JAVA_INCLUDE']
    if isinstance(includeDir, list):
      self.includeDirs[self.getLanguage()].extend(includeDir)
    else:
      self.includeDirs[self.getLanguage()].append(includeDir)
    return self.includeDirs

  def getLanguage(self):
    '''The language name'''
    return 'Java'

  def getCompileSuffixes(self):
    '''The suffix for Java files'''
    return ['.java']

  def getSIDLRuntimeLibraries(self):
    '''The SIDL runtime library for Java'''
    # Should be self.babelLibDir/sidl.jar
    runtimeLibs = bs.argDB['JAVA_RUNTIME_LIB']
    if not isinstance(runtimeLibs, list):
      runtimeLibs = [runtimeLibs]
    return runtimeLibs

  def getClientLibrary(self, project, lang, isArchive = 1, root = None, isJNI = 0):
    '''Client libraries following the naming scheme: lib<project>-<lang>-client.jar (or .a for JNI libraries)'''
    if not root: root = os.path.join(project.getRoot(), 'lib')
    if isJNI:
      if isArchive:
        ext = '.a'
      else:
        ext = '.so'
    else:
      ext = '.jar'
    return fileset.FileSet([os.path.join(root, 'lib'+project.getName()+'-'+lang.lower()+'-client'+ext)])

  def getServerLibrary(self, project, lang, isJNI = 0):
    raise RuntimeError('No server for '+self.getLanguage())

  def getClientCompileTarget(self, project):
    sourceDir = self.usingSIDL.getClientRootDir(self.getLanguage())
    compileC    = compile.CompileC(self.usingSIDL.sourceDB, self.getClientLibrary(project, self.getLanguage(), 1))
    compileJava = compile.CompileJava(self.usingSIDL.sourceDB, self.getClientLibrary(project, self.getLanguage()))
    compileC.defines.extend(self.getDefines())
    compileC.includeDirs.append(sourceDir)
    compileC.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileC.includeDirs.extend(self.includeDirs[self.getLanguage()])
    compileJava.includeDirs.extend(self.getSIDLRuntimeLibraries())
    compileJava.archiverRoot = sourceDir
    return [compile.TagC(self.usingSIDL.sourceDB, root = sourceDir), compile.TagJava(self.usingSIDL.sourceDB, root = sourceDir), compileC, compileJava]

  def getServerCompileTarget(self, project, package):
    raise RuntimeError('No server for '+self.getLanguage())

  def getExecutableCompileTarget(self, project, sources, executable):
    baseName = os.path.splitext(os.path.basename(executable[0]))[0] 
    library  = fileset.FileSet([os.path.join(project.getRoot(), 'lib', 'lib'+baseName+'.jar')])
    compileJava = compile.CompileJava(self.usingSIDL.sourceDB, library)
    compileJava.includeDirs.extend(self.getClientLibrary(project, self.getLanguage()))
    compileJava.includeDirs.extend(self.getSIDLRuntimeLibraries())
    compileJava.archiverRoot = os.path.dirname(sources[0])
    return [compile.TagJava(self.usingSIDL.sourceDB), compileJava]
  
#------------------------------------------------------------------------------------------------------
class UsingMatlab(UsingCompiler):
  '''This class handles all interaction specific to the Matlab language'''
  def __init__(self, usingSIDL):
    UsingCompiler.__init__(self, usingSIDL)
    if bs.argDB['MATLAB_DIR']:
      bs.argDB['MATLAB_INCLUDE'] = bs.argDB['MATLAB_DIR'] + '/extern/include'
      bs.argDB['MATLAB_LIB']     = [bs.argDB['MATLAB_DIR']+'/extern/lib/glnx86/libmat.a',
                                      bs.argDB['MATLAB_DIR']+'/extern/lib/glnx86/libmx.a',
                                      bs.argDB['MATLAB_DIR']+'/extern/lib/glnx86/libut.a',
                                      bs.argDB['MATLAB_DIR']+'/bin/glnx86/libmex.a']
    else:
      bs.argDB['MATLAB_INCLUDE'] = ''
      bs.argDB['MATLAB_LIB']     = ''

    self.setupIncludeDirectories()
    self.setupExtraLibraries()
    return

  def getCompiler(self, library):
    return compile.CompileMatlabCxx(self.usingSIDL.sourceDB)

  def setupIncludeDirectories(self):
    includeDir = bs.argDB['MATLAB_INCLUDE']
    if isinstance(includeDir, list):
      self.includeDirs[self.getLanguage()].extend(includeDir)
    else:
      self.includeDirs[self.getLanguage()].append(includeDir)
    return self.includeDirs

  def setupExtraLibraries(self):
    for package in self.usingSIDL.getPackages():
      self.extraLibraries[package].extend(bs.argDB['MATLAB_LIB'])
    return self.extraLibraries

  def getLanguage(self):
    '''The language name'''
    return 'Matlab'

  def getCompileSuffixes(self):
    '''The suffix for Matlab files'''
    return ['.mexglx', '.cc', '.hh']

  def getTagger(self, rootDir):
    return compile.TagCxx(self.usingSIDL.sourceDB, root = rootDir)

  def getCompiler(self, library):
    return compile.CompileCxx(self.usingSIDL.sourceDB, library)

  def getServerCompileTarget(self, project, package):
    rootDir = self.usingSIDL.getServerRootDir(self.getLanguage(), package)
    stubDir = self.usingSIDL.getStubDir(self.getLanguage(), package)
    library = self.getServerLibrary(project, self.getLanguage(), package)
    # IOR Filter
    iorFilter = transform.FileFilter(self.usingSIDL.compilerDefaults.isIOR, tags = ['c', 'old c'])
    # IOR compiler
    compileC = compile.CompileC(self.usingSIDL.sourceDB, library)
    compileC.defines.extend(self.getDefines())
    compileC.includeDirs.append(rootDir)
    compileC.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    # Server Filter
    serverFilter = transform.FileFilter(lambda source: self.usingSIDL.compilerDefaults.isServer(source, rootDir), tags = ['cxx', 'old cxx'])
    # Server compiler
    compileCxx = compile.CompileCxx(self.usingSIDL.sourceDB, library)
    compileCxx.defines.extend(self.getDefines())
    compileCxx.includeDirs.append(rootDir)
    compileCxx.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileCxx.includeDirs.append(stubDir)
    compileCxx.includeDirs.extend(self.includeDirs[package])
    compileCxx.includeDirs.extend(self.includeDirs[self.getLanguage()])
    for dir in self.usingSIDL.repositoryDirs:
      includeDir = self.usingSIDL.getClientRootDir(self.getLanguage(), root = dir)
      if os.path.isdir(includeDir):
        compileCxx.includeDirs.append(includeDir)
    targets = [compile.TagC(self.usingSIDL.sourceDB, root = rootDir), iorFilter, compileC,
               compile.TagCxx(self.usingSIDL.sourceDB, root = rootDir), serverFilter, compileCxx]
    return targets

