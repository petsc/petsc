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
    tag          = self.getLanguage().lower().replace('+', 'x')
    clientFilter = transform.FileFilter(lambda source: self.usingSIDL.compilerDefaults.isClient(source, sourceDir), tags = [tag, 'old '+tag])
    # Client compiler
    compiler  = self.getCompiler(self.getClientLibrary(project, self.getLanguage()))
    compiler.defines.extend(self.getDefines())
    compiler.includeDirs.append(sourceDir)
    compiler.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compiler.includeDirs.extend(self.includeDirs[self.getLanguage()])
    for dir in self.usingSIDL.repositoryDirs:
      includeDir = self.usingSIDL.getClientRootDir(self.getLanguage(), root = dir)
      if os.path.isdir(includeDir):
        compiler.includeDirs.append(includeDir)
    return [self.getTagger(sourceDir), clientFilter, compiler]

  def getClientLinkTarget(self, project, doLibraryCheck = 1):
    libraries = fileset.FileSet([])
    libraries.extend(self.usingSIDL.extraLibraries[self.getLanguage()])
    libraries.extend(self.extraLibraries[self.getLanguage()])
    for dir in self.usingSIDL.repositoryDirs:
      for lib in self.getClientLibrary(guessProject(dir), self.getLanguage(), isArchive = 0, root = os.path.join(dir, 'lib')):
        if os.path.isfile(lib):
          libraries.append(lib)
    linker    = link.LinkSharedLibrary(extraLibraries = libraries)
    linker.doLibraryCheck = doLibraryCheck
    return [link.TagLibrary(), linker]

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
    linker    = link.LinkSharedLibrary(extraLibraries = libraries)
    linker.doLibraryCheck = doLibraryCheck
    return [link.TagLibrary(), linker]

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
    return [link.TagLibrary(), link.LinkSharedLibrary(extraLibraries = libraries)]

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
    return compile.TagC(root = rootDir)

  def getCompiler(self, library):
    return compile.CompileC(library)

  def getServerCompileTarget(self, project, package):
    rootDir = self.usingSIDL.getServerRootDir(self.getLanguage(), package)
    stubDir = self.usingSIDL.getStubDir(self.getLanguage(), package)
    library = self.getServerLibrary(project, self.getLanguage(), package)
    # IOR and server compile are both C
    compiler = compile.CompileC(library)
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
    return [compile.TagC(root = rootDir), compiler]

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
    return compile.TagCxx(root = rootDir)

  def getCompiler(self, library):
    return compile.CompileCxx(library)

  def getServerCompileTarget(self, project, package):
    rootDir = self.usingSIDL.getServerRootDir(self.getLanguage(), package)
    stubDir = self.usingSIDL.getStubDir(self.getLanguage(), package)
    library = self.getServerLibrary(project, self.getLanguage(), package)
    # IOR Filter
    iorFilter = transform.FileFilter(self.usingSIDL.compilerDefaults.isIOR, tags = ['c', 'old c'])
    # IOR compiler
    compileC = compile.CompileC(library)
    compileC.defines.extend(self.getDefines())
    compileC.includeDirs.append(rootDir)
    compileC.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    # Server Filter
    serverFilter = transform.FileFilter(lambda source: self.usingSIDL.compilerDefaults.isServer(source, rootDir), tags = ['cxx', 'old cxx'])
    # Server compiler
    compileCxx = compile.CompileCxx(library)
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
    targets = [compile.TagC(root = rootDir), iorFilter, compileC, compile.TagCxx(root = rootDir), serverFilter, compileCxx]
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
    # This is not quite right
    for package in self.usingSIDL.getPackages():
      self.extraLibraries[package].extend([bs.argDB['PYTHON_LIB'], 'libpthread.so', 'libutil.so', 'libdl.so'])
    return self.extraLibraries

  def getLanguage(self):
    '''The language name'''
    return 'Python'

  def getCompileSuffixes(self):
    '''The suffix for Python files'''
    return ['.py']

  def getTagger(self, rootDir):
    return compile.TagC(root = rootDir)

  def getCompiler(self, library):
    return compile.CompilePythonC()

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
    compiler = compile.CompileC(library)
    compiler.defines.extend(self.getDefines())
    compiler.includeDirs.append(rootDir)
    compiler.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    # Server specific flags
    compiler.includeDirs.append(stubDir)
    compiler.includeDirs.extend(self.includeDirs[package])
    compiler.includeDirs.extend(self.includeDirs[self.getLanguage()])
    return [compile.TagC(root = rootDir), serverFilter, compiler]

  def getExecutableCompileTarget(self, project, sources, executable):
    return []

  def getExecutableLinkTarget(self, project):
    return []

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
    return compile.TagC(root = rootDir)

  def getCompiler(self, library):
    return compile.CompileC(library)

  def getServerCompileTarget(self, project, package):
    rootDir = self.usingSIDL.getServerRootDir(self.getLanguage(), package)
    stubDir = self.usingSIDL.getStubDir(self.getLanguage(), package)
    library = self.getServerLibrary(project, self.getLanguage(), package)
    # IOR compiler
    compileC = compile.CompileC(library)
    compileC.defines.extend(self.getDefines())
    compileC.includeDirs.append(rootDir)
    compileC.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    # Server compiler
    compileF77 = compile.CompileF77(library)
    compileF77.defines.extend(self.getDefines())
    compileF77.includeDirs.append(rootDir)
    compileF77.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileF77.includeDirs.append(stubDir)
    compileF77.includeDirs.extend(self.includeDirs[package])
    compileF77.includeDirs.extend(self.includeDirs[self.getLanguage()])
    return [compile.TagC(root = rootDir), compileC, compile.TagF77(root = rootDir), compileF77]

  def getExecutableCompileTarget(self, project, sources, executable):
    baseName = os.path.splitext(os.path.basename(executable[0]))[0] 
    library  = fileset.FileSet([os.path.join(project.getRoot(), 'lib', 'lib'+baseName+'.a')])
    compileC = compile.CompileC(library)
    compileC.includeDirs.append(self.usingSIDL.getClientRootDir(self.getLanguage()))
    compileC.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileC.includeDirs.extend(self.includeDirs['executable'])
    return [compile.TagC(), compileC, compile.TagF77(), compile.CompileF77(library)]

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
    compileC    = compile.CompileC(self.getClientLibrary(project, self.getLanguage(), 1))
    compileJava = compile.CompileJava(self.getClientLibrary(project, self.getLanguage()))
    compileC.defines.extend(self.getDefines())
    compileC.includeDirs.append(sourceDir)
    compileC.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileC.includeDirs.extend(self.includeDirs[self.getLanguage()])
    compileJava.includeDirs.extend(self.getSIDLRuntimeLibraries())
    compileJava.archiverRoot = sourceDir
    return [compile.TagC(root = sourceDir), compile.TagJava(root = sourceDir), compileC, compileJava]

  def getServerCompileTarget(self, project, package):
    raise RuntimeError('No server for '+self.getLanguage())

  def getExecutableCompileTarget(self, project, sources, executable):
    baseName = os.path.splitext(os.path.basename(executable[0]))[0] 
    library  = fileset.FileSet([os.path.join(project.getRoot(), 'lib', 'lib'+baseName+'.jar')])
    compileJava = compile.CompileJava(library)
    compileJava.includeDirs.extend(self.getClientLibrary(project, self.getLanguage()))
    compileJava.includeDirs.extend(self.getSIDLRuntimeLibraries())
    compileJava.archiverRoot = os.path.dirname(sources[0])
    return [compile.TagJava(), compileJava]
