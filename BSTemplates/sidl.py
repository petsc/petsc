import argtest
import babel
import bk
import bs
import compile
import fileset
import link
import target
import transform

import os
import re
import types

class SIDLConstants:
  def getLanguages():
    return ['C', 'C++', 'Python', 'F77', 'Java']

  getLanguages = staticmethod(getLanguages)

class SIDLLanguageList (list, SIDLConstants):
  def __setitem__(self, key, value):
    if not value in self.getLanguages():
      raise ValueError('Invalid SIDL language: '+value)
    self.data[key] = value

class SIDLPackageList (list):
  def __init__(self, defaults):
    list.__init__(self)
    self.defaults = defaults

  def __setitem__(self, key, value):
    self.checkPackage(value)
    self.data[key] = value

  def checkPackage(self, package):
    if not package in self.defaults.getPackages():
      raise KeyError('Invalid SIDL package: '+package)

class SIDLPackageDict (dict, SIDLConstants):
  '''We now allow packages or languages as keys'''
  def __init__(self, defaults):
    dict.__init__(self)
    self.defaults = defaults

  def checkPackage(self, package):
    if not package in self.defaults.getPackages():
      if package in self.getLanguages(): return
      if package == 'executable': return
      raise KeyError('Invalid SIDL package: '+package)

  def __getitem__(self, key):
    self.checkPackage(key)
    if not self.has_key(key): dict.__setitem__(self, key, [])
    return dict.__getitem__(self, key)

  def __setitem__(self, key, value):
    self.checkPackage(key)
    if not type(value) == types.ListType: raise ValueError('Entries must be lists')
    dict.__setitem__(self, key, value)

class UsingSIDL (SIDLConstants):
  '''This class handles all interaction specific to the SIDL language'''
  def __init__(self, project, packages, repositoryDir = None, serverBaseDir = None):
    self.project  = project
    self.packages = packages
    # The directory where XML is stored
    if repositoryDir:
      self.repositoryDir = repositoryDir
    else:
      self.repositoryDir = os.path.abspath('xml')
    # The base path for generated server source
    if serverBaseDir:
      self.serverBaseDir = serverBaseDir
    else:
      self.serverBaseDir = os.path.abspath('server')
    # The repositories used during compilation
    self.repositoryDirs  = []
    self.clientLanguages = SIDLLanguageList()
    self.serverLanguages = SIDLLanguageList()
    # Languages in which the client must be linked with the server
    self.internalClientLanguages = SIDLPackageDict(self)
    # Packages which must be compiled before the clients
    self.bootstrapPackages = SIDLPackageList(self)
    # Flags for the SIDL compiler
    self.compilerFlags  = ''
    self.includeDirs    = SIDLPackageDict(self)
    self.extraLibraries = SIDLPackageDict(self)
    self.libDir         = os.path.join(self.getRootDir(), 'lib')
    self.setupIncludeDirectories()
    self.setupExtraLibraries()

  def setupIncludeDirectories(self):
    rootDir    = self.getRootDir()
    includeDir = os.path.join(rootDir, 'server-sidl')
    for lang in self.getLanguages():
      self.includeDirs[lang].append(includeDir)
    # TODO: Fix this debacle by generating SIDLObjA and SIDLPyArrays
    self.includeDirs['Python'].append(os.path.join(rootDir, 'python'))
    return self.includeDirs

  def setupExtraLibraries(self):
    runtimeLib = self.getServerLibrary('sidlruntime', self.getBaseLanguage(), self.getBasePackage(), 0)
    self.extraLibraries['executable'].extend(runtimeLib)
    for lang in self.getLanguages():
      self.extraLibraries[lang].extend(runtimeLib)
    for package in self.getPackages():
      if not self.project == 'sidlruntime' or not package in self.bootstrapPackages:
        self.extraLibraries[package].extend(runtimeLib)
    return self.extraLibraries

  def getLanguage(self):
    '''The language name'''
    return 'SIDL'

  def getCompileSuffixes(self):
    '''The suffix for SIDL files'''
    return ['.sidl']

  def getBasePackage(self):
    '''The package which contains the runtime support, usually \'sidl\''''
    return 'sidl'

  def getBaseLanguage(self):
    '''The implementation language for the SIDL runtime library, usually C'''
    return 'C'

  def getPackages(self):
    return self.packages

  def getCompilerFlags(self):
    return self.compilerFlags

  def setCompilerFlags(self, flags):
    self.compilerFlags = flags
    return

  def getRootDir(self):
    return os.path.abspath(bs.argDB['BABEL_DIR'])

  def getServerRootDir(self, lang, package = None):
    path = self.serverBaseDir
    if len(self.serverLanguages) > 1:
      path += '-'+lang.lower()
    if package:
      path += '-'+package
    return path

  def getClientRootDir(self, lang):
    return os.path.abspath(lang.lower())

  def getStubDir(self, lang, package):
    if lang in self.internalClientLanguages[package]:
      return self.getServerRootDir(lang, package)
    elif lang in self.clientLanguages:
      return self.getClientRootDir(lang)
    else:
      raise RuntimeError('Package '+package+' needs stubs for '+lang+' which have not been configured')

  def getServerLibrary(self, project, lang, package, isArchive = 1):
    '''Server libraries following the naming scheme:
      lib<project>-<lang>-<package>-server.a    for archives
      lib<project>-<lang>-<package>-server.so   for dynamic libraries'''
    if isArchive:
      return fileset.FileSet([os.path.join(self.libDir, 'lib'+project+'-'+lang.lower()+'-'+package+'-server.a')])
    else:
      return fileset.FileSet([os.path.join(self.libDir, 'lib'+project+'-'+lang.lower()+'-'+package+'-server.so')])

class UsingCompiler:
  '''This class handles all interaction specific to a compiled language'''
  def __init__(self, usingSIDL):
    self.usingSIDL      = usingSIDL
    self.defines        = ['PIC']
    self.includeDirs    = SIDLPackageDict(usingSIDL)
    self.extraLibraries = SIDLPackageDict(usingSIDL)
    self.libDir         = os.path.abspath('lib')

  def getDefines(self):
    return self.defines

  def getClientLibrary(self, project, lang):
    '''Client libraries following the naming scheme: lib<project>-<lang>-client.a'''
    return fileset.FileSet([os.path.join(self.libDir, 'lib'+project+'-'+lang.lower()+'-client.a')])

  def getServerLibrary(self, project, lang, package):
    '''Server libraries following the naming scheme: lib<project>-<lang>-<package>-server.a'''
    return fileset.FileSet([os.path.join(self.libDir, 'lib'+project+'-'+lang.lower()+'-'+package+'-server.a')])

  def getClientCompileTarget(self, project):
    sourceDir = self.usingSIDL.getClientRootDir(self.getLanguage())
    compiler  = self.getCompiler(self.getClientLibrary(project, self.getLanguage()))
    compiler.defines.extend(self.getDefines())
    compiler.includeDirs.append(sourceDir)
    compiler.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compiler.includeDirs.extend(self.includeDirs[self.getLanguage()])
    return [self.getTagger(sourceDir), compiler]

  def getClientLinkTarget(self, project, doLibraryCheck = 1):
    libraries = fileset.FileSet([])
    libraries.extend(self.usingSIDL.extraLibraries[self.getLanguage()])
    libraries.extend(self.extraLibraries[self.getLanguage()])
    linker    = link.LinkSharedLibrary(extraLibraries = libraries)
    linker.doLibraryCheck = doLibraryCheck
    return [link.TagLibrary(), linker]

  def getServerLinkTarget(self, project, package, doLibraryCheck = 1):
    libraries = fileset.FileSet([])
    libraries.extend(self.usingSIDL.extraLibraries[package])
    libraries.extend(self.getClientLibrary(project, self.getLanguage()))
    libraries.extend(self.extraLibraries[package])
    linker    = link.LinkSharedLibrary(extraLibraries = libraries)
    linker.doLibraryCheck = doLibraryCheck
    return [link.TagLibrary(), linker]

  def getExecutableCompileTarget(self, project, sources, executable):
    baseName = os.path.splitext(os.path.basename(executable[0]))[0] 
    library  = fileset.FileSet([os.path.join(self.libDir, 'lib'+baseName+'.a')])
    compiler = self.getCompiler(library)
    compiler.includeDirs.append(self.usingSIDL.getClientRootDir(self.getLanguage()))
    compiler.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compiler.includeDirs.extend(self.includeDirs['executable'])
    return [self.getTagger(None), compiler]

  def getExecutableLinkTarget(self, project):
    libraries = fileset.FileSet()
    libraries.extend(self.getClientLibrary(project, self.getLanguage()))
    libraries.extend(self.extraLibraries['executable'])
    libraries.extend(self.usingSIDL.extraLibraries['executable'])
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
    return ['.h', '.c']

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
    return ['.hh', '.cc']

  def getTagger(self, rootDir):
    return compile.TagCxx(root = rootDir)

  def getCompiler(self, library):
    return compile.CompileCxx(library)

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
    compileCxx = compile.CompileCxx(library)
    compileCxx.defines.extend(self.getDefines())
    compileCxx.includeDirs.append(rootDir)
    compileCxx.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileCxx.includeDirs.append(stubDir)
    compileCxx.includeDirs.extend(self.includeDirs[package])
    compileCxx.includeDirs.extend(self.includeDirs[self.getLanguage()])
    return [compile.TagC(root = rootDir), compileC, compile.TagCxx(root = rootDir), compileCxx]

class UsingPython(UsingCompiler):
  '''This class handles all interaction specific to the Python language'''
  def __init__(self, usingSIDL):
    UsingCompiler.__init__(self, usingSIDL)
    bs.argDB.setTester('PYTHON_INCLUDE', argtest.DirectoryTester())
    #TODO: bs.argDB.setTester('PYTHON_LIB',     argtest.LibraryTester())
    self.setupIncludeDirectories()
    self.setupExtraLibraries()

  def setupIncludeDirectories(self):
    includeDir = bs.argDB['PYTHON_INCLUDE']
    if isinstance(includeDir, list):
      self.includeDirs[self.getLanguage()].extend(includeDir)
    else:
      self.includeDirs[self.getLanguage()].append(includeDir)
    return self.includeDirs

  def setupExtraLibraries(self):
    for package in self.usingSIDL.getPackages():
      self.extraLibraries[package].extend([bs.argDB['PYTHON_LIB'], 'libpthread.so', 'libutil.so'])
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

  def getClientLibrary(self, project, lang):
    '''Need to return empty fileset for Python client library'''
    if lang == self.getLanguage():
      return fileset.FileSet()
    else:
      return UsingCompiler.getClientLibrary(self, project, lang)

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
    return [compile.TagC(root = rootDir), compiler]

  def getExecutableCompileTarget(self, project, sources, executable):
    raise RuntimeError('No executable compilation in '+self.getLanguage())

  def getExecutableLinkTarget(self, project):
    raise RuntimeError('No executable link in '+self.getLanguage())

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
    library  = fileset.FileSet([os.path.join(self.libDir, 'lib'+baseName+'.a')])
    compileC = compile.CompileC(library)
    compileC.includeDirs.append(self.usingSIDL.getClientRootDir(self.getLanguage()))
    compileC.includeDirs.extend(self.usingSIDL.includeDirs[self.getLanguage()])
    compileC.includeDirs.extend(self.includeDirs['executable'])
    return [compile.TagC(), compileC, compile.TagF77(), compile.CompileF77(library)]

class UsingJava (UsingCompiler):
  '''This class handles all interaction specific to the Java language'''
  def __init__(self, usingSIDL):
    UsingCompiler.__init__(self, usingSIDL)
    bs.argDB.setTester('JAVA_INCLUDE', argtest.DirectoryTester())
    bs.argDB.setTester('JAVA_RUNTIME_LIB', argtest.DirectoryTester())
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

  def getClientLibrary(self, project, lang, isJNI = 0):
    '''Client libraries following the naming scheme: lib<project>-<lang>-client.jar'''
    if isJNI:
      libraryName = 'lib'+project+'-'+lang.lower()+'-client.a'
    else:
      libraryName = 'lib'+project+'-'+lang.lower()+'-client.jar'
    return fileset.FileSet([os.path.join(self.libDir, libraryName)])

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
    library  = fileset.FileSet([os.path.join(self.libDir, 'lib'+baseName+'.jar')])
    compileJava = compile.CompileJava(library)
    compileJava.includeDirs.extend(self.getClientLibrary(project, self.getLanguage()))
    compileJava.includeDirs.extend(self.getSIDLRuntimeLibraries())
    compileJava.archiverRoot = os.path.dirname(sources[0])
    return [compile.TagJava(), compileJava]

class Defaults:
  implRE     = re.compile(r'^(.*)_Impl$')
  libraryRE  = re.compile(r'^(.*)lib(.*).so$')

  def __init__(self, project, sources = None):
    self.project    = project
    self.sources    = sources
    self.usingSIDL  = UsingSIDL(project, self.getPackages())
    self.compileExt = []
    # Add C for the IOR
    self.addLanguage('C')

  def getUsing(self, lang):
    return getattr(self, 'using'+lang.replace('+', 'x'))

  def addLanguage(self, lang):
    try:
      self.getUsing(lang.replace('+', 'x'))
    except AttributeError:
      lang = lang.replace('+', 'x')
      opt  = globals()['Using'+lang](self.usingSIDL)
      setattr(self, 'using'+lang, opt)
      self.compileExt.extend(opt.getCompileSuffixes())
    return

  def addClientLanguage(self, lang):
    self.usingSIDL.clientLanguages.append(lang)
    self.addLanguage(lang)

  def addServerLanguage(self, lang):
    self.usingSIDL.serverLanguages.append(lang)
    self.addLanguage(lang)

  def isImpl(self, source):
    if os.path.splitext(source)[1] == '.pyc':      return 0
    if self.implRE.match(os.path.dirname(source)): return 1
    return 0

  def isNotLibrary(self, source):
    if self.libraryRE.match(source): return 0
    return 1

  def isNewSidl(self, sources):
    if isinstance(sources, fileset.FileSet):
      if sources.tag == 'sidl' and len(sources) > 0:
        return 1
      else:
        return 0
    elif type(sources) == types.ListType:
      isNew = 0
      for source in sources:
        isNew = isNew or self.isNewSidl(source)
      return isNew
    else:
      raise RuntimeError('Invalid type for sources: '+type(sources))

  def getPackages(self):
    if self.sources:
      sources = self.sources
    else:
      sources = []
    return map(lambda file: os.path.splitext(os.path.split(file)[1])[0], sources)

  def getRepositoryTargets(self):
    action = babel.CompileSIDLRepository(compilerFlags = self.usingSIDL.getCompilerFlags())
    action.outputDir = self.usingSIDL.repositoryDir
    action.repositoryDirs.extend(self.usingSIDL.repositoryDirs)
    return [target.Target(None, [babel.TagAllSIDL(), action])]

  def getSIDLServerCompiler(self, lang, rootDir, generatedRoots):
    action = babel.CompileSIDLServer(fileset.ExtensionFileSet(generatedRoots, self.compileExt), compilerFlags = self.usingSIDL.getCompilerFlags())
    action.language  = lang
    action.outputDir = rootDir
    action.repositoryDirs.append(self.usingSIDL.repositoryDir)
    action.repositoryDirs.extend(self.usingSIDL.repositoryDirs)
    return action

  def getSIDLServerTargets(self):
    targets = []
    for lang in self.usingSIDL.serverLanguages:
      serverSourceRoots = fileset.FileSet(map(lambda package, lang=lang, self=self: self.usingSIDL.getServerRootDir(lang, package), self.getPackages()))
      for rootDir in serverSourceRoots:
        if not os.path.isdir(rootDir):
          os.makedirs(rootDir)

      genActions = [bk.TagBKOpen(roots = serverSourceRoots),
                    bk.BKOpen(),
                    # CompileSIDLServer() will add the package automatically to the output directory
                    self.getSIDLServerCompiler(lang, self.usingSIDL.getServerRootDir(lang), serverSourceRoots),
                    bk.TagBKClose(roots = serverSourceRoots),
                    transform.FileFilter(self.isImpl, tags = 'bkadd'),
                    bk.BKClose()]

      defActions = transform.Transform(fileset.ExtensionFileSet(serverSourceRoots, self.compileExt))

      targets.append(target.Target(None, [babel.TagSIDL(), target.If(self.isNewSidl, genActions, defActions)]))
    return targets

  def getSIDLClientCompiler(self, lang, rootDir):
    compiler           = babel.CompileSIDLClient(fileset.ExtensionFileSet(rootDir, self.compileExt), compilerFlags = self.usingSIDL.getCompilerFlags())
    compiler.language  = lang
    compiler.outputDir = rootDir
    compiler.repositoryDirs.append(self.usingSIDL.repositoryDir)
    compiler.repositoryDirs.extend(self.usingSIDL.repositoryDirs)
    return compiler

  def getSIDLClientTargets(self):
    targets = []
    for lang in self.usingSIDL.clientLanguages:
      targets.append(target.Target(None, [babel.TagAllSIDL(), self.getSIDLClientCompiler(lang, self.usingSIDL.getClientRootDir(lang))]))
    # Some clients have to be linked with the corresponding server (like the Bable bootstrap)
    for package in self.getPackages():
      for lang in self.usingSIDL.internalClientLanguages[package]:
        targets.append(target.Target(None, [babel.TagAllSIDL(), self.getSIDLClientCompiler(lang, self.usingSIDL.getServerRootDir(lang, package))]))
    return targets

  def getSIDLTarget(self):
    return target.Target(self.sources, [tuple(self.getRepositoryTargets()+self.getSIDLServerTargets()+self.getSIDLClientTargets()),
                                        transform.Update(),
                                        transform.SetFilter('old sidl')])

class CompileDefaults (Defaults):
  def __init__(self, project, sidlSources, etagsFile = None):
    Defaults.__init__(self, project, sidlSources)
    self.etagsFile = etagsFile

  def getClientCompileTargets(self, doCompile = 1, doLink = 1):
    targets  = []
    for lang in self.usingSIDL.clientLanguages:
      compiler = []
      linker   = []
      try:
        if doCompile: compiler = self.getUsing(lang).getClientCompileTarget(self.project)
        if doLink:    linker   = self.getUsing(lang).getClientLinkTarget(self.project, not self.project == 'bs')
      except AttributeError:
        raise RuntimeError('Unknown client language: '+lang)

      targets.append(target.Target(None, compiler + linker))
    # Could update after each client
    targets.append(transform.Update())
    return targets

  def getServerCompileTargets(self, doCompile = 1, doLink = 1):
    targets          = []
    bootstrapTargets = []

    for lang in self.usingSIDL.serverLanguages:
      for package in self.getPackages():
        compiler = []
        linker   = []
        try:
          if doCompile: compiler = self.getUsing(lang).getServerCompileTarget(self.project, package)
          if doLink:    linker   = self.getUsing(lang).getServerLinkTarget(self.project, package, not self.project == 'bs')
          t = target.Target(None, compiler + linker + [transform.Update()])
        except AttributeError:
          raise RuntimeError('Unknown server language: '+lang)

        if package in self.usingSIDL.bootstrapPackages:
          bootstrapTargets.append(t)
        else:
          targets.append(t)

    return [bootstrapTargets, targets]

  def getEmacsTagsTargets(self):
    return [transform.FileFilter(self.isImpl), compile.TagEtags(), compile.CompileEtags(self.etagsFile)]

  def getCompileTargets(self, doCompile = 1, doLink = 1):
    serverTargets  = self.getServerCompileTargets(doCompile, doLink)
    compileTargets = serverTargets[0]+self.getClientCompileTargets(doCompile, doLink)+serverTargets[1]

    if self.etagsFile:
      return [(compileTargets, self.getEmacsTagsTargets()), transform.Update()]
    else:
      return compileTargets+[transform.Update()]

  def getCompileTarget(self):
    return target.Target(None, [self.getSIDLTarget()]+self.getCompileTargets(1, 1))

  def getExecutableDriverTargets(self, sources, lang, executable):
    try:
      compiler = self.getUsing(lang).getExecutableCompileTarget(self.project, sources, executable)
      linker   = self.getUsing(lang).getExecutableLinkTarget(self.project)
    except AttributeError, e:
      import sys
      import traceback
        
      print str(e)
      print traceback.print_tb(sys.exc_info()[2])
      raise RuntimeError('Unknown executable language: '+lang)
    return compiler+linker+[transform.Update()]

  def getExecutableTarget(self, lang, sources, executable):
    # TODO: Of course this should be determined from configure
    libraries = fileset.FileSet(['libdl.so'])

    t = [self.getCompileTarget(), transform.FileFilter(self.isNotLibrary)] + self.getExecutableDriverTargets(sources, lang, executable)
    if not lang == 'Java':
      t += [link.TagShared(), link.LinkExecutable(executable, extraLibraries = libraries)]
    return target.Target(sources, t)
