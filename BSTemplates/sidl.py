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
import string
import types
import UserDict
import UserList

class BabelLanguageList (UserList.UserList):
  languages = ['C', 'C++', 'Python', 'F77']

  def __setitem__(self, key, value):
    if not value in self.languages:
      raise ValueError('Invalid Babel langague: '+value)
    self.data[key] = value

class BabelPackageList (list):
  def __init__(self, defaults):
    list.__init__(self)
    self.defaults = defaults

  def __setitem__(self, key, value):
    self.checkPackage(value)
    self.data[key] = value

  def checkPackage(self, package):
    if not package in self.defaults.getPackages():
      raise KeyError('Invalid Babel package: '+package)

class BabelPackageDict (UserDict.UserDict):
  '''We now allow packages or languages as keys'''
  def __init__(self, defaults):
    UserDict.UserDict.__init__(self)
    self.defaults = defaults

  def checkPackage(self, package):
    if not package in self.defaults.getPackages():
      if package == 'executable': return
      if package == 'C': return
      if package == 'C++': return
      if package == 'Python': return
      if package == 'F77': return
      raise KeyError('Invalid Babel package: '+package)

  def __getitem__(self, key):
    self.checkPackage(key)
    if not self.data.has_key(key): self.data[key] = []
    return self.data[key]

  def __setitem__(self, key, value):
    self.checkPackage(key)
    if not type(value) == types.ListType: raise ValueError('Entries must be lists')
    self.data[key] = value

class Defaults:
  implRE     = re.compile(r'^(.*)_Impl$')
  libraryRE  = re.compile(r'^(.*)lib(.*).so$')
  compileExt = ['.h', '.c', '.hh', '.cc', '.f', '.f90']

  def __init__(self, sources = None, repositoryDir = None, serverBaseDir = None, compilerFlags = ''):
    self.sources         = sources
    if repositoryDir:
      self.repositoryDir = repositoryDir
    else:
      self.repositoryDir = os.path.abspath('xml')
    if serverBaseDir:
      self.serverBaseDir = serverBaseDir
    else:
      self.serverBaseDir = os.path.abspath('server')
    self.compilerFlags   = compilerFlags
    self.repositoryDirs  = []
    self.clientLanguages = BabelLanguageList()
    self.serverLanguages = BabelLanguageList()
    self.internalClientLanguages = BabelPackageDict(self)
    self.bootstrapPackages       = BabelPackageList(self)

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
      raise RuntimeError('Inalid type for sources: '+type(sources))

  def getServerRootDir(self, lang, package):
    path = self.serverBaseDir
    if len(self.serverLanguages) > 1:
      path += '-'+string.lower(lang)
    if package:
      path += '-'+package
    return path

  def getPackages(self):
    return map(lambda file: os.path.splitext(os.path.split(file)[1])[0], self.sources)

  def getClientRootDir(self, lang):
    return os.path.abspath(string.lower(lang))

  def getRepositoryTargets(self):
    action = babel.CompileSIDLRepository(compilerFlags = self.compilerFlags)
    action.outputDir = self.repositoryDir
    action.repositoryDirs.extend(self.repositoryDirs)
    return [target.Target(None, [babel.TagAllSIDL(), action])]

  def getSIDLServerCompiler(self, lang, rootDir, generatedRoots):
    action = babel.CompileSIDLServer(fileset.ExtensionFileSet(generatedRoots, self.compileExt), compilerFlags = self.compilerFlags)
    action.language  = lang
    action.outputDir = rootDir
    action.repositoryDirs.append(self.repositoryDir)
    action.repositoryDirs.extend(self.repositoryDirs)
    return action

  def getSIDLServerTargets(self):
    targets = []
    for lang in self.serverLanguages:
      serverSourceRoots = fileset.FileSet(map(lambda package, lang=lang, self=self: self.getServerRootDir(lang, package), self.getPackages()))
      for rootDir in serverSourceRoots:
        if not os.path.isdir(rootDir):
          os.makedirs(rootDir)

      genActions = [bk.TagBKOpen(roots = serverSourceRoots),
                    bk.BKOpen(),
                    # CompileSIDLServer() will add the package automatically to the output directory
                    self.getSIDLServerCompiler(lang, self.getServerRootDir(lang, ''), serverSourceRoots),
                    bk.TagBKClose(roots = serverSourceRoots),
                    transform.FileFilter(self.isImpl, tags = 'bkadd'),
                    bk.BKClose()]

      defActions = transform.Transform(fileset.ExtensionFileSet(serverSourceRoots, self.compileExt))

      targets.append(target.Target(None, [babel.TagSIDL(), target.If(self.isNewSidl, genActions, defActions)]))
    return targets

  def getSIDLClientCompiler(self, lang, rootDir):
    compiler           = babel.CompileSIDLClient(fileset.ExtensionFileSet(rootDir, self.compileExt), compilerFlags = self.compilerFlags)
    compiler.language  = lang
    compiler.outputDir = rootDir
    compiler.repositoryDirs.append(self.repositoryDir)
    compiler.repositoryDirs.extend(self.repositoryDirs)
    return compiler

  def getSIDLClientTargets(self):
    targets = []
    for lang in self.clientLanguages:
      targets.append(target.Target(None, [babel.TagAllSIDL(), self.getSIDLClientCompiler(lang, self.getClientRootDir(lang))]))
    # Some clients have to be linked with the corresponding server (like the Bable bootstrap)
    for package in self.getPackages():
      for lang in self.internalClientLanguages[package]:
        targets.append(target.Target(None, [babel.TagAllSIDL(), self.getSIDLClientCompiler(lang, self.getServerRootDir(lang, package))]))
    return targets

  def getSIDLTarget(self):
    return target.Target(self.sources, [tuple(self.getRepositoryTargets()+self.getSIDLServerTargets()+self.getSIDLClientTargets()),
                                        transform.Update(),
                                        transform.SetFilter('old sidl')])

class CompileDefaults (Defaults):
  def __init__(self, project, sidlSources, compilerFlags = ''):
    Defaults.__init__(self, sidlSources, compilerFlags = compilerFlags)
    self.project               = project
    self.libDir                = os.path.abspath('lib')
    self.pythonIncludeDir      = bs.argDB['PYTHON_INCLUDE']
    self.pythonLib             = fileset.FileSet([bs.argDB['PYTHON_LIB']])
    self.babelDir              = os.path.abspath(bs.argDB['BABEL_DIR'])
    #self.babelIncludeDir       = os.path.join(self.babelDir, 'include')
    self.babelIncludeDir       = os.path.join(self.babelDir, 'server-sidl')
    self.babelLibDir           = os.path.join(self.babelDir, 'lib')
    #self.babelLib              = fileset.FileSet([os.path.join(self.babelLibDir, 'libsidl.so')])
    self.babelLib              = fileset.FileSet([os.path.join(self.babelLibDir, 'libsidlruntime-c-sidl-server.so')])
    self.babelPythonIncludeDir = os.path.join(self.babelDir, 'python')
    self.includeDirs           = BabelPackageDict(self)
    self.extraLibraries        = BabelPackageDict(self)
    self.etagsFile             = None

  def addBabelInclude(self, compiler, lang = None):
    compiler.includeDirs.append(self.babelIncludeDir)
    if lang == 'Python':
      compiler.includeDirs.append(self.babelPythonIncludeDir)
    return compiler

  def addBabelLib(self, libraries, package = ''):
    if self.project == 'sidlruntime' and package in self.bootstrapPackages: return libraries
    libraries.extend(self.babelLib)
    return libraries

  def getClientLibrary(self, lang):
    'Client libraries following the naming scheme: lib<project>-<lang>-client.a'
    return fileset.FileSet([os.path.join(self.libDir, 'lib'+self.project+'-'+string.lower(lang)+'-client.a')])

  def getServerLibrary(self, lang, package):
    'Server libraries following the naming scheme: lib<project>-<lang>-<package>-server.a'
    return fileset.FileSet([os.path.join(self.libDir, 'lib'+self.project+'-'+string.lower(lang)+'-'+package+'-server.a')])

  def getServerCompileTargets(self, segregateBootstrapTargets):
    targets          = []
    bootstrapTargets = []

    for lang in self.serverLanguages:
      for package in self.getPackages():
        actions   = []
        rootDir   = self.getServerRootDir(lang, package)
        library   = self.getServerLibrary(lang, package)
        libraries = fileset.FileSet()

        if lang in self.internalClientLanguages[package]:
          stubDir = rootDir
        elif lang in self.clientLanguages:
          stubDir = self.getClientRootDir(lang)
        else:
          raise RuntimeError('Package '+package+' needs stubs for '+lang+' which have not been configured')

        if lang in ['Python', 'C']:
          tagger = compile.TagC(root = rootDir)
        elif lang == 'C++':
          tagger = [compile.TagC(root = rootDir), compile.TagCxx(root = rootDir)]
        elif lang == 'F77':
          tagger = [compile.TagC(root = rootDir), compile.TagF77(root = rootDir)]
        else:
          raise RuntimeError('Unknown client language: '+lang)

        # For IOR source
        iorAction = compile.CompileC(library)
        iorAction.defines.append('PIC')
        iorAction.includeDirs.append(rootDir)
        self.addBabelInclude(iorAction)
        actions.append(iorAction)

        # For skeleton and implementation source
        if lang == 'C' or lang == 'Python':
          iorAction.includeDirs.append(stubDir)
          self.addBabelInclude(iorAction)
          if self.includeDirs.has_key(package):
            iorAction.includeDirs.extend(self.includeDirs[package])
          if lang == 'Python':
            iorAction.includeDirs.append(self.pythonIncludeDir)
            # TODO: Fix this debacle by generating SIDLObjA and SIDLPyArrays
            iorAction.includeDirs.append(os.path.join(self.babelDir, 'python'))
        else:
          if lang == 'C++':
            implAction = compile.CompileCxx(library)
          elif lang == 'F77':
            implAction = compile.CompileF77(library)
          else:
            raise RuntimeError('Language '+lang+' not supported as a server')
          implAction.defines.append('PIC')
          implAction.includeDirs.append(rootDir)
          if stubDir:
            implAction.includeDirs.append(stubDir)
          self.addBabelInclude(implAction)
          if self.includeDirs.has_key(package):
            implAction.includeDirs.extend(self.includeDirs[package])
          actions.append(implAction)

        if lang in self.clientLanguages:
          libraries.extend(self.getClientLibrary(lang))
        self.addBabelLib(libraries, package)
        if self.extraLibraries.has_key(package):
          libraries.extend(self.extraLibraries[package])
        if lang == 'Python':
          libraries.extend(self.pythonLib)
          libraries.append('libpthread.so')
          libraries.append('libutil.so')

        # Allow bootstrap
        linker = link.LinkSharedLibrary(extraLibraries = libraries)
        if self.project == 'bs':
          linker.doLibraryCheck = 0

        t = target.Target(None,
                          [tagger,
                           actions,
                           link.TagLibrary(),
                           linker,
                           transform.Update()])
        if segregateBootstrapTargets and package in self.bootstrapPackages:
          bootstrapTargets.append(t)
        else:
          targets.append(t)

    if segregateBootstrapTargets:
      return [bootstrapTargets, targets]
    else:
      return targets

  def getClientCompileTargets(self):
    targets = []
    for lang in self.clientLanguages:
      sourceDir = self.getClientRootDir(lang)
      library   = self.getClientLibrary(lang)
      libraries = fileset.FileSet()

      if lang in ['Python', 'F77', 'C']:
        tagger = compile.TagC(root = sourceDir)
        action = compile.CompileC(library)
      elif lang == 'C++':
        tagger = [compile.TagC(root = sourceDir), compile.TagCxx(root = sourceDir)]
        action = compile.CompileCxx(library)
      else:
        raise RuntimeError('Unknown client language: '+lang)

      action.defines.append('PIC')
      action.includeDirs.append(sourceDir)
      self.addBabelInclude(action, lang)
      if self.includeDirs.has_key(lang):
        action.includeDirs.extend(self.includeDirs[lang])

      self.addBabelLib(libraries)
      if self.extraLibraries.has_key(lang):
        libraries.extend(self.extraLibraries[lang])

      if lang == 'Python':
        action.includeDirs.append(self.pythonIncludeDir)
        action = (babel.PythonModuleFixup(library, sourceDir), action)

      # Allow bootstrap
      linker = link.LinkSharedLibrary(extraLibraries = libraries)
      if self.project == 'bs':
        linker.doLibraryCheck = 0

      targets.append(target.Target(None,
                                   [tagger,
                                    action,
                                    link.TagLibrary(),
                                    linker]))
    targets.append(transform.Update())
    return targets

  def getEmacsTagsTargets(self):
    return [transform.FileFilter(self.isImpl), compile.TagEtags(), compile.CompileEtags(self.etagsFile)]

  def getCompileTarget(self):
    serverTargets  = self.getServerCompileTargets(1)
    compileTargets = serverTargets[0]+self.getClientCompileTargets()+serverTargets[1]

    if self.etagsFile:
      return target.Target(None, [self.getSIDLTarget(),
                                  (compileTargets, self.getEmacsTagsTargets()),
                                  transform.Update()])
    else:
      return target.Target(None, [self.getSIDLTarget()]+compileTargets)

  def getExecutableCompileTargets(self, lang, executable):
    baseName  = os.path.splitext(os.path.split(executable[0])[1])[0] 
    library   = fileset.FileSet([os.path.join(self.libDir, 'lib'+baseName+'.a')])
    libraries = fileset.FileSet()
    tags      = []
    actions   = []

    if lang == 'C':
      tags.append(compile.TagC())
      action = compile.CompileC(library)
    elif lang == 'C++':
      tags.append(compile.TagCxx())
      action = compile.CompileCxx(library)

    self.addBabelInclude(action)
    action.includeDirs.append(self.getClientRootDir(lang))
    if self.includeDirs.has_key('executable'):
      action.includeDirs.extend(self.includeDirs['executable'])
    actions.append(action)
    libraries.extend(self.getClientLibrary(lang))

    return [target.Target(None,
                         [tags,
                          actions,
                          link.TagLibrary(),
                          link.LinkSharedLibrary(extraLibraries = self.addBabelLib(libraries))])]

  def getExecutableTarget(self, lang, sources, executable):
    libraries = fileset.FileSet([])
    if self.extraLibraries.has_key('executable'):
      libraries.extend(self.extraLibraries['executable'])
    # TODO: Of course this should be determined from configure
    libraries.append('libdl.so')

    return target.Target(sources,
                         [self.getCompileTarget(),
                          transform.FileFilter(self.isNotLibrary)]+
                         self.getExecutableCompileTargets(lang, executable)+
                         [link.TagShared(),
                          link.LinkExecutable(executable, extraLibraries = libraries)])
