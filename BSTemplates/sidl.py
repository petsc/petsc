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

class BabelPackageDict (UserDict.UserDict):
  def __init__(self, defaults):
    UserDict.UserDict.__init__(self)
    self.defaults = defaults

  def checkPackage(self, package):
    if not package in self.defaults.getPackages():
      if package == 'executable': return
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
  implRE   = re.compile(r'^(.*)_Impl$')
  clientRE = re.compile(r'^(.*)lib(.*)client.so$')

  def __init__(self, sources = None, repositoryDir = None, serverBaseDir = None):
    self.sources         = sources
    if repositoryDir:
      self.repositoryDir = repositoryDir
    else:
      self.repositoryDir = os.path.abspath('xml')
    if serverBaseDir:
      self.serverBaseDir = serverBaseDir
    else:
      self.serverBaseDir = os.path.abspath('server')
    self.repositoryDirs  = []
    self.clientLanguages = BabelLanguageList()
    self.serverLanguages = BabelLanguageList()

  def isImpl(self, source):
    if self.implRE.match(os.path.dirname(source)): return 1
    return 0

  def isNotClientLibrary(self, source):
    if self.clientRE.match(source): return 0
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
    action = babel.CompileSIDLRepository()
    action.outputDir = self.repositoryDir
    action.repositoryDirs.extend(self.repositoryDirs)
    return [target.Target(None, [babel.TagAllSIDL(), action])]

  def getServerSIDLTargets(self):
    targets           = []
    for lang in self.serverLanguages:
      serverSourceRoots = fileset.FileSet(map(lambda package, lang=lang, self=self: self.getServerRootDir(lang, package), self.getPackages()))
      for rootDir in serverSourceRoots:
        if not os.path.isdir(rootDir):
          os.makedirs(rootDir)

      action = babel.CompileSIDLServer(fileset.ExtensionFileSet(serverSourceRoots, ['.h', '.c', '.hh', '.cc']))
      action.language  = lang
      action.outputDir = self.getServerRootDir(lang, '')
      action.repositoryDirs.append(self.repositoryDir)
      action.repositoryDirs.extend(self.repositoryDirs)

      genActions = [bk.TagBKOpen(roots = serverSourceRoots),
                    bk.BKOpen(),
                    action,
                    bk.TagBKClose(roots = serverSourceRoots),
                    transform.FileFilter(self.isImpl, tags = 'bkadd'),
                    bk.BKClose()]
      defActions = transform.Transform(fileset.ExtensionFileSet(serverSourceRoots, ['.h', '.c', '.hh', '.cc']))

      targets.append(target.Target(None, [babel.TagSIDL(), target.If(self.isNewSidl, genActions, defActions)]))
    return targets

  def getClientSIDLTargets(self):
    targets = []
    for lang in self.clientLanguages:
      rootDir = self.getClientRootDir(lang)
      action  = babel.CompileSIDLClient(fileset.ExtensionFileSet(rootDir, ['.h', '.c', '.cc', '.hh']))
      action.language  = lang
      action.outputDir = rootDir
      action.repositoryDirs.append(self.repositoryDir)
      action.repositoryDirs.extend(self.repositoryDirs)
      targets.append(target.Target(None, [babel.TagAllSIDL(), action]))
    return targets

  def getSIDLTarget(self):
    return target.Target(self.sources, [tuple(self.getRepositoryTargets()+self.getServerSIDLTargets()+self.getClientSIDLTargets()),
                                        transform.Update(),
                                        transform.SetFilter('old sidl')])

class CompileDefaults (Defaults):
  def __init__(self, project, sidlSources):
    Defaults.__init__(self, sidlSources)
    self.project               = project
    self.libDir                = os.path.abspath('lib')
    self.babelDir              = os.path.abspath(bs.argDB['BABEL_DIR'])
    self.babelIncludeDir       = os.path.join(self.babelDir, 'include')
    self.babelLibDir           = os.path.join(self.babelDir, 'lib')
    self.babelLib              = fileset.FileSet([os.path.join(self.babelLibDir, 'libsidl.so')])
    self.babelPythonIncludeDir = os.path.join(self.babelDir, 'python')
    self.includeDirs           = BabelPackageDict(self)
    self.extraLibraries        = BabelPackageDict(self)
    self.etagsFile             = None

  def getClientLibrary(self, lang):
    'Client libraries following the naming scheme: lib<project>-<lang>-client.a'
    return fileset.FileSet([os.path.join(self.libDir, 'lib'+self.project+'-'+string.lower(lang)+'-client.a')])

  def getServerLibrary(self, lang, package):
    'Server libraries following the naming scheme: lib<project>-<lang>-<package>-server.a'
    return fileset.FileSet([os.path.join(self.libDir, 'lib'+self.project+'-'+string.lower(lang)+'-'+package+'-server.a')])

  def getServerCompileTargets(self):
    targets = []
    for lang in self.serverLanguages:
      stubDir = self.getClientRootDir(lang)
      for package in self.getPackages():
        rootDir   = self.getServerRootDir(lang, package)
        library   = self.getServerLibrary(lang, package)
        libraries = fileset.FileSet(children = [self.getClientLibrary(lang), self.babelLib])

        # For IOR source
        cAction = compile.CompileC(library)
        cAction.defines.append('PIC')
        cAction.includeDirs.append(rootDir)
        cAction.includeDirs.append(self.babelIncludeDir)

        # For skeleton and implementation source
        cxxAction = compile.CompileCxx(library)
        cxxAction.defines.append('PIC')
        cxxAction.includeDirs.append(rootDir)
        cxxAction.includeDirs.append(stubDir)
        cxxAction.includeDirs.append(self.babelIncludeDir)

        if self.includeDirs.has_key(package):
          cxxAction.includeDirs.extend(self.includeDirs[package])
        if self.extraLibraries.has_key(package):
          libraries.extend(self.extraLibraries[package])

        # Allow bootstrap
        linker = link.LinkSharedLibrary(extraLibraries = self.babelLib)
        if self.project == 'bs':
          linker.doLibraryCheck = 0

        targets.append(target.Target(None,
                                     [compile.TagC(root = rootDir),
                                      compile.TagCxx(root = rootDir),
                                      cAction,
                                      cxxAction,
                                      link.TagLibrary(),
                                      linker]))
    targets.append(transform.Update())
    return targets

  def getClientCompileTargets(self):
    targets = []
    for lang in self.clientLanguages:
      sourceDir = self.getClientRootDir(lang)
      library   = self.getClientLibrary(lang)

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
      action.includeDirs.append(self.babelIncludeDir)

      if lang == 'Python':
        action.includeDirs.append(self.babelPythonIncludeDir)
        action.includeDirs.append(bs.argDB['PYTHON_INCLUDE'])
        action = (babel.PythonModuleFixup(library, sourceDir), action)

      # Allow bootstrap
      linker = link.LinkSharedLibrary(extraLibraries = self.babelLib)
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
    if self.etagsFile:
      return target.Target(None, [self.getSIDLTarget(),
                                  (self.getClientCompileTargets()+self.getServerCompileTargets(), self.getEmacsTagsTargets()),
                                  transform.Update()])
    else:
      return target.Target(None, [self.getSIDLTarget()]+self.getClientCompileTargets()+self.getServerCompileTargets())

  def getExecutableCompileTargets(self, executable):
    baseName  = os.path.splitext(os.path.split(executable[0])[1])[0] 
    library   = fileset.FileSet([os.path.join(self.libDir, 'lib'+baseName+'.a')])
    libraries = fileset.FileSet(children = [self.babelLib])

    action = compile.CompileCxx(library)
    action.includeDirs.append(self.babelIncludeDir)
    action.includeDirs.append(self.getClientRootDir('C++'))
    if self.includeDirs.has_key('executable'):
      action.includeDirs.extend(self.includeDirs['executable'])

    return [target.Target(None,
                         [compile.TagCxx(),
                          action,
                          link.TagLibrary(),
                          link.LinkSharedLibrary(extraLibraries = libraries)])]

  def getExecutableTarget(self, sources, executable):
    libraries = fileset.FileSet([])
    if self.extraLibraries.has_key('executable'):
      libraries.extend(self.extraLibraries['executable'])

    return target.Target(sources,
                         [self.getCompileTarget()]+self.getExecutableCompileTargets(executable)+
                         [transform.FileFilter(self.isNotClientLibrary),
                          link.TagShared(),
                          link.LinkExecutable(executable, extraLibraries = libraries)])
