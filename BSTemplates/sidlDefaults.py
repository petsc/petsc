import bs
import fileset
import logging
import transform
import BSTemplates.compileDefaults as compileDefaults
import BSTemplates.sidlStructs as sidlStructs

import os
import re

class UsingSIDL (logging.Logger):
  '''This class handles all interaction specific to the SIDL language'''
  def __init__(self, project, packages, repositoryDir = None, serverBaseDir = None, bootstrapPackages = []):
    self.project  = project
    self.packages = packages
    rootDir       = project.getRoot()
    # The repository root directory
    if repositoryDir:
      self.repositoryDir = repositoryDir
    else:
      self.repositoryDir = rootDir
    # The base path for generated server source
    if serverBaseDir:
      self.serverBaseDir = serverBaseDir
    else:
      self.serverBaseDir = os.path.join(rootDir, 'server')
    # The repositories used during compilation
    self.repositoryDirs  = []
    self.clientLanguages = sidlStructs.SIDLLanguageList()
    self.serverLanguages = sidlStructs.SIDLLanguageList()
    # Languages in which the client must be linked with the server
    self.internalClientLanguages = sidlStructs.SIDLPackageDict(self)
    # Packages which must be compiled before the clients
    self.bootstrapPackages = sidlStructs.SIDLPackageList(self)
    self.bootstrapPackages.extend(bootstrapPackages)
    # Setup compiler specific defaults
    if bs.argDB.has_key('babelCrap') and int(bs.argDB['babelCrap']):
      self.debugPrint('Compiling SIDL with Babel', 3, 'sidl')
      import BSTemplates.babelTargets
      self.compilerDefaults = BSTemplates.babelTargets.Defaults(self)
    else:
      self.debugPrint('Compiling SIDL with Scandal', 3, 'sidl')
      import BSTemplates.scandalTargets
      self.compilerDefaults = BSTemplates.scandalTargets.Defaults(self)
    # Flags for the SIDL compiler
    self.serverCompilerFlags = sidlStructs.SIDLLanguageDict(self)
    self.clientCompilerFlags = sidlStructs.SIDLLanguageDict(self)
    self.includeDirs         = sidlStructs.SIDLPackageDict(self)
    self.extraLibraries      = sidlStructs.SIDLPackageDict(self)
    self.setupIncludeDirectories()
    self.setupExtraLibraries()

  def setupIncludeDirectories(self):
    rootDir = self.getRuntimeProject().getRoot()
    for lang in sidlStructs.SIDLConstants.getLanguages():
      self.includeDirs[lang].append(self.getServerRootDir(self.getBaseLanguage(), self.getBasePackage(), root = os.path.join(rootDir, 'server')))
      if not self.compilerDefaults.generatesAllStubs() and not lang == self.getBaseLanguage():
        self.includeDirs[lang].append(self.getClientRootDir(lang, root = rootDir))
    # TODO: Fix this debacle by generating SIDLObjA and SIDLPyArrays
    self.includeDirs['Python'].append(os.path.join(rootDir, 'python'))
    return self.includeDirs

  def getRuntimeProject(self):
    for project in bs.argDB['installedprojects']+[self.project]:
      if project.getName() == 'sidlruntime':
        return project
    return bs.Project('sidlruntime', 'bk://sidl.bkbits.net/')

  def setupExtraLibraries(self):
    runtimeProject = self.getRuntimeProject()
    using     = getattr(compileDefaults, 'Using'+self.getBaseLanguage().replace('+', 'x'))(self)
    serverLib = using.getServerLibrary(runtimeProject, self.getBaseLanguage(), self.getBasePackage(), isArchive = 0)
    self.extraLibraries['executable'].extend(serverLib)
    for lang in sidlStructs.SIDLConstants.getLanguages():
      self.extraLibraries[lang].extend(serverLib)
      if not self.project == runtimeProject and not lang == self.getBaseLanguage():
        using = getattr(compileDefaults, 'Using'+lang.replace('+', 'x'))(self)
        self.extraLibraries[lang].extend(using.getClientLibrary(runtimeProject, lang, isArchive = 0))
    for package in self.getPackages():
      if not self.project == runtimeProject or not package in self.bootstrapPackages:
        self.extraLibraries[package].extend(serverLib)
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
    if bs.argDB.has_key('SIDL_LANG'):
      return bs.argDB['SIDL_LANG']
    else:
      return 'C'

  def getPackages(self):
    return self.packages

  def getServerCompilerFlags(self, language):
    return self.serverCompilerFlags[language]

  def getClientCompilerFlags(self, language):
    return self.clientCompilerFlags[language]

  def getCompilerFlags(self, language):
    flags = ''
    if self.serverCompilerFlags[language]:
      if flags: flags += ' '
      flags += self.serverCompilerFlags[language]
    if self.clientCompilerFlags[language]:
      if flags: flags += ' '
      flags += self.clientCompilerFlags[language]
    return flags

  def setServerCompilerFlags(self, flags, language = '', notLanguage = ''):
    if language:
      self.serverCompilerFlags[language] = flags
    else:
      for language in sidlStructs.SIDLConstants.getLanguages():
        if notLanguage and language == notLanguage: continue
        self.serverCompilerFlags[language] = flags
    return

  def setClientCompilerFlags(self, flags, language = '', notLanguage = ''):
    if language:
      self.clientCompilerFlags[language] = flags
    else:
      for language in sidlStructs.SIDLConstants.getLanguages():
        if notLanguage and language == notLanguage: continue
        self.clientCompilerFlags[language] = flags
    return

  def setCompilerFlags(self, flags, language = '', notLanguage = ''):
    self.setServerCompilerFlags(flags, language, notLanguage)
    self.setClientCompilerFlags(flags, language, notLanguage)
    return

  def getServerRootDir(self, lang, package = None, root = None):
    '''Returns an absolute path if root is given, otherwise a relative path'''
    if not root: root = self.serverBaseDir
    dir = self.compilerDefaults.getServerRootDir(lang, package, root)
    return dir

  def getClientRootDir(self, lang, root = None):
    '''Always returns an absolute path'''
    dir = self.compilerDefaults.getClientRootDir(lang)
    if root: dir = os.path.join(root, dir)
    return os.path.abspath(dir)

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
    library = os.path.join(project.getRoot(), 'lib', 'lib'+project.getName()+'-'+lang.lower()+'-'+package+'-server')
    if isArchive:
      library += '.a'
    else:
      library += '.so'
    return fileset.FileSet([library])

class TagSIDL (transform.GenericTag):
  def __init__(self, tag = 'sidl', ext = 'sidl', sources = None, extraExt = ''):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)

class TagAllSIDL (transform.GenericTag):
  def __init__(self, tag = 'sidl', ext = 'sidl', sources = None, extraExt = '', force = 0):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)
    self.taggedFiles = fileset.FileSet()
    self.force       = force

  def fileExecute(self, source):
    (base, ext) = os.path.splitext(source)
    if ext in self.ext:
      self.taggedFiles.append(source)
    transform.GenericTag.fileExecute(self, source)

  def execute(self):
    self.genericExecute(self.sources)
    if len(self.changed) or self.force:
      ## This is bad, should have a clear()
      self.changed.data   = []
      self.changed.extend(self.taggedFiles)
      self.unchanged.data = []
    return self.products
