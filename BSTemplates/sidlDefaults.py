import bs
import fileset
import transform

import os
import re

class SIDLConstants:
  def getLanguages():
    return ['C', 'C++', 'Python', 'F77', 'Java']
  getLanguages = staticmethod(getLanguages)

  def checkLanguage(language):
    if not language in SIDLConstants.getLanguages():
      raise ValueError('Invalid SIDL language: '+language)
  checkLanguage = staticmethod(checkLanguage)

class SIDLLanguageList (list):
  def __setitem__(self, key, value):
    SIDLConstants.checkLanguage(value)
    self.data[key] = value

class SIDLPackages:
  '''We now allow packages or languages as keys'''
  def __init__(self, defaults):
    self.defaults = defaults

  def getPackages(self):
    return self.defaults.getPackages()

  def checkPackage(self, package):
    if not package in self.getPackages():
      if package in SIDLConstants.getLanguages(): return
      if package == 'executable': return
      raise KeyError('Invalid SIDL package: '+package)

class SIDLPackageList (list, SIDLPackages):
  '''We now allow packages or languages as keys'''
  def __init__(self, defaults):
    list.__init__(self)
    SIDLPackages.__init__(self, defaults)

  def __setitem__(self, key, value):
    self.checkPackage(value)
    self.data[key] = value

class SIDLPackageDict (dict, SIDLPackages):
  '''We now allow packages or languages as keys'''
  def __init__(self, defaults):
    dict.__init__(self)
    SIDLPackages.__init__(self, defaults)

  def __getitem__(self, key):
    self.checkPackage(key)
    if not self.has_key(key): dict.__setitem__(self, key, [])
    return dict.__getitem__(self, key)

  def __setitem__(self, key, value):
    self.checkPackage(key)
    if not type(value) == types.ListType: raise ValueError('Entries must be lists')
    dict.__setitem__(self, key, value)

class UsingSIDL:
  '''This class handles all interaction specific to the SIDL language'''
  def __init__(self, project, packages, repositoryDir = None, serverBaseDir = None, bootstrapPackages = []):
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
    self.bootstrapPackages.extend(bootstrapPackages)
    # Flags for the SIDL compiler
    self.compilerFlags  = ''
    self.includeDirs    = SIDLPackageDict(self)
    self.extraLibraries = SIDLPackageDict(self)
    self.libDir         = os.path.join(self.getRootDir(), 'lib')
    self.setupIncludeDirectories()
    self.setupExtraLibraries()

  def setupIncludeDirectories(self):
    rootDir    = self.getRootDir()
    includeDir = os.path.join(rootDir, 'server-'+self.getBaseLanguage().lower()+'-'+self.getBasePackage())
    for lang in SIDLConstants.getLanguages():
      self.includeDirs[lang].append(includeDir)
    # TODO: Fix this debacle by generating SIDLObjA and SIDLPyArrays
    self.includeDirs['Python'].append(os.path.join(rootDir, 'python'))
    return self.includeDirs

  def setupExtraLibraries(self):
    runtimeLib = self.getServerLibrary('sidlruntime', self.getBaseLanguage(), self.getBasePackage(), 0)
    self.extraLibraries['executable'].extend(runtimeLib)
    for lang in SIDLConstants.getLanguages():
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
    if bs.argDB.has_key('SIDL_LANG'):
      return bs.argDB['SIDL_LANG']
    else:
      return 'C'

  def getPackages(self):
    return self.packages

  def getCompilerFlags(self):
    return self.compilerFlags

  def setCompilerFlags(self, flags):
    self.compilerFlags = flags
    return

  def getRootDir(self):
    return os.path.abspath(bs.argDB['SIDL_DIR'])

  def getServerRootDir(self, lang, package = None):
    path  = self.serverBaseDir
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
