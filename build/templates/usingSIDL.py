import base
import project

import os

class UsingSIDL (base.Base):
  def __init__(self, sourceDB, project):
    base.Base.__init__(self)
    import build.compile.SIDL
    self.sourceDB        = sourceDB
    self.project         = project
    self.serverLanguages = build.compile.SIDL.SIDLLanguageList()
    self.clientLanguages = build.compile.SIDL.SIDLLanguageList()
    # Languages in which the client must be linked with the server
    self.staticPackages  = []
    return

  def addServer(self, lang):
    '''Designate that a server for lang should be built, which also implies the client'''
    if lang in self.argDB['installedLanguages']:
      if not lang in self.serverLanguages:
        self.serverLanguages.append(lang)
      self.addClient(lang)
    else:
      self.debugPrint('Language '+lang+' not installed', 2, 'compile')
    return

  def addClient(self, lang):
    '''Designate that a client for lang should be built'''
    if lang in self.argDB['installedLanguages']:
      if not lang in self.clientLanguages:
        self.clientLanguages.append(lang)
    else:
      self.debugPrint('Language '+lang+' not installed', 2, 'compile')
    return

  def addStaticPackage(self, package):
    '''For a static package, the client is statically linked to the server since dynamic loading is not feasible'''
    self.staticPackages.append(package)
    return

  def getServerRootDir(self, lang, package):
    '''Returns a server directory name'''
    return 'server-'+lang.lower()+'-'+package

  def getClientRootDir(self, lang, root = None):
    '''Returns a client directory name'''
    return 'client-'+lang.lower()

  def getServerLibrary(self, projectName, lang, package, isShared = 0):
    '''Server libraries follow the naming scheme: lib<project>-<lang>-<package>-server.a'''
    if isShared:
      ext = '.so'
    else:
      ext = '.a'
    return os.path.join('lib', 'lib'+projectName+'-'+lang.lower()+'-'+package+'-server'+ext)

  def getRuntimeLanguage(self):
    '''Return the implementation language for the runtime'''
    return 'Cxx'

  def getRuntimePackage(self):
    '''Return the implementation package for the runtime'''
    return 'sidl'

  def getRuntimeProject(self):
    '''Return the project associated with the SIDL Runtime'''
    projects = [self.project]
    if 'installedprojects' in self.argDB:
      projects += self.argDB['installedprojects']
    for project in projects:
      if project.getUrl() == 'bk://sidl.bkbits.net/Runtime':
        return project
    raise ImportError('Could not find runtime project')

  def getRuntimeIncludes(self):
    '''Return the includes for the SIDL Runtime'''
    proj = self.getRuntimeProject()
    return [project.ProjectPath(self.getServerRootDir(self.getRuntimeLanguage(), self.getRuntimePackage()), proj.getUrl())]

  def getRuntimeLibraries(self):
    '''Return the libraries for the SIDL Runtime'''
    proj = self.getRuntimeProject()
    return [project.ProjectPath(self.getServerLibrary(proj.getName(), self.getRuntimeLanguage(), self.getRuntimePackage()), proj.getUrl())]

  def getClassesInFile(path):
    '''Return all the classes present in the SIDL file'''
    try:
      import SIDL.Loader
      import SIDLLanguage.Parser
      import ANL.SIDL.ClassFinder

      parser = SIDLLanguage.Parser.Parser(SIDL.Loader.createClass('ANL.SIDLCompilerI.SIDLCompiler'))
      ast    = parser.parseFile(path)
      finder = ANL.SIDL.ClassFinder.ClassFinder()
      ast.accept(finder)
      return [c.getFullIdentifier() for c in finder.getClasses()]
    except: pass
    return []
  getClassesInFile = staticmethod(getClassesInFile)

  def getClasses(self, package):
    '''Return all the classes present in the SIDL file for "package"'''
    return UsingSIDL.getClassesInFile(os.path.join('sidl', package+'.sidl'))

