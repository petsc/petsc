#!/usr/bin/env python
import nargs
import importer
import install.base
import sourceDatabase
import BSTemplates.sidlTargets
import BSTemplates.compileTargets

import atexit
import cPickle
import os
import sys
import traceback
import re

class BS (install.base.Base):
  targets     = {}
  batchArgs   = 0
  directories = {}
  filesets    = {}

  def __init__(self, project, clArgs = None, argDB = None):
    install.base.Base.__init__(self, self.setupArgDB(clArgs, argDB))
    self.project          = project
    self.sourceDBFilename = os.path.join(self.project.getRoot(), 'bsSource.db')
    self.setupSourceDB()
    return

  def setupArgDB(self, clArgs, initDB):
    global argDB
    if not initDB is None:
      argDB = initDB
    else:
      argDB = nargs.ArgDict('ArgDict')

    argDB.setLocalType('help',           nargs.ArgBool('Print help message'))
    argDB.setLocalType('noConfigure',    nargs.ArgBool('Suppress configure'))
    argDB.setLocalType('forceConfigure', nargs.ArgBool('Force a  reconfigure'))
    argDB.setLocalType('displayTarget',  nargs.ArgBool('Display a target'))
    # argDB manipulation
    argDB.setLocalType('arg',            nargs.ArgString('Name of an argument database key'))
    argDB.setLocalType('fileset',        nargs.ArgString('Name of a FileSet or full path of an individual file'))
    argDB.setLocalType('regExp',         nargs.ArgString('Regular expression'))

    # Cannot just check whether initDB is given since an empty one comes in during installation
    if not argDB.has_key('noConfigure'):    argDB['noConfigure']    = 0
    if not argDB.has_key('forceConfigure'): argDB['forceConfigure'] = 0
    if not argDB.has_key('displayTarget'):  argDB['displayTarget']  = 0

    argDB.insertArgs(clArgs)
    return argDB

  def setupSourceDB(self):
    self.debugPrint('Reading source database from '+self.sourceDBFilename, 2, 'sourceDB')
    if os.path.exists(self.sourceDBFilename):
      dbFile        = open(self.sourceDBFilename, 'r')
      self.sourceDB = cPickle.load(dbFile)
      dbFile.close()

      # Make sourceDB paths absolute
      newDB = sourceDatabase.SourceDB()
      pwd   = self.getRoot()
      for key in self.sourceDB:
        new_key        = pwd+key
        newDB[new_key] = self.sourceDB[key]
      self.sourceDB = newDB
    else:
      self.sourceDB = sourceDatabase.SourceDB()
    atexit.register(self.cleanup)
    self.sourceDB.setFromArgs(argDB)
    if not argDB.has_key('restart') or not int(argDB['restart']):
      for source in self.sourceDB:
        self.sourceDB.clearUpdateFlag(source)
    return

  def saveSourceDB(self):
    self.debugPrint('Saving source database in '+self.sourceDBFilename, 2, 'sourceDB')
    # Make sourceDB paths relative
    newDB = sourceDatabase.SourceDB()
    pwd   = self.getRoot()
    for key in self.sourceDB:
      new_key        = re.split(pwd,key)[-1]
      newDB[new_key] = self.sourceDB[key]
    self.sourceDB = newDB

    if os.path.exists(os.path.dirname(self.sourceDBFilename)):
      dbFile = open(self.sourceDBFilename, 'w')
      cPickle.dump(self.sourceDB, dbFile)
      dbFile.close()
    return

  def updateRepositoryDirs(self, repositoryDirs):
    for url in self.executeTarget('getDependencies'):
      project = self.getInstalledProject(url)
      if not project is None:
        root = project.getRoot()
        repositoryDirs.append(root)
        try:
          self.getMakeModule(root).PetscMake(argDB = self.argDB).updateRepositoryDirs(repositoryDirs)
        except ImportError:
          self.debugPrint('Invalid repository: No make module in '+root, 4, 'compile')
    return

  def getSIDLDefaults(self):
    if not hasattr(self, 'sidlDefaults'):
      if not self.filesets.has_key('sidl'):
        self.filesets['sidl'] = None
      if self.filesets.has_key('bootstrap'):
        self.sidlDefaults = BSTemplates.sidlTargets.Defaults(self.project, self.sourceDB, self.argDB, self.filesets['sidl'], bootstrapPackages = self.filesets['bootstrap'])
      else:
        self.sidlDefaults = BSTemplates.sidlTargets.Defaults(self.project, self.sourceDB, self.argDB, self.filesets['sidl'])
      self.updateRepositoryDirs(self.sidlDefaults.usingSIDL.repositoryDirs)
    return self.sidlDefaults

  def getCompileDefaults(self):
    if not hasattr(self, 'compileDefaults'):
      if not self.filesets.has_key('etags'):
        self.filesets['etags'] = None
      self.compileDefaults = BSTemplates.compileTargets.Defaults(self.getSIDLDefaults(), self.filesets['etags'])
    return self.compileDefaults

  def disgustingPythonLink(self, package, implModule):
    # We must makes links into the Python stub directories to the Python server directories
    # because the directories coincide. We should remove this coincidence, but I am putting
    # this off until our compiler works.
    linkPath   = os.path.join(self.getSIDLDefaults().usingSIDL.getClientRootDir('Python'), implModule)
    modulePath = os.path.join(self.directories['main'], 'server-python-'+package, implModule)
    if os.path.islink(linkPath): os.remove(linkPath)
    try: os.symlink(modulePath, linkPath)
    except: pass
    return

  def t_getDependencies(self):
    return []

  def t_setupBootstrap(self):
    '''Initialize variables involved in a bootstrap build'''
    return

  def t_configure(self):
    if argDB['noConfigure']: return
    import config.framework
    import install.base

    root      = self.getRoot()
    log       = os.path.join(root, 'configure.log')
    framework = config.framework.Framework(sys.argv[1:])
    for arg in ['debugLevel', 'debugSections']:
      framework.argDB[arg] = argDB[arg]
    framework.argDB['log'] = log
    # Perhaps these initializations should just be local arguments
    framework.argDB['CPPFLAGS'] = ''
    framework.argDB['LIBS']     = ''
    # Load default configure module
    try:
      framework.children.append(install.base.Base(framework.argDB).getMakeModule(root, 'configure').Configure(framework))
    except ImportError:
      return
    # Run configuration only if the log file was absent or it is forced (must setup logging early to check)
    framework.setupLogging()
    if not framework.logExists or argDB['forceConfigure']:
      framework.configure()
      # Debugging
      if framework.argDB.has_key('dumpSubstitutions'):
        framework.dumpSubstitutions()
    return

  def t_sidl(self):
    return self.getSIDLDefaults().getSIDLTarget().execute()

  def t_compile(self):
    return self.getCompileDefaults().getCompileTarget().execute()

  def t_install(self):
    sidl = self.getSIDLDefaults().usingSIDL
    lang = 'Python'
    if lang in sidl.clientLanguages:
      self.project.appendPythonPath(sidl.getClientRootDir(lang, root = self.project.getRoot()))
    if lang in sidl.serverLanguages:
      for package in sidl.getPackages():
        self.project.appendPythonPath(sidl.getServerRootDir(lang, package))

    if 'Matlab' in sidl.clientLanguages:
      root = sidl.getClientRootDir('Matlab', root = self.project.getRoot())
      # sidl.getPackages() gives the names of the files, not the SIDL packages
      import commands
      packages = os.listdir(root)
      if len(packages) > 1: packages.remove('SIDL')
      self.project.appendPackages(packages)
      self.project.setMatlabPath(root)

    p = self.getInstalledProject(self.project.getUrl())
    if p is None:
      argDB['installedprojects'] = argDB['installedprojects']+[self.project]
    else:
      projects = argDB['installedprojects']
      projects.remove(p)
      argDB['installedprojects'] = projects+[self.project]
    return p

  def t_uninstall(self):
    p = self.getInstalledProject(self.project.getUrl())
    if not p is None:
      projects = argDB['installedprojects']
      projects.remove(p)
      argDB['installedprojects'] = projects
    return p

  def t_print(self):
    return self.getSIDLDefaults().getSIDLPrintTarget().execute()

  def t_default(self):
    self.executeTarget('configure')
    self.executeTarget('compile')
    return self.executeTarget('install')

  def t_recalc(self):
    return self.sourceDB.calculateDependencies()

  def t_printTargets(self):
    targets = self.targets.keys()
    for attr in dir(self):
      if attr[0:2] == 't_':
        targets.append(attr[2:])
    print 'Available targets: '+str(targets)

  def t_printSourceDB(self):
    print self.sourceDB

  def t_purge(self):
    if argDB.has_key('arg'):
      argNames = argDB['arg']
      if not isinstance(argNames, list): argNames = [argNames]
      for argName in argNames:
        if argDB.has_key(argName):
          self.debugPrint('Purging '+argName, 3, 'argDB')
          del argDB[argName]
      del argDB['arg']
    elif argDB.has_key('fileset'):
      setName = argDB['fileset']
      try:
        self.debugPrint('Purging source database of fileset '+setName, 1, 'sourceDB')
        for file in self.filesets[setName]:
          if self.sourceDB.has_key(file):
            self.debugPrint('Purging '+file, 3, 'sourceDB')
            del self.sourceDB[file]
      except KeyError:
        try:
          if self.sourceDB.has_key(setName):
            self.debugPrint('Purging '+setName, 3, 'sourceDB')
            del self.sourceDB[setName]
        except KeyError:
          print 'FileSet '+setName+' not found for purge'
    else:
      import re

      purgeRE = re.compile(argDB['regExp'])
      purges  = []
      for key in self.sourceDB:
        m = purgeRE.match(key)
        if m: purges.append(key)
      for source in purges:
        self.debugPrint('Purging '+source, 3, 'sourceDB')
        del self.sourceDB[source]

  def t_update(self):
    if argDB.has_key('fileset'):
      setName = argDB['fileset']
      try:
        self.debugPrint('Updating source database of fileset '+setName, 1, 'sourceDB')
        for file in self.filesets[setName]:
          self.debugPrint('Updating '+file, 3, 'sourceDB')
          self.sourceDB.updateSource(file)
      except KeyError:
        try:
          self.debugPrint('Updating '+setName, 3, 'sourceDB')
          self.sourceDB.updateSource(setName)
        except KeyError:
          print 'FileSet '+setName+' not found for update'
    else:
      import re

      print argDB['regExp']
      updateRE = re.compile(argDB['regExp'])
      updates  = []
      for key in self.sourceDB:
        m = updateRE.match(key)
        if m: updates.append(key)
      for source in updates:
        self.debugPrint('Updating '+source, 3, 'sourceDB')
        self.sourceDB.updateSource(source)

  def cleanup(self):
    self.saveSourceDB()
    return

  def printIndent(self, indent):
    for i in range(indent): sys.stdout.write(' ')

  def displayTransform(self, t, indent = 0):
    import transform

    if isinstance(t, transform.Transform):
      self.printIndent(indent)
      print str(t)
    elif isinstance(t, list):
      self.displayTransformPipe(t, indent)
    elif isinstance(t, tuple):
      self.displayTransformFan(t, indent)
    else:
      raise RuntimeError('Invalid transform '+str(t))
    return

  def displayTransformPipe(self, l, indent = 0):
    for t in l:
      self.displayTransform(t, indent)
      indent = indent + 2
    return

  def displayTransformFan(self, tup, indent = 0):
    for t in tup:
      self.displayTransform(t, indent)
    return

  def displayTarget(self, target, indent = 0):
    print str(target)
    self.displayTransform(target.transforms, indent = indent+2)
    return

  def displayBSTarget(self, target):
    print 'Displaying '+str(target)
    if self.targets.has_key(target):
      target = self.targets[target]
    elif target == 'sidl':
      target = self.getSIDLDefaults().getSIDLTarget()
    elif target == 'compile':
      target = self.getCompileDefaults().getCompileTarget()
    else:
      raise RuntimeError('Cannot display target '+str(target))
    self.displayTarget(target)
    return

  def executeTarget(self, target):
    if self.targets.has_key(target):
      output = self.targets[target].execute()
    elif hasattr(self, 't_'+target):
      output = getattr(self, 't_'+target)()
    else:
      print 'Invalid target: '+target
      output = ''
    return output

  def setupBuild(self):
    return

  def main(self, target = None):
    # Hook for user setup after creation
    self.setupBuild()

    # add to database list of packages in current project
##    try:
##      import SIDL.Loader
##      import SIDLLanguage.Parser
##      import SIDLLanguage.Visitor

##      compiler = SIDLLanguage.Parser.Parser(SIDL.Loader.createClass('ANL.SIDLCompilerI.SIDLCompiler'))
##      if argDB.has_key('installedpackages'):
##        ipackages = argDB['installedpackages']
##      else: ipackages = []
##      for source in self.filesets['sidl'].getFiles():
##        tree = compiler.parseFile(source)
##        v = SIDLLanguage.Visitor.Visitor(SIDL.Loader.createClass('ANL.SIDLVisitorI.GetPackageNames'))
##        tree.accept(v)
##        for p in v.getnames():
##          if not p in ipackages:
##            ipackages.append(p)
##      argDB['installedpackages'] = ipackages
##    except: pass

    try:
      if target is None:               target = argDB.target
      if not isinstance(target, list): target = [target]
      if argDB['displayTarget']:
        map(self.displayBSTarget, target)
      else:
        map(self.executeTarget, target)
    except Exception, e:
      print str(e)
      if not argDB.has_key('noStackTrace') or not int(argDB['noStackTrace']):
        print traceback.print_tb(sys.exc_info()[2])
    self.cleanupDir(self.tmpDir)
    return
