import importer
import base
import sourceDatabase

import atexit
import cPickle
import os

class Framework(base.Base):
  '''This is the base class for all user make modules'''
  def __init__(self, project, clArgs = None, argDB = None):
    '''Setup the project, argument database, and source database'''
    base.Base.__init__(self, clArgs, argDB)
    import build.builder
    self.project         = project
    self.targets         = {}
    self.directories     = {}
    self.filesets        = {}
    self.configureHeader = None
    self.builder         = build.builder.Builder(None)
    self.setupSourceDB(os.path.join(self.project.getRoot(), 'bsSource.db'))
    self.setupDependencies()
    self.createTmpDir()
    return

  def setupArgDB(self, argDB, clArgs):
    '''Setup argument types, using the database created by base.Base'''
    import nargs

    # Generic arguments
    argDB.setType('help',           nargs.ArgBool(None, 0, 'Print help message',   isTemporary = 1), forceLocal = 1)
    argDB.setType('noConfigure',    nargs.ArgBool(None, 0, 'Suppress configure',   isTemporary = 1), forceLocal = 1)
    argDB.setType('forceConfigure', nargs.ArgBool(None, 0, 'Force a  reconfigure', isTemporary = 1), forceLocal = 1)
    argDB.setType('displayTarget',  nargs.ArgBool(None, 0, 'Display a target',     isTemporary = 1), forceLocal = 1)
    argDB.setType('noStackTrace',   nargs.ArgBool(None, 0, 'Suppress a stack trace on error'), forceLocal = 1)
    argDB.setType('checkpoint',     nargs.Arg(None, None,  'Pickled state of evaluation'), forceLocal = 1)
    # Source database manipulation
    argDB.setType('restart',        nargs.ArgBool(None, 0, 'Restart the build',    isTemporary = 1), forceLocal = 1)
    # Argument database manipulation
    argDB.setType('fileset',        nargs.Arg(None, None, 'Name of a FileSet or full path of an individual file', isTemporary = 1), forceLocal = 1)
    argDB.setType('regExp',         nargs.Arg(None, None, 'Regular expression',                                   isTemporary = 1), forceLocal = 1)
    base.Base.setupArgDB(self, argDB, clArgs)
    return argDB

  def setupSourceDB(self, filename):
    '''Load any existing source database and use the project root to make all paths absolute
       - Also register the save method, provide the argument database, and clear update flags if necessary'''
    self.sourceDBFilename = filename
    self.debugPrint('Reading source database from '+self.sourceDBFilename, 2, 'sourceDB')
    if os.path.exists(self.sourceDBFilename):
      try:
        dbFile        = open(self.sourceDBFilename, 'r')
        self.sourceDB = self.makeSourceDBPathsAbsolute(cPickle.load(dbFile))
        dbFile.close()
      except Exception:
        self.sourceDB = sourceDatabase.SourceDB(self.argDB)
    else:
      self.sourceDB = sourceDatabase.SourceDB(self.argDB)
    atexit.register(self.saveSourceDB)
    if not self.argDB['restart']:
      for source in self.sourceDB:
        self.sourceDB.clearUpdateFlag(source)
    return

  def saveSourceDB(self):
    '''Save the source database to a file. The saved database with have path names relative to the project root.'''
    if os.path.exists(os.path.dirname(self.sourceDBFilename)):
      self.debugPrint('Saving source database in '+self.sourceDBFilename, 2, 'sourceDB')
      dbFile = open(self.sourceDBFilename, 'w')
      cPickle.dump(self.makeSourceDBPathsRelative(self.sourceDB), dbFile)
      dbFile.close()
    else:
      self.debugPrint('Could not save source database in '+self.sourceDBFilename, 1, 'sourceDB')
    return

  def makeSourceDBPathsAbsolute(self, sourceDB):
    '''Return another source database in which all paths are absolute'''
    newDB = sourceDatabase.SourceDB(self.argDB)
    pwd   = self.project.getRoot()
    for key in sourceDB:
      new_key        = pwd+key
      newDB[new_key] = sourceDB[key]
    return newDB

  def makeSourceDBPathsRelative(self, sourceDB):
    '''Return another source database in which all paths are relative to the root of this project'''
    import re

    newDB = sourceDatabase.SourceDB(self.argDB)
    pwd   = self.project.getRoot()
    for key in sourceDB:
      new_key        = re.split(pwd, key)[-1]
      newDB[new_key] = sourceDB[key]
    return newDB

  def checkTmpDir(self, tmpDir):
    '''Check that the temporary direcotry exists and has sufficient space available'''
    if not os.path.exists(tmpDir):
      del self.argDB['TMPDIR']
      self.argDB.setType('TMPDIR', nargs.ArgDir(None, None, 'Temporary directory '+tmpDir+' does not exist. Select another directory'))
      newTmp = self.argDB['TMPDIR']
      return 0

    try:
      stats     = os.statvfs(tmpDir)
      freeSpace = stats.f_bavail*stats.f_frsize
      if freeSpace < 50*1024*1024:
        del self.argDB['TMPDIR']
        self.argDB.setType('TMPDIR', nargs.ArgDir(None, None,'Insufficient space ('+str(freeSpace/1024)+'K) on '+tmpDir+'. Select another directory'))
        newTmp = self.argDB['TMPDIR']
        return 0
    except: pass
    return 1

  def createTmpDir(self):
    '''Create a valid temporary directory and store it in argDB["TMPDIR"]'''
    import tempfile

    if 'TMPDIR' in self.argDB:
      tmpDir = self.argDB['TMPDIR']
    elif 'TMPDIR' in os.environ:
      tmpDir = os.environ['TMPDIR']
    else:
      tmpDir = '/tmp'

    self.argDB['TMPDIR'] = tmpDir
    while not self.checkTmpDir(tmpDir):
      tmpDir = self.argDB['TMPDIR']

    self.tmpDir = os.path.join(tmpDir, 'bs', str(os.getpid()))
    if not os.path.exists(self.tmpDir):
      try:
        os.makedirs(self.tmpDir)
      except:
        raise RuntimeError('Cannot create tmp directory '+self.tmpDir)
    tempfile.tempdir = self.tmpDir
    atexit.register(self.destroyTmpDir)
    return

  def destroyTmpDir(self):
    if not os.path.exists(self.tmpDir): return
    import shutil
    return shutil.rmtree(self.tmpDir)

  def getMakeModule(self, root, name = 'make'):
    import imp

    (fp, pathname, description) = imp.find_module(name, [root])
    try:
      return imp.load_module(name, fp, pathname, description)
    finally:
      if fp: fp.close()

  def setupDependencies(self):
    self.dependenceGraph = self.argDB['projectDependenceGraph']
    self.dependenceGraph.addVertex(self.project)
    self.dependenceGraph.clearEdges(self.project)
    self.dependenceGraph.addEdges(self.project, outputs = map(self.getInstalledProject, self.executeTarget('getDependencies')))
    return self.dependenceGraph

  def getSIDLTemplate(self):
    if not hasattr(self, '_sidlTemplate'):
      import build.templates.SIDL

      self._sidlTemplate = build.templates.SIDL.Template(self.sourceDB, self.project, self.dependenceGraph)
    return self._sidlTemplate
  sidlTemplate = property(getSIDLTemplate, doc = 'This is the default template for SIDL operations')

  def getCompileTemplate(self):
    if not hasattr(self, '_compileTemplate'):
      import build.templates.Compile

      packages = map(lambda f: os.path.splitext(os.path.basename(f))[0], self.filesets['sidl'])
      self._compileTemplate = build.templates.Compile.Template(self.sourceDB, self.project, self.dependenceGraph, self.sidlTemplate.usingSIDL, packages)
    return self._compileTemplate
  compileTemplate = property(getCompileTemplate, doc = 'This is the default template for source operations')

  def t_getDependencies(self):
    '''Return a list of the URLs for projects upon which this one depends'''
    return []

  def t_activate(self):
    '''Load all necessary data for this project into the current RDict, without destroying previous data'''
    # Update language specific information
    self.compileTemplate.install()
    # Update project in 'installedprojects'
    self.argDB['installedprojects'] = [self.project]+self.argDB['installedprojects']
    self.debugPrint('Activated project '+str(self.project), 2, 'install')
    return self.project

  def t_deactivate(self):
    '''Unload the first matching project in the current RDict'''
    # Remove project from 'installedprojects'
    p = self.getInstalledProject(self.project.getUrl())
    if not p is None:
      projects = self.argDB['installedprojects']
      projects.remove(p)
      self.argDB['installedprojects'] = projects
    return self.project

  def t_configure(self):
    '''Runs configure.py if it is present, and either configure.log is missing or -forceConfigure is given'''
    if self.argDB['noConfigure']: return
    import config.framework
    import sys

    root      = self.project.getRoot()
    framework = config.framework.Framework(sys.argv[1:])
    for arg in ['debugLevel', 'debugSections']:
      framework.argDB[arg] = self.argDB[arg]
    framework.argDB['log'] = os.path.join(root, 'configure.log')
    if not self.configureHeader is None:
      framework.header     = self.configureHeader
    # Perhaps these initializations should just be local arguments
    framework.argDB['CPPFLAGS'] = ''
    framework.argDB['LIBS']     = ''
    # Load default configure module
    try:
      framework.children.append(self.getMakeModule(root, 'configure').Configure(framework))
    except ImportError:
      return
    # Run configuration only if the log file was absent or it is forced (must setup logging early to check)
    framework.setupLogging()
    if not framework.logExists or self.argDB['forceConfigure']:
      framework.configure()
    return

  def t_sidl(self):
    '''Recompile the SIDL for this project'''
    return self.executeGraph(self.sidlTemplate.getTarget(), input = self.filesets['sidl'])

  def t_sidlCheckpoint(self):
    '''Recompile the SIDL for this project'''
    import build.buildGraph

    sidlGraph    = self.sidlTemplate.getTarget()
    endVertex    = build.buildGraph.BuildGraph.getLeaves(sidlGraph)[0]
    compileGraph = self.compileTemplate.getTarget()
    compileGraph.prependGraph(sidlGraph)
    return self.executeGraph(compileGraph, input = self.filesets['sidl'], checkpoint = endVertex)

  def t_compile(self):
    '''Recompile the entire source for this project'''
    import build.buildGraph

    if 'checkpoint' in self.argDB:
      input        = None
      self.builder = cPickle.loads(self.argDB['checkpoint'])
      compileGraph = self.builder.buildGraph
      #del self.argDB['checkpoint']
    else:
      input        = self.filesets['sidl']
      sidlGraph    = self.sidlTemplate.getTarget()
      compileGraph = self.compileTemplate.getTarget()
      compileGraph.prependGraph(sidlGraph)
    return self.executeGraph(compileGraph, input = input)

  def t_install(self):
    '''Install all necessary data for this project into the current RDict'''
    # Update language specific information
    self.compileTemplate.install()
    # Update project in 'installedprojects'
    projects = self.argDB['installedprojects']
    map(lambda p: projects.remove(p), self.getInstalledProject(self.project.getUrl(), returnAll = 1))
    self.argDB['installedprojects'] = projects+[self.project]
    self.debugPrint('Installed project '+str(self.project), 2, 'install')
    # Update project in 'projectDependenceGraph'
    import build.buildGraph

    self.argDB['projectDependenceGraph'] = self.dependenceGraph
    self.debugPrint('Updated project dependence graph with project '+str(self.project), 2, 'install')
    return self.project

  def t_uninstall(self):
    '''Remove all instances of this project from the current RDict'''
    # Remove project from 'installedprojects'
    projects = self.argDB['installedprojects']
    map(lambda p: projects.remove(p), self.getInstalledProject(self.project.getUrl(), returnAll = 1))
    self.argDB['installedprojects'] = projects
    # TODO: Remove project from 'projectDependenceGraph'
    dependenceGraph = self.argDB['projectDependenceGraph']
    dependenceGraph.removeVertex(self.project)
    self.argDB['projectDependenceGraph'] = dependenceGraph
    return p

  def t_default(self):
    '''Configure, build, and install this project'''
    self.executeTarget('configure')
    self.executeTarget('compile')
    return self.executeTarget('install')

  def t_printTargets(self):
    '''Prints a list of all the targets available'''
    for target in self.targets:
      print target+':'
      print '  No help available'
    for attr in dir(self):
      if attr[0:2] == 't_':
        print attr[2:]+':'
        if getattr(self, attr).__doc__:
          print '  '+getattr(self, attr).__doc__
        else:
          print '  No help available'
    return

  def t_purge(self):
    '''Purge a value from a database:
  - With -fileset=<set name>, it purge an entire set from the source database
  - With -fileset=<filename>, it purge one file from the source database
  - With -regExp=<re>, it purge all files matching the expression from the source database'''
    if 'fileset' in self.argDB:
      setName = self.argDB['fileset']
      try:
        self.debugPrint('Purging source database of fileset '+setName, 1, 'sourceDB')
        for file in self.filesets[setName]:
          if file in self.sourceDB:
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

      purgeRE = re.compile(self.argDB['regExp'])
      purges  = []
      for key in self.sourceDB:
        m = purgeRE.match(key)
        if m: purges.append(key)
      for source in purges:
        self.debugPrint('Purging '+source, 3, 'sourceDB')
        del self.sourceDB[source]
    return

  def t_update(self):
    '''Purge a value in the source database:
  - With -fileset=<set name>, it updates an entire set
  - With -fileset=<filename>, it updates one file'''
    if 'fileset' in self.argDB:
      setName = self.argDB['fileset']
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

      updateRE = re.compile(self.argDB['regExp'])
      updates  = []
      for key in self.sourceDB:
        m = updateRE.match(key)
        if m: updates.append(key)
      for source in updates:
        self.debugPrint('Updating '+source, 3, 'sourceDB')
        self.sourceDB.updateSource(source)
    return

  def setupProject(self):
    '''Hook for user operations before project activation'''
    return

  def setupBuild(self):
    '''Hook for user operations after project activation, but before build'''
    return

  def executeGraph(self, graph, input = None, checkpoint = None):
    '''Execute a BuildGraph'''
    output = None
    if self.argDB['displayTarget']:
      graph.display()
    else:
      self.builder.buildGraph = graph
      for vertex in self.builder.execute(input = input):
        output = vertex.output
        if checkpoint == vertex:
          self.argDB['checkpoint'] = cPickle.dumps(self.builder)
          break
    return output

  def executeTarget(self, target):
    '''Execute the target and return the output'''
    output = None
    if self.targets.has_key(target):
      self.executeGraph(self.targets[target])
    elif hasattr(self, 't_'+target):
      output = getattr(self, 't_'+target)()
    else:
      print 'Invalid target: '+str(target)
    return output

  def main(self, target = None):
    '''Execute the build operation'''
    try:
      if target is None:
        target = self.argDB.target
      if not isinstance(target, list): target = [target]

      self.setupProject()
      # The activation target should happen before setup
      if 'activate' in target:
        self.executeTarget('activate')
        target.remove('activate')
      self.setupBuild()
      map(self.executeTarget, target)
    except Exception, e:
      print str(e)
      if not self.argDB['noStackTrace']:
        import traceback
        import sys
        print traceback.print_tb(sys.exc_info()[2])
    return
