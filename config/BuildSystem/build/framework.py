import user
import importer
import base
import sourceDatabase

import atexit
import cPickle
import os
import sys
import nargs

if not hasattr(sys, 'version_info'):
  raise RuntimeError('You must have Python version 2.4 or higher to run the build system')

class Framework(base.Base):
  '''This is the base class for all user make modules'''
  def __init__(self, project, clArgs = None, argDB = None):
    '''Setup the project, argument database, and source database'''
    try:
      import gc
      #gc.set_debug(gc.DEBUG_LEAK)
    except ImportError: pass
    base.Base.__init__(self, clArgs, argDB)
    import build.builder
    self.project         = project
    self.targets         = {}
    self.directories     = {}
    self.filesets        = {}
    self.configureHeader = None
    self.builder         = build.builder.Builder(None)
    self.createTmpDir()
    return

  def setupArgDB(self, argDB, clArgs):
    '''Setup argument types, using the database created by base.Base'''

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

    if not 'installedprojects'  in self.argDB: self.argDB['installedprojects']  = []
    if not 'installedLanguages' in self.argDB: self.argDB['installedLanguages'] = ['Python', 'Cxx']
    if not 'clientLanguages'    in self.argDB: self.argDB['clientLanguages']    = []

    base.Base.setupArgDB(self, argDB, clArgs)
    return argDB

  def setupSourceDB(self, proj):
    '''Load any existing source database for the given project, and register its save method'''
    import project

    root     = project.ProjectPath('', proj.getUrl())
    filename = project.ProjectPath('bsSource.db', proj.getUrl())
    self.debugPrint('Reading source database for '+proj.getUrl()+' from '+str(filename), 2, 'sourceDB')
    if os.path.exists(str(filename)):
      try:
        dbFile        = open(str(filename), 'r')
        self.sourceDB = cPickle.load(dbFile)
        self.sourceDB.filename = filename
        dbFile.close()
      except Exception, e:
        self.debugPrint('Source database '+str(filename)+' could not be read: '+str(e)+'. Creating a new one', 2, 'sourceDB')
        self.sourceDB = sourceDatabase.SourceDB(root, filename)
    else:
      self.debugPrint('Source database '+str(filename)+' does not exist. Creating a new one', 2, 'sourceDB')
      self.sourceDB = sourceDatabase.SourceDB(root, filename)
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
      if isinstance(self.argDB['TMPDIR'], int):
        # How in the hell is TMPDIR being set to 1?
        tmpDir = '/tmp'
      else:
        tmpDir = self.argDB['TMPDIR']
    elif 'TMPDIR' in os.environ:
      tmpDir = os.environ['TMPDIR']
    else:
      tmpDir = '/tmp'

    self.argDB['TMPDIR'] = tmpDir
    while not self.checkTmpDir(tmpDir):
      tmpDir = self.argDB['TMPDIR']

    self.tmpDir = os.path.join(tmpDir, 'bs-'+str(os.getpid()))
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
    '''Augment the project dependence graph with this project
       - The project and dependencies MUST be activated prior to calling this method'''
    if not 'projectDependenceGraph' in self.argDB:
      import build.buildGraph
      self.argDB['projectDependenceGraph'] = build.buildGraph.BuildGraph()
    self.dependenceGraph = self.argDB['projectDependenceGraph']
    self.dependenceGraph.addVertex(self.project)
    self.dependenceGraph.clearEdges(self.project, outOnly = 1)
    self.dependenceGraph.addEdges(self.project, outputs = map(self.getInstalledProject, self.executeTarget('getDependencies')))
    self.argDB['projectDependenceGraph'] = self.dependenceGraph
    return self.dependenceGraph

  def getSIDLTemplate(self):
    if not hasattr(self, '_sidlTemplate'):
      import build.templates.SIDL

      self._sidlTemplate = build.templates.SIDL.Template(self.sourceDB, self.project, self.dependenceGraph)
      # Add default client languages
      map(self._sidlTemplate.addClient, self.argDB['clientLanguages'])
    return self._sidlTemplate
  sidlTemplate = property(getSIDLTemplate, doc = 'This is the default template for SIDL operations')

  def getCompileTemplate(self):
    if not hasattr(self, '_compileTemplate'):
      import build.templates.Compile

      packages = map(lambda f: os.path.splitext(os.path.basename(f))[0], self.filesets['sidl'])
      self._compileTemplate = build.templates.Compile.Template(self.argDB, self.sourceDB, self.project, self.dependenceGraph, self.sidlTemplate.usingSIDL, packages)
    return self._compileTemplate
  compileTemplate = property(getCompileTemplate, doc = 'This is the default template for source operations')

  def t_getDependencies(self):
    '''Return a list of the URLs for projects upon which this one depends'''
    return []

  def t_activate(self):
    '''Load all necessary data for this project into the current RDict, without destroying previous data'''
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
    self.debugPrint('Deactivated project '+str(self.project), 2, 'install')
    return self.project

  def t_configure(self):
    '''Runs configure.py if it is present, and either configure.log is missing or -forceConfigure is given'''
    if self.argDB['noConfigure']: return
    import config.framework

    root      = self.project.getRoot()
    framework = config.framework.Framework(sys.argv[1:])
    for arg in ['debugLevel', 'debugSections']:
      framework.argDB[arg] = self.argDB[arg]
    framework.argDB['log'] = os.path.join(root, 'configure.log')
    if not self.configureHeader is None:
      framework.header     = self.configureHeader
    # Load default configure module
    try:
      framework.addChild(self.getMakeModule(root, 'configure').Configure(framework))
    except ImportError:
      return
    # Run configuration only if the log file was absent or it is forced
    if self.argDB['forceConfigure'] or not framework.checkLog(framework.logName):
      try:
        framework.configure()
      except Exception, e:
        import traceback

        msg = 'CONFIGURATION FAILURE:\n'+str(e)+'\n'
        print msg
        framework.log.write(msg)
        traceback.print_tb(sys.exc_info()[2], file = framework.log)
        raise e
      framework.storeSubstitutions(self.argDB)
    return

  def t_sidl(self):
    '''Recompile the SIDL for this project'''
    return self.executeGraph(self.sidlTemplate.getTarget(), input = self.filesets['sidl'])

  def buildClient(self, proj, lang):
    import build.buildGraph

    clientDir = self.compileTemplate.usingSIDL.getClientRootDir(lang)
    self.debugPrint('Building '+lang+' client in '+proj.getRoot(), 1, 'build')
    maker  = self.getMakeModule(proj.getRoot()).PetscMake(None, self.argDB)
    maker.setupProject()
    maker.setupDependencies()
    maker.setupSourceDB(maker.project)
    sidlGraph    = maker.sidlTemplate.getClientTarget(lang, fullTarget = 1, forceRebuild = 1)
    compileGraph = maker.compileTemplate.getClientTarget(lang)
    compileGraph.prependGraph(sidlGraph)
    maker.executeGraph(compileGraph, input = maker.filesets['sidl'])
    return

  def missingClients(self):
    '''Check that this project has built all the clients, and if not return True'''
    import build.buildGraph

    for lang in self.compileTemplate.usingSIDL.clientLanguages:
      clientDir = self.compileTemplate.usingSIDL.getClientRootDir(lang)
      if not os.path.isdir(os.path.join(self.project.getRoot(), clientDir)):
        self.debugPrint('Building missing '+lang+' client in '+self.project.getRoot(), 1, 'build')
        return 1
    return 0

  def getProjectCompileGraph(self, forceRebuild = 0):
    '''Return the compile graph for the given project without dependencies'''
    input        = {None: self.filesets['sidl']}
    sidlGraph    = self.sidlTemplate.getTarget(forceRebuild = forceRebuild)
    compileGraph = self.compileTemplate.getTarget()
    compileGraph.prependGraph(sidlGraph)
    return (compileGraph, input)

  def getCompileGraph(self):
    if 'checkpoint' in self.argDB:
      input        = {}
      self.builder = cPickle.loads(self.argDB['checkpoint'])
      compileGraph = self.builder.buildGraph
      self.debugPrint('Loaded checkpoint for '+str(self.project), 2, 'build')
    else:
      import build.buildGraph

      compileGraph = build.buildGraph.BuildGraph()
      input        = {}
      for p in build.buildGraph.BuildGraph.topologicalSort(self.dependenceGraph, self.project):
        try:
          if p == self.project:
            maker = self
          else:
            maker = self.getMakeModule(p.getRoot()).PetscMake(None, self.argDB)
            maker.setupProject()
            maker.setupDependencies()
            maker.setupSourceDB(maker.project)
            maker.setupBuild()
          (depGraph, depInput) = maker.getProjectCompileGraph(forceRebuild = maker.missingClients())
          compileGraph.prependGraph(depGraph)
          self.debugPrint('Prepended graph for '+str(maker.project), 4, 'build')
          if None in depInput:
            for r in build.buildGraph.BuildGraph.getRoots(depGraph): depInput[r] = depInput[None]
            del depInput[None]
          input.update(depInput)
        except ImportError:
          self.debugPrint('No make module present in '+p.getRoot(), 2, 'build')
    return (compileGraph, input)

  def t_sidlCheckpoint(self):
    '''Recompile the SIDL for this project'''
    import build.buildGraph

    # Add project dependency compile graphs
    # TODO: Remove all "forward" edges in dependenceGraph (edges which connect further down to already reachable nodes)
    depGraphs = []
    for v in self.dependenceGraph.outEdges[self.project]:
      try:
        maker = self.getMakeModule(v.getRoot()).PetscMake(None, self.argDB)
        maker.setupProject()
        maker.setupDependencies()
        maker.setupSourceDB(maker.project)
        maker.setupBuild()
        depGraphs.append(maker.executeTarget('sidlCheckpoint'))
      except ImportError:
        self.debugPrint('No make module present in '+v.getRoot(), 2, 'build')

    sidlGraph    = self.sidlTemplate.getTarget()
    articGraph   = build.buildGraph.BuildGraph([build.transform.Transform()])
    compileGraph = self.compileTemplate.getTarget()
    startVertex  = build.buildGraph.BuildGraph.getRoots(sidlGraph)[0]
    input        = {startVertex: self.filesets['sidl']}
    endVertex    = build.buildGraph.BuildGraph.getRoots(articGraph)[0]
    compileGraph.prependGraph(articGraph)
    compileGraph.prependGraph(sidlGraph)

    output = self.executeGraph(compileGraph, start = startVertex, input = input, end = endVertex)
    compileGraph.removeSubgraph(sidlGraph)
    for g in depGraphs:
      compileGraph.prependGraph(g)
    self.builder.currentVertex = None
    self.argDB['checkpoint']   = cPickle.dumps(self.builder)
    return compileGraph

  def t_compile(self):
    '''Recompile the entire source for this project'''
    (compileGraph, input) = self.getCompileGraph()
    return self.executeGraph(compileGraph, input = input)

  def t_compilePrograms(self):
    '''Recompile executables for this project'''
    (compileGraph, input) = self.getCompileGraph()
    for program in self.filesets['programs']:
      compileGraph.appendGraph(self.compileTemplate.getExecutableTarget(program))
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
    # Remove any build checkpoint
    if 'checkpoint' in self.argDB:
      del self.argDB['checkpoint']
    return self.project

  def t_uninstall(self):
    '''Remove all instances of this project from the current RDict'''
    # Remove project from 'installedprojects'
    projects = self.argDB['installedprojects']
    map(lambda p: projects.remove(p), self.getInstalledProject(self.project.getUrl(), returnAll = 1))
    self.argDB['installedprojects'] = projects
    # Remove project from 'projectDependenceGraph'
    dependenceGraph = self.argDB['projectDependenceGraph']
    dependenceGraph.removeVertex(self.project)
    self.argDB['projectDependenceGraph'] = dependenceGraph
    # Remove configure log
    logName = os.path.join(self.project.getRoot(), 'configure.log')
    if os.path.isfile(logName):
      os.remove(logName)
    return self.project

  def t_citool(self):
    '''Run bk citool on all the projects'''
    for p in self.argDB['installedprojects']:
      print 'Running bk citool on '+p.getRoot()
      self.executeShellCommand('cd '+p.getRoot()+'; bk citool')

  def t_push(self):
    '''Run bk push on all the projects'''
    for p in self.argDB['installedprojects']:
      print 'Running bk push on '+p.getRoot()
      try:
        self.executeShellCommand('cd '+p.getRoot()+'; bk push')
      except:
        pass

  def t_pull(self):
    '''Run bk pull on all the projects'''
    for p in self.argDB['installedprojects']:
      print 'Running bk pull on '+p.getRoot()
      self.executeShellCommand('cd '+p.getRoot()+'; bk pull')

  def getHeadRevision(self, proj):
    import install.retrieval
    return install.retrieval.Retriever().bkHeadRevision(proj.getRoot())

  def t_makeStamp(self):
    import build.buildGraph

    stamp  = {}
#    bsProj = self.getInstalledProject('bk://sidl.bkbits.net/BuildSystem')
#    stamp[bsProj.getUrl()] = self.getHeadRevision(bsProj)
    for p in build.buildGraph.BuildGraph.depthFirstVisit(self.dependenceGraph, self.project):
      stamp[p.getUrl()] = self.getHeadRevision(p)
    return stamp

  def t_default(self):
    '''Activate, configure, build, and install this project'''
    return ['activate', 'configure', 'compile', 'install']

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
    '''Purge a fileset from the source database, identified using -fileset=<set name>'''
    if 'fileset' in self.argDB:
      setName = self.argDB['fileset']
      if setName in self.filesets:
        self.debugPrint('Purging source database of fileset '+setName, 1, 'sourceDB')
        for f in self.filesets[setName]:
          self.debugPrint('Purging '+f, 3, 'sourceDB')
          try:
            del self.sourceDB[f]
          except KeyError:
            print 'File '+f+' not found for purge'
      else:
        print 'FileSet '+setName+' not found for purge'
      self.sourceDB.save()
    return

  def t_update(self):
    '''Update a value in the source database, identifier using -fileset=<set name>'''
    if 'fileset' in self.argDB:
      setName = self.argDB['fileset']
      if setName in self.filesets:
        self.debugPrint('Updating source database of fileset '+setName, 1, 'sourceDB')
        for f in self.filesets[setName]:
          self.debugPrint('Updating '+f, 3, 'sourceDB')
          try:
            self.sourceDB.updateSource(f)
          except KeyError:
            print 'File '+f+' not found in source database'
      else:
        print 'FileSet '+setName+' not found for update'
      self.sourceDB.save()
    return

  def t_printSIDLHTML(self):
    '''Print all the SIDL dependencies as HTML'''
    import build.compile.SIDL

    self.argDB.target = []
    for v in self.sidlTemplate.getClientTarget('Python').vertices:
      if hasattr(v, 'getIncludeFlags'):
        includes = v.getIncludeFlags(None)
    mod      = build.compile.SIDL.Compiler(self.sourceDB, 'Python', None, 0, self.sidlTemplate.usingSIDL).getCompilerModule('scandalDoc')
    args     = ['-printer=[ANL.SIDL.PrettyPrinterHTML]']+includes+self.filesets['sidl']
    self.debugPrint('Running scandalDoc with arguments '+str(args), 3, 'build')
    compiler = mod.ScandalDoc(args)
    compiler.run()
    return compiler.outputFiles

  def t_printSIDL(self):
    '''Print all the SIDL dependencies as plain text'''
    import build.compile.SIDL

    self.argDB.target = []
    for v in self.sidlTemplate.getClientTarget('Python').vertices:
      if hasattr(v, 'getIncludeFlags'):
        includes = v.getIncludeFlags(None)
    mod      = build.compile.SIDL.Compiler(self.sourceDB, 'Python', None, 0, self.sidlTemplate.usingSIDL).getCompilerModule('scandalDoc')
    args     = ['-printer=[ANL.SIDL.PrettyPrinter]']+includes+self.filesets['sidl']
    self.debugPrint('Running scandalDoc with arguments '+str(args), 3, 'build')
    compiler = mod.ScandalDoc(args)
    compiler.run()
    return compiler.outputFiles

  def t_printSIDLBabel(self,exportDir):
    '''Print the SIDL for this project and all dependent projects in
       a format Babel can parse'''
    import build.compile.SIDL

    self.argDB.target = []
    for v in self.sidlTemplate.getClientTarget('Python').vertices:
      if hasattr(v, 'getIncludeFlags'):
        includes = v.getIncludeFlags(None)
    mod      = build.compile.SIDL.Compiler(self.sourceDB, 'Python', None, 0, self.sidlTemplate.usingSIDL).getCompilerModule('scandalDoc')
    args     = ['-filename='+os.path.join(exportDir,'allsidl.sidl')]+['-printer=[ANL.SIDL.PrettyPrinterBabel]']+includes+self.filesets['sidl']
    self.debugPrint('Running scandalDoc with arguments '+str(args), 3, 'build')
    compiler = mod.ScandalDoc(args)
    compiler.run()
    return compiler.outputFiles

  def t_exportBabel(self):
    '''Exports all the SIDL projects and impls in a form that Babel can handle'''
    self.argDB.setType('exportDir', nargs.ArgString(key='exportDir', help='Directory to export for Babel'))
    exportDir = self.argDB['exportDir']
    if not os.path.isdir(exportDir): os.makedirs(exportDir)
    self.t_printSIDLBabel(exportDir)

    directories = self.getDependencyPaths()
    import getsplicers
    getsplicers.getSplicers(directories)
    try:
      #output = self.executeShellCommand('cd '+exportDir+'; babel --server=C allsidl.sidl')
      import commands
      (status,output) = commands.getstatusoutput('cd '+exportDir+';babel --server=C++ allsidl.sidl')
    except:
      pass
    print status
    print output
    import setsplicers
    setsplicers.setSplicers(exportDir)

  def getDependencyPaths(self):
    directories = [self.project.getRoot()]
    ip          = self.argDB['installedprojects']
    for j in self.t_getDependencies():
      for l in ip:
        if l.getUrl() == j:
          maker = self.getMakeModule(l.getRoot()).PetscMake(None, self.argDB)
          dirs  = maker.getDependencyPaths()
          for d in dirs:
            if not d in directories: directories.append(d)
    return directories
    
  def t_updateBootstrap(self):
    '''Create a bootstrap tarball and copy it to the FTP site'''
    import install.installerclass

    installer = install.installerclass.Installer()
    tarball   = installer.backup(self.project.getUrl())
    #self.executeShellCommand('scp '+tarball+' petsc@terra.mcs.anl.gov://mcs/ftp/pub/petsc/sidl/'+tarball)
    os.rmdir('backup')
    raise RuntimeError('Need to fix the path')
    return

    
  def t_updateWebsite(self):
    '''Print all the SIDL dependencies as HTML and move to the website'''
    for f in self.executeTarget('printSIDLHTML'):
      self.executeShellCommand('scp '+f+' '+self.project.getWebDirectory()+'/'+f)
      os.remove(f)
    return

  def cpFile(self, localFile, remoteDirectory, remoteFile = None, recursive = 0):
    cmd = 'scp '
    if recursive: cmd += '-r '
    if remoteFile:
      try: self.executeShellCommand(cmd+localFile+' '+os.path.join(remoteDirectory, remoteFile))
      except: pass
    else:
      try: self.executeShellCommand(cmd+localFile+' '+remoteDirectory)
      except: pass
    return

  def cpWebsite(self, localFile, remoteFile = None, recursive = 0):
    return self.cpFile(localFile, self.project.getWebDirectory(), remoteFile, recursive)

  def setupProject(self):
    '''Hook for user operations before project activation'''
    return

  def setupBuild(self):
    '''Hook for user operations after project activation, but before build'''
    return

  def stampBuild(self):
    '''Create a version stamp for this build, store it in the RDict and log it'''
    stamp = self.executeTarget('makeStamp')
    self.argDB['stamp-'+self.project.getUrl()] = stamp
    self.debugPrint('Build stamp: '+str(stamp), 4, 'build')
    return stamp

  def executeGraph(self, graph, start = None, input = None, end = None):
    '''Execute a BuildGraph'''
    output = None
    if self.argDB['displayTarget']:
      graph.display()
    else:
      self.builder.buildGraph = graph
      for vertex in self.builder.execute(start = start, input = input):
        output = vertex.output
        if end == vertex:
          break
    return output

  def executeTarget(self, target):
    '''Execute the target and return the output'''
    self.debugPrint('Executing target '+target, 4, 'build')
    output = None
    if self.targets.has_key(target):
      self.executeGraph(self.targets[target])
    elif hasattr(self, 't_'+target):
      output = getattr(self, 't_'+target)()
    else:
      print 'Invalid target: '+str(target)
    return output

  def expandTargets(self,target):
    '''Return a copy of targets, after expansion of special targets'''
    if target is None:
      target = self.argDB.target[:]
    else:
      if not isinstance(target, list):
        target = [target]
      else:
        target = target[:]
    if 'default' in target:
      idx = target.index('default')
      target[idx:idx] = self.executeTarget('default')
      target.remove('default')
    return target

  def mainBuild(self, target = None):
    '''Execute the build operation'''
    if self.argDB['help']:
      self.executeTarget('printTargets')
      return
    target = self.expandTargets(target)
    self.setupProject()
    if 'activate' in target:
      self.executeTarget('activate')
      target.remove('activate')
      if not len(target): return
    self.setupDependencies()
    self.setupSourceDB(self.project)
    if 'configure' in target:
      self.executeTarget('configure')
      target.remove('configure')
      if not len(target): return
    self.setupBuild()
    self.stampBuild()
    map(self.executeTarget, target)
    return

  def main(self, target = None):
    '''Execute the build operation and handle any exceptions'''
    try:
      return self.mainBuild(target)
    except Exception, e:
      import traceback

      msg = 'BUILD FAILURE:\n'+str(e)+'\n'
      print msg
      self.log.write(msg)
      traceback.print_tb(sys.exc_info()[2], file = self.log)
      if not self.argDB['noStackTrace']:
        traceback.print_tb(sys.exc_info()[2])
