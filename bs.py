#!/usr/bin/env python
import nargs
import maker
import sourceDatabase
import BSTemplates.sidlTargets
import BSTemplates.compileTargets

import atexit
import cPickle
import os
import sys
import traceback
import re

class Project:
  def __init__(self, name, url, root = None):
    if root is None: root = os.path.abspath(os.getcwd())
    # Needs to be immutable
    self.name = name
    self.url  = url
    self.root = root
    return

  def __str__(self):
    return self.name

  def __hash__(self):
    return self.name.__hash__()

  def __lt__(self, other):
    return self.name.__lt__(other.getName())

  def __le__(self, other):
    return self.name.__le__(other.getName())

  def __eq__(self, other):
    return self.name.__eq__(other.getName())

  def __ne__(self, other):
    return self.name.__ne__(other.getName())

  def __gt__(self, other):
    return self.name.__gt__(other.getName())

  def __ge__(self, other):
    return self.name.__ge__(other.getName())

  def getName(self):
    return self.name

  def getUrl(self):
    return self.url

  def getRoot(self):
    return self.root

class BS (maker.Maker):
  targets     = {}
  batchArgs   = 0
  directories = {}
  filesets    = {}

  def __init__(self, projectObj, clArgs = None):
    self.setupArgDB(clArgs)
    maker.Maker.__init__(self)
    self.project          = projectObj
    self.sourceDBFilename = os.path.join(projectObj.getRoot(), 'bsSource.db')
    self.setupSourceDB()
    # Put current project name into the database
    if not argDB.has_key('installedprojects'):
      argDB['installedprojects'] = []
    if projectObj not in argDB['installedprojects']:
      argDB['installedprojects'] = argDB['installedprojects']+[projectObj]
    return

  def setupArgDB(self, clArgs):
    global argDB
    argDB = nargs.ArgDict('ArgDict', clArgs)
    return

  def saveSourceDB(self):
    self.debugPrint('Saving source database in '+self.sourceDBFilename, 2, 'sourceDB')
    global sourceDB
    # Make sourceDB paths relative
    newDB = sourceDatabase.SourceDB()
    pwd   = self.getRoot()
    for key in sourceDB:
      new_key        = re.split(pwd,key)[-1]
      newDB[new_key] = sourceDB[key]
    sourceDB = newDB

    dbFile = open(self.sourceDBFilename, 'w')
    cPickle.dump(sourceDB, dbFile)
    dbFile.close()
    return
  
  def setupSourceDB(self):
    self.debugPrint('Reading source database from '+self.sourceDBFilename, 2, 'sourceDB')
    global sourceDB

    if os.path.exists(self.sourceDBFilename):
      dbFile   = open(self.sourceDBFilename, 'r')
      sourceDB = cPickle.load(dbFile)
      dbFile.close()

      # Make sourceDB paths absolute
      newDB = sourceDatabase.SourceDB()
      pwd   = self.getRoot()
      for key in sourceDB:
        new_key        = pwd+key
        newDB[new_key] = sourceDB[key]
      sourceDB = newDB
    else:
      sourceDB = sourceDatabase.SourceDB()
    atexit.register(self.cleanup)
    sourceDB.setFromArgs(argDB)
    if not int(argDB['restart']):
      for source in sourceDB:
        sourceDB.clearUpdateFlag(source)
    return

  def getSIDLDefaults(self):
    if not hasattr(self, 'sidlDefaults'):
      if not self.filesets.has_key('sidl'):
        self.filesets['sidl'] = None
      if self.filesets.has_key('bootstrap'):
        self.sidlDefaults = BSTemplates.sidlTargets.Defaults(self.project, self.filesets['sidl'], bootstrapPackages = self.filesets['bootstrap'])
      else:
        self.sidlDefaults = BSTemplates.sidlTargets.Defaults(self.project, self.filesets['sidl'])
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

  def t_sidl(self):
    return self.getSIDLDefaults().getSIDLTarget().execute()

  def t_compile(self):
    return self.getCompileDefaults().getCompileTarget().execute()

  def t_print(self):
    return self.getSIDLDefaults().getSIDLPrintTarget().execute()

  def t_default(self):
    return self.executeTarget('compile')

  def t_recalc(self):
    return sourceDB.calculateDependencies()

  def t_printTargets(self):
    targets = self.targets.keys()
    for attr in dir(self):
      if attr[0:2] == 't_':
        targets.append(attr[2:])
    print 'Available targets: '+str(targets)

  def t_printSourceDB(self):
    print sourceDB

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
          if sourceDB.has_key(file):
            self.debugPrint('Purging '+file, 3, 'sourceDB')
            del sourceDB[file]
      except KeyError:
        try:
          if sourceDB.has_key(setName):
            self.debugPrint('Purging '+setName, 3, 'sourceDB')
            del sourceDB[setName]
        except KeyError:
          print 'FileSet '+setName+' not found for purge'
    else:
      import re

      purgeRE = re.compile(argDB['regExp'])
      purges  = []
      for key in sourceDB:
        m = purgeRE.match(key)
        if m: purges.append(key)
      for source in purges:
        self.debugPrint('Purging '+source, 3, 'sourceDB')
        del sourceDB[source]

  def t_update(self):
    if argDB.has_key('fileset'):
      setName = argDB['fileset']
      try:
        self.debugPrint('Updating source database of fileset '+setName, 1, 'sourceDB')
        for file in self.filesets[setName]:
          self.debugPrint('Updating '+file, 3, 'sourceDB')
          sourceDB.updateSource(file)
      except KeyError:
        try:
          self.debugPrint('Updating '+setName, 3, 'sourceDB')
          sourceDB.updateSource(setName)
        except KeyError:
          print 'FileSet '+setName+' not found for update'
    else:
      import re

      print argDB['regExp']
      updateRE = re.compile(argDB['regExp'])
      updates  = []
      for key in sourceDB:
        m = updateRE.match(key)
        if m: updates.append(key)
      for source in updates:
        self.debugPrint('Updating '+source, 3, 'sourceDB')
        sourceDB.updateSource(source)

  def cleanup(self):
    self.saveSourceDB()
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
    try:
      import SIDL.Loader
      import SIDLLanguage.Parser
      import SIDLLanguage.Visitor

      compiler = SIDLLanguage.Parser.Parser(SIDL.Loader.createClass('ANL.SIDLCompilerI.SIDLCompiler'))
      if argDB.has_key('installedpackages'):
        ipackages = argDB['installedpackages']
      else: ipackages = []
      for source in self.filesets['sidl'].getFiles():
        tree = compiler.parseFile(source)
        v = SIDLLanguage.Visitor.Visitor(SIDL.Loader.createClass('ANL.SIDLVisitorI.GetPackageNames'))
        tree.accept(v)
        for p in v.getnames():
          if not p in ipackages:
            ipackages.append(p)
      argDB['installedpackages'] = ipackages
    except: pass

    try:
      if target is None:               target = argDB.target
      if not isinstance(target, list): target = [target]
      print target
      map(self.executeTarget, target)
    except Exception, e:
      print str(e)
      if not argDB.has_key('noStackTrace') or not int(argDB['noStackTrace']):
        print traceback.print_tb(sys.exc_info()[2])
    self.cleanupDir(self.tmpDir)
    return
