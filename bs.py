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

class BS (maker.Maker):
  targets     = {}
  batchArgs   = 0
  directories = {}
  filesets    = {}

  def __init__(self, project, clArgs = None):
    self.setupArgDB(clArgs)
    maker.Maker.__init__(self)
    self.project = project
    self.sourceDBFilename = os.path.join(os.getcwd(), 'bsSource.db')
    self.setupSourceDB()
    return

  def setupArgDB(self, clArgs):
    global argDB
    argDB = nargs.ArgDict('ArgDict', clArgs)
    # put current package name into the database
    package = os.path.basename(os.getcwd())
    if argDB.has_key('installedpackages'):
      packages = argDB['installedpackages']
      if package not in packages:
        packages.append(package)
        argDB['installedpackages'] = packages
    else:
      argDB['installedpackages'] = [package]

  def saveSourceDB(self):
    self.debugPrint('Saving source database in '+self.sourceDBFilename, 2, 'sourceDB')
    global sourceDB
    newDB = sourceDatabase.SourceDB()
    pwd = os.getcwd()
    for key in sourceDB:
      new_key = re.split(pwd,key)[-1]
      newDB[new_key] = sourceDB[key]
    sourceDB = newDB

    dbFile = open(self.sourceDBFilename, 'w')
    cPickle.dump(sourceDB, dbFile)
    dbFile.close()
  
  def setupSourceDB(self):
    self.debugPrint('Reading source database from '+self.sourceDBFilename, 2, 'sourceDB')
    global sourceDB

    if os.path.exists(self.sourceDBFilename):
      dbFile   = open(self.sourceDBFilename, 'r')
      sourceDB = cPickle.load(dbFile)
      dbFile.close()

      newDB = sourceDatabase.SourceDB()
      pwd = os.getcwd()
      for key in sourceDB:
        new_key = pwd+key
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
      self.sidlDefaults = BSTemplates.sidlTargets.Defaults(self.project, self.filesets['sidl'])
    return self.sidlDefaults

  def getCompileDefaults(self):
    if not hasattr(self, 'compileDefaults'):
      if not self.filesets.has_key('etags'):
        self.filesets['etags'] = None
      self.compileDefaults = BSTemplates.compileTargets.Defaults(self.getSIDLDefaults(), self.filesets['etags'])
    return self.compileDefaults

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

  def executeTarget(self, target):
    if self.targets.has_key(target):
      self.targets[target].execute()
    elif hasattr(self, 't_'+target):
      getattr(self, 't_'+target)()
    else:
      print 'Invalid target: '+target
    return

  def main(self):
    try:
      print argDB.target
      map(self.executeTarget, argDB.target)
    except Exception, e:
      print str(e)
      if not argDB.has_key('noStackTrace') or not int(argDB['noStackTrace']):
        print traceback.print_tb(sys.exc_info()[2])
    self.cleanupDir(self.tmpDir)
