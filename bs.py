#!/usr/bin/env python
import args
import argtest
import maker
import sourceDatabase

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

  def __init__(self, clArgs = None):
    self.setupArgDB(clArgs)
    maker.Maker.__init__(self)
    for key in argDB.keys():
      self.debugPrint('Set '+key+' to '+str(argDB[key]), 3, 'argDB')
    self.sourceDBFilename = os.path.join(os.getcwd(), 'bsSource.db')
    self.setupSourceDB()
    self.setupDefaultTargets()

  def setupDefaultArgs(self):
    argDB.setDefault('target',      ['default'])
    argDB.setDefault('TMPDIR',      '/tmp')
    argDB.setDefault('checksumType', 'md5')

    argDB.setHelp('BS_DIR', 'The directory in which BS was installed')

  def setupArgDB(self, clArgs):
    global argDB
    filename = 'bsArg.db'
    parent   = 'bs'

    if os.path.exists(filename):
      f     = file(filename)
      argDB = cPickle.load(f)
      f.close()
    else:
      argDB = args.ArgDict(os.path.join(os.getcwd(), filename), parent)
    self.setupDefaultArgs()
    argDB.input(clArgs)
    return argDB

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
    argDB.setTester('restart',argtest.IntTester())
    if not int(argDB['restart']):
      for source in sourceDB:
        sourceDB.clearUpdateFlag(source)

  def setupDefaultTargets(self):
    import transform
    import target
    
    self.targets['recalc'] = target.Target(None, transform.SimpleFunction(sourceDB.calculateDependencies))

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

  def cleanup(self):
    if argDB.has_key('target'):  del argDB['target']
    if argDB.has_key('restart'): argDB['restart'] = '0'
    self.saveSourceDB()

  def main(self):
    try:
      if argDB.has_key('target'):
        for target in argDB['target']:
          if self.targets.has_key(target):
            self.targets[target].execute()
          elif hasattr(self, 't_'+target):
            getattr(self, 't_'+target)()
          else:
            print 'Invalid target: '+target
    except Exception, e:
      print str(e)
      if not argDB.has_key('noStackTrace') or not int(argDB['noStackTrace']):
        print traceback.print_tb(sys.exc_info()[2])
    self.cleanupDir(self.tmpDir)
