#!/usr/bin/env python
import fileset

import atexit
import cPickle
import commands
import os
import os.path
import re
import string
import sys
import time
import traceback
import types
import UserDict

# Debugging
debugLevel    = 1
debugSections = []

class Maker:
  def __init__(self):
    self.setupTmpDir()
    self.cleanupDir(self.tmpDir)
    self.debugLevel    = debugLevel
    self.debugSections = list(debugSections)
    self.debugIndent   = '  '

  def setupTmpDir(self):
    try:
      if not os.path.exists(argDB['TMPDIR']): raise KeyError('TMPDIR')
      self.tmpDir = os.path.join(argDB['TMPDIR'], 'bs')
    except KeyError:
      raise RuntimeError('Please set the TMPDIR argument')

  def forceRemove(self, file):
    if (os.path.exists(file)):
      if (os.path.isdir(file)):
        for f in os.listdir(file):
          self.forceRemove(os.path.join(file, f))
        os.rmdir(file)
      else:
        os.remove(file)
    
  def cleanupDir(self, dir, remove = 0):
    if not os.path.exists(dir): os.makedirs(dir)
    oldDir = os.getcwd()
    os.chdir(dir)
    map(self.forceRemove, os.listdir(dir))
    os.chdir(oldDir)
    if remove: os.rmdir(dir)

  def debugListStr(self, list):
    if (self.debugLevel > 4) or (len(list) < 4):
      return str(list)
    else:
      return '['+str(list[0])+'-<'+str(len(list)-2)+'>-'+str(list[-1])+']'

  def debugFileSetStr(self, set):
    if isinstance(set, fileset.FileSet):
      if set.tag:
        return '('+set.tag+')'+self.debugListStr(set.getFiles())
      else:
        return self.debugListStr(set.getFiles())
    elif type(set) == types.ListType:
      output = '['
      for fs in set:
        output += self.debugFileSetStr(fs)
      return output+']'
    else:
      raise RuntimeError('Invalid fileset '+set)

  def debugPrint(self, msg, level = 1, section = None):
    indentLevel = len(traceback.extract_stack())-4
    if self.debugLevel >= level:
      if (not section) or (not self.debugSections) or (section in self.debugSections):
        for i in range(indentLevel):
          sys.stdout.write(self.debugIndent)
        print msg

  def defaultCheckCommand(self, command, status, output):
    if status: raise RuntimeError('Could not execute \''+command+'\': '+output)

  def executeShellCommand(self, command, checkCommand = None):
    self.debugPrint('sh: '+command, 4, 'shell')
    (status, output) = commands.getstatusoutput(command)
    if checkCommand:
      checkCommand(command, status, output)
    else:
      self.defaultCheckCommand(command, status, output)
    return output

  def updateSourceDB(self, source):
    dependencies = ()
    try:
      (checksum, mtime, timestamp, dependencies) = sourceDB[source]
    except KeyError:
      pass
    sourceDB[source] = (self.getChecksum(source), os.path.getmtime(source), time.time(), dependencies)

class BS (Maker):
  includeRE   = re.compile(r'^#include (<|")(?P<includeFile>.+)\1')
  targets     = {}
  args        = {}
  defaultArgs = {}
  directories = {}
  filesets    = {}

  def __init__(self, args = None):
    self.argDBFilename = os.path.join(os.getcwd(), 'bsArg.db')
    self.setupArgDB()
    self.defaultArgs['target'] =  'default'
    self.defaultArgs['TMPDIR'] =  '/tmp'
    self.processArgs(args)
    Maker.__init__(self)
    self.debugPrint('Read source database from '+self.argDBFilename, 2, 'argDB')
    self.sourceDBFilename = os.path.join(os.getcwd(), 'bsSource.db')
    self.setupSourceDB()
    for key in argDB.keys():
      self.debugPrint('Set '+key+' to '+str(argDB[key]), 3, 'argDB')

  def processArgs(self, args):
    global argDB
    global debugLevel
    global debugSections

    for key in self.defaultArgs.keys():
      if not argDB.has_key(key): argDB[key] = self.defaultArgs[key]
    argDB.update(os.environ)
    argDB.update(self.parseArgs(args))
    # Here we could have an object BSOptions which had nothing but member
    # variables. Then we could use reflection to get all the names of the
    # variables, and look them up in args[]
    try:
      debugLevel    = int(argDB['debugLevel'])
      debugSections = argDB['debugSections']
    except KeyError: pass

  def parseArgs(self, argList):
    if not type(argList) == types.ListType: return
    args = {}
    for arg in argList:
      if not arg[0] == '-':
        args['target'] = arg
      else:
        # Could try just using eval() on val, but we would need to quote lots of stuff
        (key, val) = string.split(arg[1:], '=')
        if val[0] == '[' and val[-1] == ']': val = string.split(val[1:-1], ',')
        args[key]  = val
    return args

  def saveArgDB(self):
    self.debugPrint('Saving argument database in '+self.argDBFilename, 2, 'argDB')
    dbFile = open(self.argDBFilename, 'w')
    cPickle.dump(argDB.data, dbFile)
    dbFile.close()

  def setupArgDB(self):
    global argDB

    argDB = ArgDict()
    if os.path.exists(self.argDBFilename):
      dbFile     = open(self.argDBFilename, 'r')
      argDB.data = cPickle.load(dbFile)
      dbFile.close()
    atexit.register(self.saveArgDB)

  def saveSourceDB(self):
    self.debugPrint('Saving source database in '+self.sourceDBFilename, 2, 'sourceDB')
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
    else:
      sourceDB = {}
    atexit.register(self.saveSourceDB)

  def calculateDependencies(self):
    self.debugPrint('Recalculating dependencies', 1, 'sourceDB')
    for source in sourceDB.keys():
      (checksum, mtime, timestamp, dependencies) = sourceDB[source]
      newDep = []
      file   = open(source, 'r')
      for line in file.readlines():
        m = self.includeRE.match(line)
        if m:
          filename = m.group('includeFile')
          for s in sourceDB.keys():
            if string.find(s, filename) >= 0:
              filename = s
              break
          newDep.append(filename)
      # Grep for #include, then put these files in a tuple, we can be recursive later in a fixpoint algorithm
      sourceDB[source] = (checksum, mtime, timestamp, tuple(newDep))

  def purge(self):
    if argDB.has_key('arg'):
      argName = argDB['arg']
      if argDB.has_key(argName):
        self.debugPrint('Purging '+argName, 3, 'argDB')
        del argDB[argName]
      del argDB['arg']
    else:
      setName = argDB['fileset']
      try:
        self.debugPrint('Purging source database of fileset '+setName, 1, 'sourceDB')
        for file in self.filesets[setName]:
          if sourceDB.has_key(file):
            self.debugPrint('Purging '+file, 3, 'sourceDB')
            del sourceDB[file]
      except KeyError:
        print 'FileSet '+setName+' not found for purge'

  def checkDirectory(self, dirname):
    if not os.path.isdir(self.directories[dirname]):
      raise RuntimeError('Directory '+dirname+' ==> '+self.directories[dirname]+' does not exist')

  def main(self):
    map(self.checkDirectory, self.directories.keys())
    if argDB.has_key('target'):
      if self.targets.has_key(argDB['target']):
        self.targets[argDB['target']].execute()
      elif argDB['target'] == 'recalc':
        self.calculateDependencies()
      elif argDB['target'] == 'purge':
        self.purge()
      else:
        print 'Invalid target: '+argDB['target']
    if argDB.has_key('target'): del argDB['target']
    self.cleanupDir(self.tmpDir)

class ArgDict (UserDict.UserDict):
  def __getitem__(self, key):
    if not self.data.has_key(key):
      try:
        self[key] = raw_input('Please enter value for '+key+':')
      except KeyboardInterrupt:
        print
        sys.exit('Unable to get argument \''+key+'\'')
    return self.data[key]
