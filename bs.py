#!/usr/bin/env python
import args
import fileset
import logging

import atexit
import cPickle
import commands
import errno
import md5
import os
import os.path
import re
import statvfs
import string
import sys
import time
import types

class ChecksumError (RuntimeError):
  def __init__(self, value):
    self.value = value

  def __str__(self):
    return str(self.value)

class Maker (logging.Logger):
  """
  Base class for all build objects, which handles:
    - Temporary storage
    - Checksums
    - Shell commands
    - Source database update
    - Advanced debug output
  """
  def __init__(self, locArgDB = None):
    if not locArgDB: locArgDB = argDB
    logging.Logger.__init__(self, locArgDB)
    self.setupTmpDir()
    self.cleanupDir(self.tmpDir)
    self.setupChecksum()

  def checkTmpDir(self, mainTmp):
    if not os.path.exists(mainTmp):
      del argDB['TMPDIR']
      argDB.setHelp('TMPDIR', 'Temporary directory '+mainTmp+' does not exist. Select another directory')
      newTmp = argDB['TMPDIR']
      return 0
      
    stats     = os.statvfs(mainTmp)
    freeSpace = stats[statvfs.F_BAVAIL]*stats[statvfs.F_FRSIZE]
    if freeSpace < 50*1024*1024:
      del argDB['TMPDIR']
      argDB.setHelp('TMPDIR', 'Insufficient space ('+str(freeSpace/1024)+'K) on '+mainTmp+'. Select another directory')
      newTmp = argDB['TMPDIR']
      return 0
    return 1

  def setupTmpDir(self):
    mainTmp = argDB['TMPDIR']
    while not self.checkTmpDir(mainTmp):
      mainTmp = argDB['TMPDIR']
    self.tmpDir = os.path.join(mainTmp, 'bs')

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

  def setupChecksum(self):
    if (argDB['checksumType'] == 'md5'):
      self.getChecksum = self.getMD5Checksum
    elif (argDB['checksumType'] == 'bk'):
      self.getChecksum = self.getBKChecksum
    else:
      raise RuntimeError('Invalid checksum type: '+argDB['checksumType'])

  def getMD5Checksum(self, source):
    #output = self.executeShellCommand('md5sum --binary '+source, self.checkChecksumCall)
    #return string.split(output)[0]
    f = open(source, 'r')
    m = md5.new()
    size = 1024*1024
    buf  = f.read(size)
    while buf:
      m.update(buf)
      buf = f.read(size)
    f.close()
    return m.hexdigest()

  def checkChecksumCall(self, command, status, output):
    if (status): raise ChecksumError(output)

  def getBKChecksum(self, source):
    checksum = 0
    try:
      output   = self.executeShellCommand('bk checksum -s8 '+source, self.checkChecksumCall)
      checksum = string.split(output)[1]
    except ChecksumError:
      pass
    return checksum

  def defaultCheckCommand(self, command, status, output):
    if status: raise RuntimeError('Could not execute \''+command+'\': '+output)

  def executeShellCommand(self, command, checkCommand = None):
    self.debugPrint('sh: '+command, 3, 'shell')
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
    self.debugPrint('Updating '+source+' in source database', 3, 'sourceDB')
    sourceDB[source] = (self.getChecksum(source), os.path.getmtime(source), time.time(), dependencies)

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
      raise RuntimeError('Invalid fileset '+str(set))

class BS (Maker):
  includeRE   = re.compile(r'^#include (<|")(?P<includeFile>.+)\1')
  targets     = {}
  batchArgs   = 0
  directories = {}
  filesets    = {}

  def __init__(self, clArgs = None):
    self.setupArgDB(clArgs)
    Maker.__init__(self)
    for key in argDB.keys():
      self.debugPrint('Set '+key+' to '+str(argDB[key]), 3, 'argDB')
    self.sourceDBFilename = os.path.join(os.getcwd(), 'bsSource.db')
    self.setupSourceDB()

  def setupDefaultArgs(self):
    argDB.setDefault('target',      ['default'])
    argDB.setDefault('TMPDIR',      '/tmp')
    argDB.setDefault('checksumType', 'md5')

    argDB.setHelp('BS_DIR', 'The directory in which BS was installed')

  def setupArgDB(self, clArgs):
    global argDB

    argDB = args.ArgDict(os.path.join(os.getcwd(), 'bsArg.db'))
    self.setupDefaultArgs()
    argDB.input(clArgs)

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
    atexit.register(self.cleanup)

  def calculateDependencies(self):
    self.debugPrint('Recalculating dependencies', 1, 'sourceDB')
    for source in sourceDB.keys():
      self.debugPrint('Calculating '+source, 3, 'sourceDB')
      (checksum, mtime, timestamp, dependencies) = sourceDB[source]
      newDep = []
      try:
        file = open(source, 'r')
      except IOError, e:
        if e.errno == errno.ENOENT:
          del sourceDB[source]
        else:
          raise e
      comps  = string.split(source, '/')
      for line in file.readlines():
        m = self.includeRE.match(line)
        if m:
          filename  = m.group('includeFile')
          matchNum  = 0
          matchName = filename
          self.debugPrint('  Includes '+filename, 3, 'sourceDB')
          for s in sourceDB.keys():
            if string.find(s, filename) >= 0:
              self.debugPrint('    Checking '+s, 3, 'sourceDB')
              c = string.split(s, '/')
              for i in range(len(c)):
                if not comps[i] == c[i]: break
              if i > matchNum:
                self.debugPrint('    Choosing '+s+'('+str(i)+')', 3, 'sourceDB')
                matchName = s
                matchNum  = i
          newDep.append(matchName)
      # Grep for #include, then put these files in a tuple, we can be recursive later in a fixpoint algorithm
      sourceDB[source] = (checksum, mtime, timestamp, tuple(newDep))

  def debugDependencies(self):
    for source in sourceDB.keys():
      (checksum, mtime, timestamp, dependencies) = sourceDB[source]
      print source
      print '  Checksum:  '+str(checksum)
      print '  Mod Time:  '+str(mtime)
      print '  Timestamp: '+str(timestamp)
      print '  Deps: '+str(dependencies)

  def purge(self):
    if argDB.has_key('arg'):
      argNames = argDB['arg']
      if not type(argNames) == types.ListType: argNames = [argNames]
      for argName in argNames:
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
        try:
          if sourceDB.has_key(setName):
            self.debugPrint('Purging '+setName, 3, 'sourceDB')
            del sourceDB[setName]
        except KeyError:
          print 'FileSet '+setName+' not found for purge'

  def update(self):
    setName = argDB['fileset']
    try:
      self.debugPrint('Updating source database of fileset '+setName, 1, 'sourceDB')
      for file in self.filesets[setName]:
        if sourceDB.has_key(file):
          self.debugPrint('Updating '+file, 3, 'sourceDB')
          self.updateSourceDB(file)
    except KeyError:
      try:
        self.debugPrint('Updating '+setName, 3, 'sourceDB')
        self.updateSourceDB(setName)
      except KeyError:
        print 'FileSet '+setName+' not found for update'

  def cleanup(self):
    if argDB.has_key('target'): del argDB['target']
    self.saveSourceDB()

  def main(self):
    if argDB.has_key('target'):
      for target in argDB['target']:
        if self.targets.has_key(target):
          self.targets[target].execute()
        elif target == 'listTargets':
          print 'Available targets: '+str(self.targets.keys())
        elif target == 'purge':
          self.purge()
        elif target == 'update':
          self.update()
        elif target == 'recalc':
          self.calculateDependencies()
        elif target == 'debugDep':
          self.debugDependencies()
        else:
          print 'Invalid target: '+target
    self.cleanupDir(self.tmpDir)
