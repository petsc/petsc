#!/usr/bin/env python
import fileset

import atexit
import cPickle
import commands
import os
import os.path
import string
import sys
import traceback
import types

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
      self.tmpDir = os.path.join(os.environ['TMPDIR'], 'bs')
    except KeyError:
      if (os.path.exists('/tmp')):
        self.tmpDir = os.path.join('/tmp', 'bs')
      else:
        raise RuntimeError('Please set the TMPDIR variable')

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
    else:
      output = '['
      for fs in set:
        output += self.debugFileSetStr(fs)
      return output+']'

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

class BS (Maker):
  def __init__(self, args = None):
    self.targets = {}
    if args: self.processArgs(args)
    Maker.__init__(self)
    self.sourceDBFilename = os.path.join(os.getcwd(), 'bsSource.db')
    self.setupSourceDB()
    for key in self.args.keys():
      self.debugPrint('Set '+key+' to '+self.args[key])

  def parseArgs(self, argList):
    args = {}
    args['target'] = 'default'
    for arg in argList:
      if not arg[0] == '-':
        args['target'] = arg
      else:
        (key, val) = string.split(arg[1:], '=')
        args[key]  = val
    return args

  def processArgs(self, args):
    global debugLevel
    global debugSections

    if type(args) == types.ListType:
      args = self.parseArgs(args)
    # Here we could have an object BSOptions which had nothing but member
    # variables. Then we could use reflection to get all the names of the
    # variables, and look them up in args[]
    try:
      debugLevel    = int(args['debugLevel'])
      debugSections = string.split(args['debugSections'])
    except KeyError: pass
    self.args = args

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

  def main(self):
    if self.targets.has_key(self.args['target']):
      self.targets[self.args['target']].execute()
    else:
      print 'Invalid target: '+self.args['target']
    self.cleanupDir(self.tmpDir)
