'''This module provides base functionality for all members of BuildSystem'''
import logging

import os
import sys

class Base(logging.Logger):
  '''The Base class handles the argument database and shell commands'''
  defaultDB  = None
  defaultLog = None

  def __init__(self, clArgs = None, argDB = None):
    '''Setup the argument database'''
    self.argDB = self.createArgDB(argDB)
    self.setupArgDB(self.argDB, clArgs)
    logging.Logger.__init__(self, self.argDB, self.createLog(None))
    self.getRoot()
    return

  def __getstate__(self):
    '''We do not want to pickle the default RDict or log stream'''
    d = self.__dict__.copy()
    if 'argDB' in d and d['argDB'] is Base.defaultDB:
      del d['argDB']
    if 'log' in d and d['log'] is Base.defaultLog:
      del d['log']
    return d

  def __setstate__(self, d):
    '''We must create the default RDict and log stream'''
    if not 'argDB' in d:
      self.argDB = self.createArgDB(None)
    if not 'log' in d:
      self.log   = self.createLog(None)
    self.__dict__.update(d)
    return

  def createArgDB(self, initDB):
    '''Create an argument database unless initDB is provided, and insert the command line arguments'''
    if not initDB is None:
      argDB = initDB
    else:
      if Base.defaultDB is None:
        import RDict
        import os
        Base.defaultDB = RDict.RDict(parentDirectory = os.path.dirname(os.path.abspath(sys.modules['RDict'].__file__)))
      argDB = Base.defaultDB
    return argDB

  def setupArgDB(self, argDB, clArgs):
    '''Setup types in the argument database'''
    if Base.defaultLog is None:
      import nargs

      argDB.setType('debugLevel',    nargs.ArgInt(None, None, 'Integer 0 to 4, where a higher level means more detail', 0, 5))
      argDB.setType('debugSections', nargs.Arg(None, None, 'Message types to print, e.g. [compile,link,bk,install]'))

    argDB.insertArgs(clArgs)
    return argDB

  def createLog(self, initLog):
    '''Create a default log stream, unless initLog is given, which accepts all messages'''
    if not initLog is None:
      log = initLog
    else:
      if Base.defaultLog is None:
        Base.defaultLog = file('make.log', 'w')
      log = Base.defaultLog
    return log

  def getRoot(self):
    '''Return the directory containing this module
       - This has the problem that when we reload a module of the same name, this gets screwed up
         Therefore, we call it in the initializer, and stash it'''
    if not hasattr(self, '_root_'):
      if hasattr(sys.modules[self.__module__], '__file__'):
        self._root_ = os.path.abspath(os.path.dirname(sys.modules[self.__module__].__file__))
      else:
        self._root_ = os.getcwd()
    return self._root_

  def defaultCheckCommand(self, command, status, output):
    '''Raise an error if the exit status is nonzero'''
    if status: raise RuntimeError('Could not execute \''+command+'\':\n'+output)

  def executeShellCommand(self, command, checkCommand = None):
    '''Execute a shell command returning the output, and optionally provide a custom error checker'''
    import commands

    self.debugPrint('sh: '+command, 3, 'shell')
    (status, output) = commands.getstatusoutput(command)
    self.debugPrint('sh: '+output, 4, 'shell')
    if checkCommand:
      checkCommand(command, status, output)
    else:
      self.defaultCheckCommand(command, status, output)
    return output

  def getInstalledProject(self, url, returnAll = 0):
    if returnAll:
      projects = []
    else:
      projects = None
    for project in self.argDB['installedprojects']:
      if project.getUrl() == url:
        self.debugPrint('Project '+project.getUrl()+'('+url+') is installed', 4, 'install')
        if not returnAll:
          return project
        else:
          projects.append(project)
    self.debugPrint('Project '+url+' is not installed', 4, 'install')
    return projects
