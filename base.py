'''This module provides base functionality for all members of BuildSystem'''
import logging

import os
import sys

class Base(logging.Logger):
  '''The Base class handles the argument database and shell commands'''
  defaultDB = None

  def __init__(self, clArgs = None, argDB = None):
    '''Setup the argument database'''
    self.argDB = self.createArgDB(argDB)
    # SHOULD BE JUST self.setupArgDB(self.argDB) here, but this way preserves logging.Logger for now
    logging.Logger.__init__(self, self.argDB)
    self.setupArgDB(self.argDB, clArgs)

    self.getRoot()
    return

  def __setstate__(self, d):
    '''Assume that we do not want to unpickle custom RDict objects'''
    # Should put in some swtich like self.saveArgDB
    if 'argDB' in d: del d['argDB']
    self.argDB = self.createArgDB(None)
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
    logging.Logger.setFromArgs(self, argDB)
    argDB.insertArgs(clArgs)
    return argDB

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
        self.debugPrint('Project '+project.getName()+'('+url+') is installed', 3, 'install')
        if not returnAll:
          return project
        else:
          projects.append(project)
    self.debugPrint('Project '+url+' is not installed', 3, 'install')
    return projects
