'''This module provides base functionality for all members of BuildSystem'''
import logging

import os
import sys

class Base(logging.LoggerOld):
  '''The Base class handles the argument database and shell commands'''
  defaultDB  = None
  defaultLog = None

  def __init__(self, clArgs = None, argDB = None):
    '''Setup the argument database'''
    self.argDB = self.createArgDB(argDB)
    self.setupArgDB(self.argDB, clArgs)
    logging.LoggerOld.__init__(self, self.argDB, self.createLog(None))
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
      # Work around a bug with pdb in 2.3
      if hasattr(sys.modules[self.__module__], '__file__') and not os.path.basename(sys.modules[self.__module__].__file__) == 'pdb.py':
        self._root_ = os.path.abspath(os.path.dirname(sys.modules[self.__module__].__file__))
      else:
        self._root_ = os.getcwd()
    return self._root_

  def defaultCheckCommand(self, command, status, output):
    '''Raise an error if the exit status is nonzero'''
    if status: raise RuntimeError('Could not execute \''+command+'\':\n'+output)

  def openPipe(self, command):
    '''We need to use the asynchronous version here since we want to avoid blocking reads'''
    import popen2

    pipe = None
    if hasattr(popen2, 'Popen3'):
      pipe   = popen2.Popen3(command, 1)
      input  = pipe.tochild
      output = pipe.fromchild
      err    = pipe.childerr
    else:
      (input, output, err) = os.popen3(command)
    return (input, output, err, pipe)

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

  def executeShellCommandSafely(self, command, checkCommand = None):
    '''If the command is interrupted by a password request, raise an exception instead of hanging'''
    import select

    self.debugPrint('sh: '+command, 3, 'shell')
    ret        = None
    out        = ''
    err        = ''
    loginError = 0
    (input, output, error, pipe) = self.openPipe(command)
    input.close()
    outputClosed = 0
    errorClosed  = 0
    while 1:
      ready = select.select([output, error], [], [])
      if len(ready[0]):
        if error in ready[0]:
          msg = error.read()
          if msg:
            err += msg
          else:
            errorClosed = 1
        if output in ready[0]:
          msg = output.read()
          if msg:
            out += msg
          else:
            outputClosed = 1
        if out.find('password:') >= 0 or err.find('password:') >= 0:
          loginError = 1
          break
      if outputClosed and errorClosed:
        break
    output.close()
    error.close()
    if pipe:
      # We would like the NOHANG argument here
      ret = pipe.wait()
    if loginError:
      raise RuntimeError('Could not login to site')
    if checkCommand:
      checkCommand(command, ret, out)
    else:
      self.defaultCheckCommand(command, ret, out)
    return out

  def getInstalledProject(self, url, returnAll = 0):
    if returnAll:
      projects = []
    else:
      projects = None
    for project in self.argDB['installedprojects']:
      if project.getUrl() == url:
        self.debugPrint('Project '+project.getUrl()+' is installed', 4, 'install')
        if not returnAll:
          return project
        else:
          projects.append(project)
    self.debugPrint('Project '+url+' is not installed', 4, 'install')
    return projects
