import fileset
import logging
import nargs

import commands
import os
import string
import sys
import tempfile

class ChecksumError (RuntimeError):
  def __init__(self, value):
    self.value = value

  def __str__(self):
    return str(self.value)

class Maker (logging.Logger):
  """
  Base class for all build objects, which handles:
    - Temporary storage
    - BK Checksums
    - Shell commands
    - Advanced debug output
  """
  def __init__(self, argDB = None):
    if argDB is None:
      import bs
      self.argDB = bs.argDB
    else:
      self.argDB = argDB
    logging.Logger.__init__(self, self.argDB)
    self.getRoot()
    self.setupTmpDir()
    self.cleanupDir(self.tmpDir)
    return

  def getRoot(self):
    # This has the problem that when we reload a module of the same name, this gets screwed up
    #   Therefore, we call it in the initializer, and stash it
    if not hasattr(self, '_root_'):
      if hasattr(sys.modules[self.__module__], '__file__'):
        self._root_ = os.path.abspath(os.path.dirname(sys.modules[self.__module__].__file__))
      else:
        self._root_ = os.getcwd()
    return self._root_

  def checkTmpDir(self, mainTmp):
    if not os.path.exists(mainTmp):
      del self.argDB['TMPDIR']
      self.argDB.setType('TMPDIR', nargs.ArgDir(1,'Temporary directory '+mainTmp+' does not exist. Select another directory'))
      newTmp = self.argDB['TMPDIR']
      return 0
      
    try:
      stats     = os.statvfs(mainTmp)
      freeSpace = stats.f_bavail*stats.f_frsize
      if freeSpace < 50*1024*1024:
        del self.argDB['TMPDIR']
        self.argDB.setType('TMPDIR', nargs.ArgDir(1,'Insufficient space ('+str(freeSpace/1024)+'K) on '+mainTmp+'. Select another directory'))
        newTmp = self.argDB['TMPDIR']
        return 0
    except: pass
    return 1

  def setupTmpDir(self):
        
    #  get tmp directory; needs to be different for each user
    if not self.argDB.has_key('TMPDIR') and os.environ.has_key('TMPDIR'):
      self.argDB['TMPDIR'] = os.environ['TMPDIR']

    if not self.argDB.has_key('TMPDIR') or self.argDB['TMPDIR'] == '/tmp':
      import getpass

      self.argDB['TMPDIR'] = os.path.join('/tmp', getpass.getuser())
    if not os.path.exists(self.argDB['TMPDIR']):
      try:
        os.makedirs(self.argDB['TMPDIR'])
      except:
        raise RuntimeError("Cannot create tmp directory "+self.argDB['TMPDIR'])

    mainTmp = self.argDB['TMPDIR']
    while not self.checkTmpDir(mainTmp):
      mainTmp = self.argDB['TMPDIR']
    self.tmpDir = os.path.join(mainTmp, 'bs')
    tempfile.tempdir = self.tmpDir
    return

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
    try:
      os.chdir(oldDir)
    except OSError, e:
      print 'ERROR: '+str(e)
    if remove: os.rmdir(dir)

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
    if status: raise RuntimeError('Could not execute \''+command+'\':\n'+output)

  def executeShellCommand(self, command, checkCommand = None):
    self.debugPrint('sh: '+command, 3, 'shell')
    (status, output) = commands.getstatusoutput(command)
    self.debugPrint('sh: '+output, 4, 'shell')
    if checkCommand:
      checkCommand(command, status, output)
    else:
      self.defaultCheckCommand(command, status, output)
    return output

  def debugFileSetStr(self, set):
    if isinstance(set, fileset.FileSet):
      if set.tag:
        return '('+set.tag+')'+self.debugListStr(set.getFiles())
      else:
        return self.debugListStr(set.getFiles())
    elif isinstance(set, list):
      output = '['
      for fs in set:
        output += self.debugFileSetStr(fs)
      return output+']'
    else:
      raise RuntimeError('Invalid fileset '+str(set))

  def guessProject(self, dir):
    for project in self.argDB['installedprojects']:
      if project.getRoot() == dir:
        return project
    return bs.Project(os.path.basename(dir).lower(), '')

  def getMakeModule(self, root, name = 'make'):
    import imp

    (fp, pathname, description) = imp.find_module(name, [root])
    try:
      return imp.load_module(name, fp, pathname, description)
    finally:
      if fp: fp.close()
