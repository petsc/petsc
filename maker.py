import bs
import fileset
import logging
import nargs

import commands
import os
import string
import pwd

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
  def __init__(self, locArgDB = None):
    if not locArgDB: locArgDB = bs.argDB
    logging.Logger.__init__(self, locArgDB)
    self.setupTmpDir()
    self.cleanupDir(self.tmpDir)

  def checkTmpDir(self, mainTmp):
    if not os.path.exists(mainTmp):
      del bs.argDB['TMPDIR']
      bs.argDB.setType('TMPDIR', nargs.ArgDir(1,'Temporary directory '+mainTmp+' does not exist. Select another directory'))
      newTmp = bs.argDB['TMPDIR']
      return 0
      
    try:
      stats     = os.statvfs(mainTmp)
      freeSpace = stats.f_bavail*stats.f_frsize
      if freeSpace < 50*1024*1024:
        del bs.argDB['TMPDIR']
        bs.argDB.setType('TMPDIR', nargs.ArgDir(1,'Insufficient space ('+str(freeSpace/1024)+'K) on '+mainTmp+'. Select another directory'))
        newTmp = bs.argDB['TMPDIR']
        return 0
    except: pass
    return 1

  def setupTmpDir(self):
        
    #  get tmp directory; needs to be different for each user
    if not bs.argDB.has_key('TMPDIR') and os.environ.has_key('TMPDIR'):
      bs.argDB['TMPDIR'] = os.environ['TMPDIR']

    if not bs.argDB.has_key('TMPDIR') or bs.argDB['TMPDIR'] == '/tmp':
      bs.argDB['TMPDIR'] = os.path.join('/tmp', pwd.getpwuid(os.getuid())[0])
    if not os.path.exists(bs.argDB['TMPDIR']):
      try:
        os.makedirs(bs.argDB['TMPDIR'])
      except:
        raise RuntimeError("Cannot create tmp directory "+bs.argDB['TMPDIR'])

    mainTmp = bs.argDB['TMPDIR']
    while not self.checkTmpDir(mainTmp):
      mainTmp = bs.argDB['TMPDIR']
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
