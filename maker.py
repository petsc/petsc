import bs
import fileset
import logging

import commands
import os
import string

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
      bs.argDB.setHelp('TMPDIR', 'Temporary directory '+mainTmp+' does not exist. Select another directory')
      newTmp = bs.argDB['TMPDIR']
      return 0
      
    stats     = os.statvfs(mainTmp)
    freeSpace = stats.f_bavail*stats.f_frsize
    if freeSpace < 50*1024*1024:
      del bs.argDB['TMPDIR']
      bs.argDB.setHelp('TMPDIR', 'Insufficient space ('+str(freeSpace/1024)+'K) on '+mainTmp+'. Select another directory')
      newTmp = bs.argDB['TMPDIR']
      return 0
    return 1

  def setupTmpDir(self):
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
