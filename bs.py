#!/usr/bin/env python

# Debugging
debugLevel    = 1
debugSections = []

class Maker:
  def __init__(self):
    self.setupTmpDir()
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
    
  def cleanupTmpDir(self):
    if not os.path.exists(self.tmpDir): os.makedirs(self.tmpDir)
    oldDir = os.getcwd()
    os.chdir(self.tmpDir)
    map(self.forceRemove, os.listdir(self.tmpDir))
    os.chdir(oldDir)

  def debugListStr(self, list):
    if (self.debugLevel > 1) or (len(list) < 4):
      return str(list)
    else:
      return '['+str(list[0])+'-<'+str(len(list)-2)+'>-'+str(list[-1])+']'

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
  sourceDBFilename = os.path.join(os.getcwd(), 'bsSource.db')

  def saveSourceDB():
    self.debugPrint('Saving source database in '+sourceDBFilename, 2, 'sourceDB')
    dbFile = open(sourceDBFilename, 'w')
    cPickle.dump(sourceDB, dbFile)
    dbFile.close()

  def main():
    self.debugPrint('Reading source database from '+sourceDBFilename, 2, 'sourceDB')
    global sourceDB

    if os.path.exists(sourceDBFilename):
      dbFile   = open(sourceDBFilename, 'r')
      sourceDB = cPickle.load(dbFile)
      dbFile.close()
    else:
      sourceDB = {}
    atexit.register(saveSourceDB)
