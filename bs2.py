#!/usr/bin/env python
import atexit
import commands
import cPickle
import os.path
import string
import sys
import time
import traceback
import types

debugLevel = 1

sourceDB = {}

class Maker:
  def __init__(self):
    self.setupTmpDir()
    self.debugLevel       = debugLevel
    self.debugIndentLevel = 0
    self.debugIndent      = '  '

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

  def defaultCheckCommand(self, command, status, output):
    if status: raise RuntimeError('Could not execute \''+command+'\': '+output)

  def executeShellCommand(self, command, checkCommand = None):
    self.debugPrint('sh: '+command, 4)
    (status, output) = commands.getstatusoutput(command)
    if checkCommand:
      checkCommand(command, status, output)
    else:
      self.defaultCheckCommand(command, status, output)
    return output

  def debugListStr(self, list):
    if (self.debugLevel > 1) or (len(list) < 4):
      return str(list)
    else:
      return '['+str(list[0])+'-<'+str(len(list)-2)+'>-'+str(list[-1])+']'

  def debugPrint(self, msg, level = 1):
    #indentLevel = self.debugIndentLevel
    indentLevel = len(traceback.extract_stack())-4
    if self.debugLevel >= level:
      for i in range(indentLevel):
        sys.stdout.write(self.debugIndent)
      print msg

class FileGroup (Maker):
  def __init__(self, data = [], func = None, children = []):
    Maker.__init__(self)
    self.data     = data[:]
    self.func     = func
    self.children = children[:]

  def __len__(self):
    return len(self.getFiles())

  def append(self, item):
    if not item in self.data:
      self.data.append(item)

  def extend(self, group):
    for item in group.getFiles():
      self.append(item)

  def remove(self, item):
    self.data.remove(item)

  def getFiles(self):
    funcData = []
    if (self.func):
      funcData = self.func(self)
    childData = []
    for child in self.children:
      childData += child.getFiles()
    return self.data+funcData+childData

class TreeFileGroup (FileGroup):
  def __init__(self, root = '.', fileTest = None):
    FileGroup.__init__(self, func = self.walkTree)
    self.root       = root
    if (fileTest):
      self.fileTest = fileTest
    else:
      self.fileTest = self.defaultTest

  def walkTree(self, fileGroup):
    files = []
    os.path.walk(self.root, self.fileTest, files)
    return files

  def defaultTest(self, defaultFiles, directory, fileList):
    if (os.path.basename(directory) == 'SCCS'): return
    for file in fileList:
      if (os.path.isdir(os.path.join(directory, file))): continue
      defaultFiles.append(os.path.join(directory, file))

class ExtensionFileGroup (TreeFileGroup):
  def __init__(self, root, ext):
    TreeFileGroup.__init__(self, root, self.extTest)
    self.ext = ext

  def extTest(self, extFiles, directory, fileList):
    if (os.path.basename(directory) == 'SCCS'): return
    for file in fileList:
      if (os.path.isdir(os.path.join(directory, file))): continue
      (base, ext) = os.path.splitext(file)
      if (ext == self.ext): extFiles.append(os.path.join(directory, file))

class Transform (Maker):
  "Transforms map one FileGroup into another. By default, this map is the identity"
  def __init__(self, sources = FileGroup(), fileFilter = lambda file: 1):
    Maker.__init__(self)
    self.sources    = sources
    self.products   = self.sources
    self.fileFilter = fileFilter

  def checkChecksum(self, command, status, output):
    if (status):
      self.checksumError = 1
    else:
      self.checksumError = 0

  def getChecksum(self, source):
    output = self.executeShellCommand('bk checksum -s8 '+source, self.checkChecksum)
    if self.checksumError:
      return 0
    else:
      return string.split(output)[1]

  def getObjectName(self, source):
    (dir, file) = os.path.split(source)
    (base, ext) = os.path.splitext(file)
    checksum    = self.getChecksum(source)
    return os.path.join(self.tmpDir, string.replace(dir, '/', '_')+'_'+base+'_'+str(checksum)+'.o')

  def updateSourceDB(self, source):
    sourceDB[source] = (self.getChecksum(source), os.path.getmtime(source), time.time());

  def execute(self):
    return self.products

class FileCompare (Transform):
  "This class maps sources into the set of files for which self.compare(target, source) returns true"
  def __init__(self, targets, sources = FileGroup(), fileFilter = lambda file: 1, returnAll = 0):
    Transform.__init__(self, sources, fileFilter)
    self.targets   = targets
    self.returnAll = returnAll

  def compare(self, target, source): return 1

  def singleTargetExecute(self, target):
    files = self.sources.getFiles()
    for source in files:
      if (not os.path.exists(source)):
        self.debugPrint(source+' does not exist')
        self.products.append(source)
      else:
        if (self.compare(target, source)):
          self.products.append(source)
    if (self.returnAll and len(self.products)):
      self.products = self.sources
    return self.products

  def execute(self):
    self.debugPrint('Comparing targets '+str(self.targets.getFiles())+' to sources '+str(self.sources.getFiles()), 2)
    self.products = FileGroup()
    files = self.targets.getFiles()
    if (not files):
      self.products = self.sources
    else:
      for target in files:
        if (not os.path.exists(target)):
          self.products = self.sources
          break
        self.singleTargetExecute(target)
    return self.products

class OlderThan (FileCompare):
  "This class maps sources into the set of files which are older than target"
  def __init__(self, targets, sources = FileGroup(), fileFilter = lambda file: 1, returnAll = 0):
    FileCompare.__init__(self, targets, sources, fileFilter, returnAll)

  def compare(self, target, source):
    self.debugPrint('Checking if '+source+' is older than '+target, 3)
    targetTime = os.path.getmtime(target)
    sourceTime = os.path.getmtime(source)
    if (targetTime > sourceTime):
      self.debugPrint(source+' is older than '+target, 3)
      return 1
    else:
      return 0

class NewerThan (FileCompare):
  "This class maps sources into the set of files which are newer than target"
  def __init__(self, targets, sources = FileGroup(), fileFilter = lambda file: 1, returnAll = 0):
    FileCompare.__init__(self, targets, sources, fileFilter, returnAll)

  def compare(self, target, source):
    self.debugPrint('Checking if '+source+' is newer than '+target, 3)
    targetTime = os.path.getmtime(target)
    sourceTime = os.path.getmtime(source)
    if (targetTime < sourceTime):
      self.debugPrint(source+' is newer than '+target, 3)
      return 1
    else:
      return 0

class NewerThanLibraryObject (FileCompare):
  "This class maps sources into the set of files which are newer than the corresponding object in the target library"
  def __init__(self, targets, sources = FileGroup(), fileFilter = lambda file: 1, returnAll = 0):
    FileCompare.__init__(self, targets, sources, fileFilter, returnAll)

  def getObjectTimestamp(self, object, library):
    output = self.executeShellCommand('ar tv '+library+' '+object)
    if (output[0:8] != 'no entry'):
      objectTime = time.mktime(time.strptime(output[25:42], "%b %d %H:%M %Y"))
    else:
      self.debugPrint('No entry for object '+object+' in '+library, 3)
      objectTime = 0
    return objectTime

  def compare(self, target, source):
    object     = self.getObjectName(source)
    self.debugPrint('Checking if '+source+' is newer than '+target+'('+object+')', 3)
    objectTime = self.getObjectTimestamp(object, target)
    sourceTime = os.path.getmtime(source)
    if (objectTime < sourceTime):
      self.debugPrint(source+' is newer than '+target+'('+object+')', 3)
      return 1
    else:
      return 0

class Action (Transform):
  def __init__(self, program, sources = FileGroup(), flags = '', fileFilter = lambda file: 1, allAtOnce = 0, errorHandler = None):
    Transform.__init__(self, sources, fileFilter)
    self.program      = program
    self.flags        = flags
    self.allAtOnce    = allAtOnce
    self.errorHandler = errorHandler

  def doFunction(self):
    files = filter(self.fileFilter, self.sources.getFiles())
    self.debugPrint('Applying '+str(self.program)+' to '+str(files), 2)
    if (self.allAtOnce):
      self.program(FileGroup(files))
    else:
      map(self.program, files)

  def doShellCommand(self):
    commandBase = self.program+' '+self.flags
    if (self.allAtOnce):
      command = commandBase
      files   = self.sources.getFiles()
      if (not files): return ''
      for file in files:
        if (self.fileFilter(file)):
          command += ' '+file
      if (command == commandBase): return ''
      return self.executeShellCommand(command, self.errorHandler)
    else:
      output = ''
      for file in self.sources.getFiles():
        if (self.fileFilter(file)):
          command = commandBase+' '+file
          output += self.executeShellCommand(command, self.errorHandler)
      return output

  def execute(self):
    if (callable(self.program)):
      self.doFunction()
    else:
      self.doShellCommand()
    return self.products

class RemoveFiles (Action):
  def __init__(self, sources = FileGroup()):
    Action.__init__(self, self.forceRemove, sources)

class ArchiveObjects (Action):
  def __init__(self, library, objects, archiver = 'ar', archiverFlags = 'cvf'):
    Action.__init__(self, archiver, objects, archiverFlags)
    self.library = library

  def execute(self):
    (dir, file) = os.path.split(self.library)
    if not os.path.exists(dir): os.makedirs(dir)
    return Action.execute(self)

class BKEditFiles (Action):
  def __init__(self, sources = FileGroup(), extraSources = None, flags = '', fileFilter = lambda file: 1):
    Action.__init__(self, 'bk', sources, 'edit '+flags, fileFilter, 1, self.checkEdit)
    self.extraSources = extraSources

  def checkEdit(self, command, status, output):
    if (status):
      lines    = string.split(output, '\n')
      badLines = ''
      for line in lines:
        if line[0:4] == 'edit:':
          badLines += line
      if badLines:
        raise RuntimeError('Could not execute \''+command+'\': '+output)

  def execute(self):
    if self.sources: self.sources.extend(self.extraSources)
    return Action.execute(self)

class BKCloseFiles (Action):
  def __init__(self, sources = TreeFileGroup(), flags = '', fileFilter = lambda file: 1):
    Action.__init__(self, 'bk', sources, flags, fileFilter, 1)

  def execute(self):
    oldFlags   = self.flags
    oldSources = self.sources
    root       = self.sources.root
    # Add files which were just generated and check them out
    self.debugPrint('Putting new files under version control', 2)
    sources    = FileGroup(string.split(self.executeShellCommand('bk sfiles -ax '+root)))
    self.sources = sources
    self.flags = 'add '+oldFlags
    Action.execute(self)
    self.flags = 'co -q '+oldFlags
    Action.execute(self)
    # Remove files with no changes
    self.debugPrint('Reverting unchanged files', 2)
    sources    = FileGroup(string.split(self.executeShellCommand('bk sfiles -lg '+root)))
    self.sources = sources
    map(self.sources.remove, string.split(self.executeShellCommand('bk sfiles -cg '+root)))
    self.flags = 'unedit '+oldFlags
    Action.execute(self)
    self.flags = 'co -q '+oldFlags
    Action.execute(self)
    # Checkin files with harmless deltas
    #self.flags = 'fooci -u -f -y\'Babel generation\' '+oldFlags
    #Action.execute(self)
    # Cleanup
    self.flags = oldFlags
    return oldSources

class CompileFiles (Action):
  def __init__(self, library, sources, fileFilter, compiler, compilerFlags, archiver, archiverFlags, allAtOnce = 0):
    Action.__init__(self, self.fullCompile, sources, '', fileFilter, allAtOnce)
    if (library):
      self.library     = library.getFiles()[0]
    else:
      self.library     = None
    self.compiler      = compiler
    self.compilerFlags = compilerFlags
    self.archiver      = archiver
    self.archiverFlags = archiverFlags
    self.includeDirs   = []

  def getIncludeFlags(self):
    flags = ''
    for dir in self.includeDirs: flags += ' -I'+dir
    return flags

  def compile(self, source):
    command  = self.compiler
    flags    = self.compilerFlags+self.getIncludeFlags()
    command += ' '+flags
    object   = self.getObjectName(source)
    if (object):
      command += ' -o '+object
    command += ' '+source
    self.executeShellCommand(command)
    return object

  def archive(self, object):
    command = self.archiver+' '+self.archiverFlags+' '+self.library+' '+object
    self.executeShellCommand(command)
    os.remove(object)

  def fullCompile(self, source):
    self.cleanupTmpDir()
    if (isinstance(source, FileGroup)):
      files   = source.getFiles()
      if (not files): return
      sources = ''
      for file in files:
        self.updateSourceDB(file)
        sources += ' '+file
      if (not sources): return
      self.compile(sources)
    else:
      self.updateSourceDB(source)
      object = self.compile(source)
      self.archive(object)

  def execute(self):
    if (self.library):
      (dir, file) = os.path.split(self.library)
      if not os.path.exists(dir): os.makedirs(dir)
    return Action.execute(self)

class CompileCFiles (CompileFiles):
  def __init__(self, library, sources = FileGroup(), compiler='gcc', compilerFlags='-c -g -Wall', archiver = 'ar', archiverFlags = 'crv', allAtOnce = 0):
    CompileFiles.__init__(self, library, sources, self.cFilter, compiler, compilerFlags, archiver, archiverFlags, allAtOnce)
    self.includeDirs.append('.')
    self.products = library

  def cFilter(self, source):
    (dir, file) = os.path.split(source)
    (base, ext) = os.path.splitext(file)
    if (ext == ".c"):
      return 1
    else:
      return 0

class CompileCxxFiles (CompileFiles):
  def __init__(self, library, sources = FileGroup(), compiler='g++', compilerFlags='-c -g -Wall', archiver = 'ar', archiverFlags = 'crv', allAtOnce = 0):
    CompileFiles.__init__(self, library, sources, self.cxxFilter, compiler, compilerFlags, archiver, archiverFlags, allAtOnce)
    self.includeDirs.append('.')
    self.products = library

  def cxxFilter(self, source):
    (dir, file) = os.path.split(source)
    (base, ext) = os.path.splitext(file)
    if (ext == ".cc"):
      return 1
    else:
      return 0

class CompileSIDLFiles (CompileFiles):
  def __init__(self, generatedSources, sources = FileGroup(), compiler='babel', compilerFlags='-sC++ -ogenerated', archiver = '', archiverFlags = '', allAtOnce = 1):
    CompileFiles.__init__(self, None, sources, self.sidlFilter, compiler, '--suppress-timestamp '+compilerFlags, archiver, archiverFlags, allAtOnce)
    self.products = generatedSources

  def sidlFilter(self, source):
    (dir, file) = os.path.split(source)
    (base, ext) = os.path.splitext(file)
    if (ext == ".sidl"):      return 1
    else:
      return 0

  def getObjectName(self, source): pass

  def archive(self, source): pass

class LinkSharedLibrary (Action):
  def __init__(self, sources = FileGroup(), linker = 'g++', linkerFlags = '-g -shared', archiver = 'ar', archiverFlags = 'x', extraLibraries = FileGroup()):
    Action.__init__(self, linker, sources, linkerFlags, allAtOnce = 0)
    self.linker         = linker
    self.linkerFlags    = linkerFlags
    self.archiver       = archiver
    self.archiverFlags  = archiverFlags
    self.extraLibraries = extraLibraries

  def getSharedName(self, library):
    (base, ext) = os.path.splitext(library)
    return base+'.so'

  def execute(self):
    for source in self.sources.getFiles():
      sharedLibrary = self.getSharedName(source)
      self.products.append(sharedLibrary)

      linkDir = os.path.join(self.tmpDir, 'link')
      oldDir  = os.getcwd()
      os.mkdir(linkDir)
      os.chdir(linkDir)
      command = self.archiver+' '+self.archiverFlags+' '+source
      self.executeShellCommand(command)
      command = self.linker+' '+self.linkerFlags+' -o '+sharedLibrary+' *.o'
      for lib in self.extraLibraries.getFiles():
        (dir, file) = os.path.split(lib)
        (base, ext) = os.path.splitext(file)
        command += ' -L'+dir+' -l'+base[3:]
      self.executeShellCommand(command)
      map(os.remove, os.listdir(linkDir))
      os.chdir(oldDir)
      os.rmdir(linkDir)
      return self.products

class LinkExecutable (Action):
  def __init__(self, executable, sources = FileGroup(), linker = 'g++', linkerFlags = '', extraLibraries = FileGroup()):
    Action.__init__(self, linker, sources, '-o '+executable.getFiles()[0]+' '+linkerFlags, allAtOnce = 1)
    self.executable  = executable
    self.linker      = linker
    self.linkerFlags = linkerFlags
    self.extraLibraries = extraLibraries
    self.products    = executable

  def execute(self):
    if (self.executable):
      (dir, file) = os.path.split(self.executable.getFiles()[0])
      if not os.path.exists(dir): os.makedirs(dir)
    if (self.sources):
      self.sources.extend(self.extraLibraries)
      return Action.execute(self)

class Target (Transform):
  def __init__(self, sources, transforms = []):
    Transform.__init__(self, sources)
    self.transforms = transforms[:]

  def executeTransform(self, sources, transform):
    if isinstance(transform, Transform):
      files = sources.getFiles()
      self.debugPrint('Executing transform '+str(transform)+' with sources '+self.debugListStr(files))
      transform.sources.extend(sources)
      products = transform.execute()
    elif isinstance(transform, types.ListType):
      for t in transform:
        products = self.executeTransform(sources, t)
        sources  = products
    elif isinstance(transform, types.TupleType):
      products = FileGroup()
      for t in transform:
        products.extend(self.executeTransform(sources, t))
    else:
      raise RuntimeError('Invalid transform type '+type(transform))
    return products

  def execute(self):
    return self.executeTransform(self.sources, self.transforms)

sourceDBFilename = 'bsSource.db'
def saveSourceDB():
  dbFile = open(sourceDBFilename, 'w')
  cPickle.dump(sourceDB, dbFile)
  dbFile.close()

def main():
  if os.path.exists(sourceDBFilename):
    dbFile   = open(sourceDBFilename, 'r')
    sourceDB = cPickle.load(dbFile)
    dbFile.close()
  atexit.register(saveSourceDB)
