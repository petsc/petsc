#!/usr/bin/env python
import commands
import os.path
import string
import time
import types

echo = 2

class Maker:
  def executeShellCommand(self, command):
    if echo: print command
    (status, output) = commands.getstatusoutput(command)
    if status: raise IOError('Could not execute \''+command+'\': '+output)
    return output

class FileGroup (Maker):
  def __init__(self, data = [], func = None, children = []):
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

  def getFiles(self):
    funcData = []
    if (self.func):
      funcData = self.func(self)
    childData = []
    for child in self.children:
      childData += child.getFiles()
    return self.data+funcData+childData

class TreeFileGroup (FileGroup):
  def __init__(self, root, fileTest):
    FileGroup.__init__(self, func=self.walkTree)
    self.root     = root
    self.fileTest = fileTest

  def walkTree(self, fileGroup):
    files = []
    os.path.walk(self.root, self.fileTest, files)
    return files

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
  def __init__(self, sources = FileGroup()):
    self.sources  = sources
    self.products = FileGroup()

  def execute(self):
    return self.sources

class FileCompare (Transform):
  "This class maps sources into the set of files for which self.compare(target, source) returns true"
  def __init__(self, targets, sources = FileGroup(), returnAll = 0):
    Transform.__init__(self, sources)
    self.targets   = targets
    self.returnAll = returnAll

  def compare(self, target, source): return 1

  def singleTargetExecute(self, target):
    files = self.sources.getFiles()
    for source in files:
      if (not os.path.exists(source)):
        if echo: print source+' does not exist'
        self.products.append(source)
      else:
        if (self.compare(target, source)):
          if echo: print target+' is older than '+source
          self.products.append(source)
    if (self.returnAll and len(self.products)):
      self.products = self.sources
    return self.products

  def execute(self):
    if echo > 1: print 'FileCompare: Comparing '+str(self.targets.getFiles())+' to '+str(self.sources.getFiles())
    files = self.targets.getFiles()
    for target in files:
      if (not os.path.exists(target)): return self.sources
      self.singleTargetExecute(target)
    return self.products

class OlderThan (FileCompare):
  "This class maps sources into the set of files which are older than target"
  def __init__(self, targets, sources = FileGroup(), returnAll = 0):
    FileCompare.__init__(self, targets, sources, returnAll)

  def compare(self, target, source):
    targetTime = os.path.getmtime(target)
    sourceTime = os.path.getmtime(source)
    if (targetTime > sourceTime):
      return 1
    else:
      return 0

class NewerThan (FileCompare):
  "This class maps sources into the set of files which are newer than target"
  def __init__(self, targets, sources = FileGroup(), returnAll = 0):
    FileCompare.__init__(self, targets, sources, returnAll)

  def compare(self, target, source):
    targetTime = os.path.getmtime(target)
    sourceTime = os.path.getmtime(source)
    if (targetTime < sourceTime):
      return 1
    else:
      return 0

class NewerThanLibraryObject (FileCompare):
  "This class maps sources into the set of files which are newer than the corresponding object in the target library"
  def __init__(self, targets, sources = FileGroup(), returnAll = 0):
    FileCompare.__init__(self, targets, sources, returnAll)

  def getObjectTimestamp(self, object, library):
    output = self.executeShellCommand('ar tv '+library+' '+object)
    if (output[0:8] != 'no entry'):
      objectTime = time.mktime(time.strptime(output[25:42], "%b %d %H:%M %Y"))
    else:
      if echo: print 'No entry for object '+object+' in '+library
      objectTime = 0
    return objectTime

  def compare(self, target, source):
    (dir, file) = os.path.split(source)
    (base, ext) = os.path.splitext(file)
    object      = base+'.o'
    objectTime  = self.getObjectTimestamp(object, target)
    sourceTime  = os.path.getmtime(source)
    if (string.find(source, 'Error') >= 0): print 'sourceTime: '+str(sourceTime)+' objectTime: '+str(objectTime)
    if (objectTime < sourceTime):
      return 1
    else:
      return 0

class Action (Transform):
  def __init__(self, program, sources = FileGroup(), flags = '', allAtOnce = 0):
    Transform.__init__(self, sources)
    self.program   = program
    self.flags     = flags
    self.allAtOnce = allAtOnce

  def doFunction(self):
    files = self.sources.getFiles()
    if echo: print 'Applying '+str(self.program)+' to '+str(files)
    if (self.allAtOnce):
      self.program(self.sources)
    else:
      map(self.program, files)

  def doShellCommand(self):
    commandBase = self.program+' '+self.flags
    if (self.allAtOnce):
      command = commandBase
      for file in self.sources.getFiles():
        command += ' '+file
      return self.executeShellCommand(command)
    else:
      output = ''
      for file in self.sources.getFiles():
        command = commandBase+' '+file
        output += self.executeShellCommand(command)
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

  def forceRemove(self, file):
    if (os.path.exists(file)): os.remove(file)

class ArchiveObjects (Action):
  def __init__(self, library, objects, archiver = 'ar', archiverFlags = 'cvf'):
    Action.__init__(self, archiver, objects, archiverFlags)
    self.library = library

  def execute(self):
    (dir, file) = os.path.split(self.library)
    if not os.path.exists(dir): os.makedirs(dir)
    return Action.execute(self)

class CompileFiles (Action):
  def __init__(self, library, sources, filter, compiler, compilerFlags, archiver, archiverFlags, allAtOnce = 0):
    Action.__init__(self, self.fullCompile, sources, '', allAtOnce)
    if (library):
      self.library     = library.getFiles()[0]
    else:
      self.library     = None
    self.filter        = filter
    self.compiler      = compiler
    self.compilerFlags = compilerFlags
    self.archiver      = archiver
    self.archiverFlags = archiverFlags
    self.tmpDir        = os.path.join(os.environ['TMPDIR'], 'bs')
    self.includeDirs   = []

  def setupTmpDir(self):
    if not os.path.exists(self.tmpDir): os.makedirs(self.tmpDir)
    oldDir = os.getcwd()
    os.chdir(self.tmpDir)
    map(os.remove, os.listdir(self.tmpDir))
    os.chdir(oldDir)

  def getIncludeFlags(self):
    flags = ''
    for dir in self.includeDirs: flags += ' -I'+dir
    return flags

  def getObjectName(self, source):
    (dir, file) = os.path.split(source)
    (base, ext) = os.path.splitext(file)
    return os.path.join(self.tmpDir, base+'.o')

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
    self.setupTmpDir()
    if (isinstance(source, FileGroup)):
      files   = source.getFiles()
      if (not files): return
      sources = ''
      for file in files:
        if (self.filter(file)): sources += ' '+file
      self.compile(sources)
    else:
      if (self.filter(source)):
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

  def cxxFilter(self, source):
    (dir, file) = os.path.split(source)
    (base, ext) = os.path.splitext(file)
    if (ext == ".cc"):
      return 1
    else:
      return 0

class CompileSIDLFiles (CompileFiles):
  def __init__(self, generatedSources, sources = FileGroup(), compiler='babel', compilerFlags='-sC++ -ogenerated', archiver = '', archiverFlags = '', allAtOnce = 1):
    CompileFiles.__init__(self, None, sources, self.sidlFilter, compiler, compilerFlags, archiver, archiverFlags, allAtOnce)
    self.products = generatedSources

  def sidlFilter(self, source):
    (dir, file) = os.path.split(source)
    (base, ext) = os.path.splitext(file)
    if (ext == ".sidl"):
      return 1
    else:
      return 0

  def getObjectName(self, source): pass

  def archive(self, source): pass

class Target (Transform):
  def __init__(self, sources, transforms = []):
    Transform.__init__(self, sources)
    self.transforms = transforms[:]

  def executeTransform(self, sources, transform):
    if isinstance(transform, Transform):
      if echo > 1: print 'Executing transform '+str(transform)+' with '+str(sources.getFiles())
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

