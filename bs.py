#!/usr/bin/env python
import commands
import os.path
import string
import time

echo = 1

class Maker:
  def executeShellCommand(self, command):
    if echo: print command
    (status, output) = commands.getstatusoutput(command)
    if status: raise IOError('Could not execute \''+command+'\': '+output)
    return output

class FileGroup (Maker):
  def __init__(self, data = [], func = None, children = []):
    self.data     = data
    self.func     = func
    self.children = children

  def append(self, item):
    self.data.append(item)

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

class Precondition (Maker):
  # True here means that the precondition was satisfied
  def __nonzero__(self):
    if echo: print 'Checking precondition'
    return 1

class Action (Maker):
  def __init__(self, program, flags, sources):
    self.program = program
    self.flags   = flags
    self.sources = sources

  def execute(self):
    command = self.program+' '+self.flags
    for file in self.sources.getFiles():
      command += ' '+file
    return self.executeShellCommand(command)

class Target (Maker):
  preconditions = []
  actions       = []

  def __init__(self, preconditions=None, actions=None):
    if preconditions: self.preconditions = preconditions
    if actions:       self.actions       = actions

  def execute(self):
    if (reduce(lambda a,b: a or b, self.preconditions)):
      map(lambda x: x.execute(), self.actions)

class OlderThan (Precondition):
  def __init__(self, target, sources):
    self.target  = target
    self.sources = sources

  def __nonzero__(self):
    if (not os.path.exists(self.target)): return 1
    targetTime = os.path.getmtime(self.target)
    files = self.sources.getFiles()
    if (not len(files)): return 1
    for source in files:
      sourceTime = os.path.getmtime(source)
      if (targetTime < sourceTime):
        print self.target+' is older than '+source
        return 1
    return 0

class NewerThan (Precondition):
  def __init__(self, target, sources):
    self.target  = target
    self.sources = sources

  def __nonzero__(self):
    if (not os.path.exists(self.target)): return 1
    targetTime = os.path.getmtime(self.target)
    files = self.sources.getFiles()
    if (not len(files)): return 1
    for source in files:
      sourceTime = os.path.getmtime(source)
      if (targetTime > sourceTime):
        print self.target+' is newer than '+source
        return 1
    return 0

class LibraryOlderThan (OlderThan):
  def __nonzero__(self):
    if (not os.path.exists(self.target)): return 1
    files = self.sources.getFiles()
    if (not len(files)): return 1
    for source in files:
      (dir, file) = os.path.split(source)
      (base, ext) = os.path.splitext(file)
      object      = base+'.o'

      output      = self.executeShellCommand('ar tv '+self.target+' '+object)
      if (output[0:8] != 'no entry'):
        objectTime = time.strptime(output[25:42], "%b %d %H:%M %Y")
      else:
        objectTime = 0

      sourceTime = os.path.getmtime(source)
      if (objectTime < sourceTime):
        print self.target+'('+object+')'+' is older than '+source
        return 1
    return 0

class CompileFiles (Action):
  def __init__(self, compiler, compilerFlags, sources):
    self.compiler      = compiler
    self.compilerFlags = compilerFlags
    self.sources       = sources
    self.includeDirs   = []
    self.objects       = FileGroup([])

  def addIncludeDir(self, dir):
    self.includeDirs.append(dir)

  def getIncludeFlags(self):
    flags = ''
    for dir in self.includeDirs: flags += ' -I'+dir
    return flags

  def allAtOnceCompile(self):
    files = self.sources.getFiles()
    if echo: print 'Compiling '+str(files)
    flags   = self.compilerFlags + self.getIncludeFlags()
    command = self.compiler+' '+flags
    for source in files:
      command    += ' '+source
      (base, ext) = os.path.splitext(source)
      object      = base+'.o'
      self.objects.append(object)
    output = self.executeShellCommand(command)
    if (string.find(output, 'err') >= 0):
      print output

  def fileByFileCompile(self):
    files = self.sources.getFiles()
    if echo: print 'Compiling '+str(files)
    flags = self.compilerFlags + self.getIncludeFlags()
    for source in files:
      (dir, file) = os.path.split(source)
      (base, ext) = os.path.splitext(file)
      object      = os.path.join(dir, string.replace(dir, '/', '_')+'_'+base+'.o')
      command = self.compiler+' '+flags+' -c -o '+object+' '+source
      self.executeShellCommand(command)
      self.objects.append(object)

  def execute(self):
    self.allAtOnceCompile()

class CompileCFiles (CompileFiles):
  def __init__(self, sources, compiler='gcc', compilerFlags='-g -Wall'):
    CompileFiles.__init__(self, compiler, compilerFlags, sources)

  def execute(self):
    self.fileByFileCompile()

class CompileCxxFiles (CompileFiles):
  def __init__(self, sources, compiler='g++', compilerFlags='-g -Wall'):
    CompileFiles.__init__(self, compiler, compilerFlags, sources)

  def execute(self):
    self.fileByFileCompile()

class ArchiveObjects (Action):
  def __init__(self, archiver, archiverFlags, objects, library):
    self.archiver      = archiver
    self.archiverFlags = archiverFlags
    self.objects       = objects
    self.library       = library
  
  def execute(self):
    (dir, file) = os.path.split(self.library)
    if not os.path.exists(dir): os.makedirs(dir)
    command = self.archiver+' '+self.archiverFlags+' '+self.library
    for obj in self.objects.getFiles(): command += ' '+obj
    self.executeShellCommand(command)

class RemoveFiles (Action):
  def __init__(self, files):
    self.files = files

  def forceRemove(self, file):
    if (os.path.exists(file)): os.remove(file)

  def execute(self):
    files = self.files.getFiles()
    if echo: print 'Removing '+str(files)
    map(self.forceRemove, files)
