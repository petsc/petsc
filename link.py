#!/usr/bin/env python
import action
import fileset
import transform

import os

class TagLibrary (transform.GenericTag):
  def __init__(self, tag = 'lib', ext = 'a', sources = None, extraExt = ''):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)

class LinkSharedLibrary (action.Action):
  def __init__(self, sources = None, linker = 'g++', linkerFlags = '-g -shared', archiver = 'ar', archiverFlags = 'x', extraLibraries = None):
    action.Action.__init__(self, self.link, sources, linkerFlags, 0)
    self.linker         = linker
    self.linkerFlags    = linkerFlags
    self.archiver       = archiver
    self.archiverFlags  = archiverFlags
    if extraLibraries:
      self.extraLibraries = extraLibraries
    else:
      self.extraLibraries = fileset.FileSet()
    self.sharedLibs     = fileset.FileSet(tag = 'shared')
    self.products       = [self.sharedLibs]

  def getSharedName(self, libName):
    (base, ext) = os.path.splitext(libName)
    return base+'.so'

  def link(self, source):
    linkDir = os.path.join(self.tmpDir, 'link')
    oldDir  = os.getcwd()
    os.mkdir(linkDir)
    os.chdir(linkDir)
    sharedLibrary = self.getSharedName(source)
    self.sharedLibs.append(sharedLibrary)

    command = self.archiver+' '+self.archiverFlags+' '+source
    self.executeShellCommand(command)
    command = self.linker+' '+self.linkerFlags+' -o '+sharedLibrary+' *.o'
    for lib in self.extraLibraries.getFiles():
      (dir, file) = os.path.split(lib)
      (base, ext) = os.path.splitext(file)
      command += ' -L'+dir+' -l'+base[3:]
    self.executeShellCommand(command)
    self.updateSourceDB(source)
    map(os.remove, os.listdir(linkDir))
    os.chdir(oldDir)
    os.rmdir(linkDir)
    return self.products

  def setExecute(self, set):
    if set.tag == 'lib':
      transform.Transform.setExecute(self, set)
    else:
      self.products.append(set)

class LinkExecutable (action.Action):
  def __init__(self, executable, sources = None, linker = 'g++', linkerFlags = '', extraLibraries = None):
    action.Action.__init__(self, self.link, sources, '-o '+executable.getFiles()[0]+' '+linkerFlags, setwiseExecute = 1)
    self.executable  = executable
    self.linker      = linker
    self.linkerFlags = linkerFlags
    if extraLibraries:
      self.extraLibraries = extraLibraries
    else:
      self.extraLibraries = fileset.FileSet()
    self.rebuildAll  = 0
    self.products    = executable

  def link(self, set):
    self.sources.extend(self.extraLibraries)
    command = self.linker+' '+self.flags
    files   = set.getFiles()
    if files:
      for file in files: command += ' '+file
      output = self.executeShellCommand(command, self.errorHandler)
    for source in files:
      self.updateSourceDB(source)
    return self.products

  def execute(self):
    if (self.executable):
      (dir, file) = os.path.split(self.executable.getFiles()[0])
      if not os.path.exists(dir): os.makedirs(dir)
    return action.Action.execute(self)
