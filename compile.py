#!/usr/bin/env python
import action
import fileset
import transform

import os

class TagSIDL (transform.FileChanged):
  def __init__(self, sources = None):
    transform.FileChanged.__init__(self, sources)
    self.changed.tag   = 'sidl'
    self.unchanged.tag = 'old sidl'
    self.products      = []

  def fileExecute(self, source):
    (base, ext) = os.path.splitext(source)
    if ext == '.sidl':
      transform.FileChanged.fileExecute(self, source)
    else:
      self.currentSet.append(source)

  def execute(self):
    self.genericExecute(self.sources)
    if len(self.changed):
      self.changed.extend(self.unchanged)
      self.products.append(self.changed)
    else:
      self.products.append(self.unchanged)
    return self.products

class CompileSIDL (action.Action):
  def __init__(self, generatedSources, sources = None, compiler = 'babel', compilerFlags = '-sC++ -ogenerated'):
    action.Action.__init__(self, compiler, sources, '--suppress-timestamp '+compilerFlags, 1)
    self.generatedSources = generatedSources
    self.products         = self.generatedSources

  def shellSetAction(self, set):
    if set.tag == 'sidl':
      action.Action.shellSetAction(self, set)
    elif set.tag == 'old sidl':
      pass
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)

class TagForCompile (transform.FileChanged):
  def __init__(self, tag, ext, sources = None, extraExt = ''):
    transform.FileChanged.__init__(self, sources)
    self.ext           = '.'+ext
    self.extraExt      = '.'+extraExt
    self.changed.tag   = tag
    self.unchanged.tag = 'old '+tag
    self.products      = [self.changed, self.unchanged]

  def fileExecute(self, source):
    (base, ext) = os.path.splitext(source)
    if ext == self.ext:
      transform.FileChanged.fileExecute(self, source)
    elif ext == self.extraExt:
      self.updateSourceDB(source)
    else:
      self.currentSet.append(source)

  def setExecute(self, set):
    self.currentSet = fileset.FileSet(tag = set.tag)
    for file in set.getFiles():
      self.fileExecute(file)
    if len(self.currentSet): self.products.append(self.currentSet)
    return self.products

class Compile (action.Action):
  def __init__(self, library, sources, tag, compiler, compilerFlags, archiver, archiverFlags):
    action.Action.__init__(self, self.compile, sources, compilerFlags, 0)
    self.library       = library
    self.tag           = tag
    self.compiler      = compiler
    self.compilerFlags = compilerFlags
    self.archiver      = archiver
    self.archiverFlags = archiverFlags
    self.products      = self.library
    self.includeDirs   = []
    self.rebuildAll    = 0

  def getIncludeFlags(self):
    flags = ''
    for dir in self.includeDirs: flags += ' -I'+dir
    return flags

  def compile(self, source):
    # Compile file
    command  = self.compiler
    flags    = self.compilerFlags+self.getIncludeFlags()
    command += ' '+flags
    object   = self.getIntermediateFileName(source)
    if (object): command += ' -o '+object
    command += ' '+source
    output   = self.executeShellCommand(command)
    # Update source DB if it compiled successfully
    self.updateSourceDB(source)
    # Archive file
    command = self.archiver+' '+self.archiverFlags+' '+self.library+' '+object
    output  = self.executeShellCommand(command)
    os.remove(object)
    return object

  def setExecute(self, set):
    if set.tag == self.tag:
      transform.Transform.setExecute(self, set)
    elif set.tag == 'old '+self.tag:
      if self.rebuildAll: transform.Transform.setExecute(self, set)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)

  def execute(self):
    if not os.path.exists(self.library[0]):
      self.rebuildAll = 1
    return action.Action.execute(self)

class TagC (TagForCompile):
  def __init__(self, tag = 'c', ext = 'c', sources = None, extraExt = 'h'):
    TagForCompile.__init__(self, tag, ext, sources, extraExt)

class CompileC (Compile):
  def __init__(self, library, sources = None, tag = 'c', compiler = 'gcc', compilerFlags = '-c -g -Wall', archiver = 'ar', archiverFlags = 'crv'):
    Compile.__init__(self, library, sources, tag, compiler, compilerFlags, archiver, archiverFlags)
    self.includeDirs.append('.')

class TagCxx (TagForCompile):
  def __init__(self, tag = 'cxx', ext = 'cc', sources = None, extraExt = 'hh'):
    TagForCompile.__init__(self, tag, ext, sources, extraExt)

class CompileCxx (Compile):
  def __init__(self, library, sources = None, tag = 'cxx', compiler = 'g++', compilerFlags = '-c -g -Wall', archiver = 'ar', archiverFlags = 'crv'):
    Compile.__init__(self, library, sources, tag, compiler, compilerFlags, archiver, archiverFlags)
    self.includeDirs.append('.')
