#!/usr/bin/env python
import action
import fileset
import transform

import os

class TagSIDL (transform.GenericTag):
  def __init__(self, tag = 'sidl', ext = 'sidl', sources = None, useAll = 1, extraExt = ''):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)
    self.useAll = useAll

  def execute(self):
    self.genericExecute(self.sources)
    if len(self.changed) and self.useAll:
      self.changed.extend(self.unchanged)
      # This is bad
      self.unchanged.data = []
    return self.products

class CompileSIDL (action.Action):
  def __init__(self, generatedSources, sources, compiler, compilerFlags):
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

class CompileSIDLRepository (CompileSIDL):
  def __init__(self, sources = None, compiler = 'babel', compilerFlags = '--xml --output-directory=xml'):
    CompileSIDL.__init__(self, fileset.FileSet(), sources, compiler, compilerFlags)

class CompileSIDLServer (CompileSIDL):
  def __init__(self, generatedSources, sources = None, compiler = 'babel', compilerFlags = '--server=C++ --output-directory=generated --repository-path=xml'):
    CompileSIDL.__init__(self, generatedSources, sources, compiler, compilerFlags)

class CompileSIDLClient (CompileSIDL):
  def __init__(self, generatedSources, sources = None, compiler = 'babel', compilerFlags = '--client=Python --output-directory=generated --repository-path=xml'):
    CompileSIDL.__init__(self, generatedSources, sources, compiler, compilerFlags)

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

class TagC (transform.GenericTag):
  def __init__(self, tag = 'c', ext = 'c', sources = None, extraExt = 'h'):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)

class CompileC (Compile):
  def __init__(self, library, sources = None, tag = 'c', compiler = 'gcc', compilerFlags = '-c -g -Wall', archiver = 'ar', archiverFlags = 'crv'):
    Compile.__init__(self, library, sources, tag, compiler, compilerFlags, archiver, archiverFlags)
    self.includeDirs.append('.')

class TagCxx (transform.GenericTag):
  def __init__(self, tag = 'cxx', ext = 'cc', sources = None, extraExt = 'hh'):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)

class CompileCxx (Compile):
  def __init__(self, library, sources = None, tag = 'cxx', compiler = 'g++', compilerFlags = '-c -g -Wall', archiver = 'ar', archiverFlags = 'crv'):
    Compile.__init__(self, library, sources, tag, compiler, compilerFlags, archiver, archiverFlags)
    self.includeDirs.append('.')
