#!/usr/bin/env python
import action
import fileset
import transform

import os

class Process (action.Action):
  def __init__(self, products, tag, sources, compiler, compilerFlags, noUpdate = 0):
    action.Action.__init__(self, compiler, sources, compilerFlags, 1)
    self.products      = products
    self.tag           = tag
    self.buildProducts = 0
    self.noUpdate      = noUpdate

  def shellSetAction(self, set):
    if set.tag == self.tag:
      action.Action.shellSetAction(self, set)
      if not self.noUpdate:
        for file in set:
          self.updateSourceDB(file)
    elif set.tag == 'old '+self.tag:
      pass
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)

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

class CompileSIDL (Process):
  def __init__(self, generatedSources, sources, compiler, compilerFlags):
    Process.__init__(self, generatedSources, 'sidl', sources, compiler, '--suppress-timestamp '+compilerFlags)
    self.generatedSources = generatedSources

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
    self.debugPrint('Compiling '+source+' into '+self.library[0], 3, 'compile')
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
    command = self.archiver+' '+self.archiverFlags+' '+self.library[0]+' '+object
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
      (dir, file) = os.path.split(self.library[0])
      if not os.path.exists(dir): os.makedirs(dir)
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

class TagEtags (transform.GenericTag):
  def __init__(self, tag = 'etags', ext = ['c', 'h', 'cc', 'hh', 'py'], sources = None, extraExt = ''):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)

class CompileEtags (Process):
  def __init__(self, tagsFile, sources = None, compiler = 'etags', compilerFlags = '-a'):
    Process.__init__(self, tagsFile, 'etags', sources, compiler, compilerFlags+' -f '+tagsFile[0], 1)
