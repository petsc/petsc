#!/usr/bin/env python
import action
import bs
import fileset
import logging
import transform

import os
import types

class Process (action.Action):
  def __init__(self, products, tag, sources, compiler, compilerFlags, noUpdate = 0):
    action.Action.__init__(self, compiler, sources, compilerFlags, 1)
    if products:
      self.products    = products
    else:
      self.products    = fileset.FileSet()
    self.tag           = tag
    self.buildProducts = 0
    self.noUpdate      = noUpdate

  def shellSetAction(self, set):
    if set.tag == self.tag:
      self.debugPrint(self.program+' processing '+self.debugFileSetStr(set), 3, 'compile')
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
    self.defines       = []
    self.rebuildAll    = 0
    self.buildProducts = 0

  def getIncludeFlags(self):
    flags = ''
    for dir in self.includeDirs: flags += ' -I'+dir
    return flags

  def getDefines(self):
    flags = ''
    for define in self.defines:
      if type(define) == types.TupleType:
        flags += ' -D'+define[0]+'='+define[1]
      else:
        flags += ' -D'+define
    return flags

  def compile(self, source):
    self.debugPrint('Compiling '+source+' into '+self.library[0], 3, 'compile')
    # Compile file
    command  = self.compiler
    flags    = self.compilerFlags+self.getDefines()+self.getIncludeFlags()
    command += ' '+flags
    object   = self.getIntermediateFileName(source)
    if (object): command += ' -o '+object
    command += ' '+source
    output   = self.executeShellCommand(command, self.errorHandler)
    # Update source DB if it compiled successfully
    self.updateSourceDB(source)
    # Archive file
    command = self.archiver+' '+self.archiverFlags+' '+self.library[0]+' '+object
    output  = self.executeShellCommand(command, self.errorHandler)
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
    library = self.library[0]
    if not os.path.exists(library):
      self.rebuildAll = 1
      if bs.sourceDB.has_key(library): del bs.sourceDB[library]
      (dir, file) = os.path.split(library)
      if not os.path.exists(dir): os.makedirs(dir)
    elif not bs.sourceDB.has_key(library):
      self.rebuildAll = 1
    return action.Action.execute(self)

class TagC (transform.GenericTag):
  def __init__(self, tag = 'c', ext = 'c', sources = None, extraExt = 'h'):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)

class CompileC (Compile):
  def __init__(self, library, sources = None, tag = 'c', compiler = 'gcc', compilerFlags = '-g -Wall', archiver = 'ar', archiverFlags = 'crv'):
    Compile.__init__(self, library, sources, tag, compiler, '-c '+compilerFlags, archiver, archiverFlags)
    self.includeDirs.append('.')

class TagCxx (transform.GenericTag):
  def __init__(self, tag = 'cxx', ext = 'cc', sources = None, extraExt = 'hh'):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)

class CompileCxx (Compile):
  def __init__(self, library, sources = None, tag = 'cxx', compiler = 'g++', compilerFlags = '-g -Wall', archiver = 'ar', archiverFlags = 'crv'):
    Compile.__init__(self, library, sources, tag, compiler, '-c '+compilerFlags, archiver, archiverFlags)
    self.includeDirs.append('.')

class TagFortran (transform.GenericTag):
  def __init__(self, tag = 'fortran', ext = 'f', sources = None, extraExt = ''):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)

class CompileF77 (Compile):
  def __init__(self, library, sources = None, tag = 'fortran', compiler = 'g77', compilerFlags = '-g', archiver = 'ar', archiverFlags = 'crv'):
    Compile.__init__(self, library, sources, tag, compiler, '-c '+compilerFlags, archiver, archiverFlags)

class CompileF90 (Compile):
  def __init__(self, library, sources = None, tag = 'fortran', compiler = 'f90', compilerFlags = '-g', archiver = 'ar', archiverFlags = 'crv'):
    Compile.__init__(self, library, sources, tag, compiler, '-c '+compilerFlags, archiver, archiverFlags)

class TagEtags (transform.GenericTag):
  def __init__(self, tag = 'etags', ext = ['c', 'h', 'cc', 'hh', 'py'], sources = None, extraExt = ''):
    transform.GenericTag.__init__(self, tag, ext, sources, extraExt)

class CompileEtags (Process):
  def __init__(self, tagsFile, sources = None, compiler = 'etags', compilerFlags = '-a'):
    Process.__init__(self, tagsFile, 'etags', sources, compiler, compilerFlags+' -f '+tagsFile[0], 1)
