#!/usr/bin/env python
import action
import fileset
import logging
import transform

import os
import string
import types
import commands
import re

class Process (action.Action):
  def __init__(self, sourceDB, products, tag, sources, compiler, compilerFlags, setwiseExecute, updateType = 'immediate'):
    self.sourceDB = sourceDB
    if setwiseExecute:
      action.Action.__init__(self, compiler,     sources, compilerFlags, setwiseExecute)
    else:
      action.Action.__init__(self, self.process, sources, compilerFlags, setwiseExecute)
    self.compiler        = compiler
    self.buildProducts   = 0
    if isinstance(products, fileset.FileSet):
      self.products      = products
    else:
      self.products      = fileset.FileSet()
    self.tag             = tag
    self.updateType      = updateType
    if updateType == 'deferred':
      self.deferredUpdates = fileset.FileSet(tag = 'update '+self.tag)
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products, self.deferredUpdates]
      else:
        self.products.append(self.deferredUpdates)

  def shellSetAction(self, set):
    if set.tag == self.tag:
      self.debugPrint(self.program+' processing '+self.debugFileSetStr(set), 3, 'compile')
      action.Action.shellSetAction(self, set)
      if self.updateType == 'immediate':
        for file in set:
          self.sourceDB.updateSource(file)
      elif self.updateType == 'deferred':
        set.tag = 'update '+set.tag
        if isinstance(self.products, fileset.FileSet):
          self.products = [self.products]
        self.products.append(set)
    elif set.tag == 'old '+self.tag:
      pass
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)

  def process(self, source):
    self.debugPrint(self.compiler+' processing '+source, 3, 'compile')
    # Compile file
    command  = self.compiler
    command += ' '+self.constructFlags(source, self.flags)
    command += ' '+source
    output   = self.executeShellCommand(command, self.errorHandler)
    # Update source DB if it compiled successfully
    if self.updateType == 'immediate':
      self.sourceDB.updateSource(source)
    elif self.updateType == 'deferred':
      self.deferredUpdates.append(source)
    return source

  def setExecute(self, set):
    if set.tag == self.tag:
      transform.Transform.setExecute(self, set)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)

class Compile (action.Action):
  def __init__(self, sourceDB, library, tag, sources, compiler, compilerFlags, archiver, archiverFlags, setwiseExecute, updateType = 'immediate'):
    self.sourceDB = sourceDB
    if setwiseExecute:
      action.Action.__init__(self, self.compileSet, sources, compilerFlags, setwiseExecute)
    else:
      action.Action.__init__(self, self.compile,    sources, compilerFlags, setwiseExecute)
    self.tag           = tag
    self.compiler      = compiler
    self.archiver      = archiver
    self.archiverFlags = archiverFlags
    self.archiverRoot  = ''
    self.library       = library
    self.buildProducts = 0
    self.products      = self.library
    self.updateType    = updateType
    if updateType == 'deferred':
      self.deferredUpdates = fileset.FileSet(tag = 'update '+self.tag)
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products, self.deferredUpdates]
      else:
        self.products.append(self.deferredUpdates)
    self.includeDirs   = []
    self.defines       = []
    self.rebuildAll    = 0
    return

  def checkIncludeDirectory(self, dirname):
    if not os.path.isdir(dirname):
      if not dirname or not dirname[0:2] == '-I':
        raise RuntimeError('Include directory '+dirname+' does not exist')

  def getIncludeFlags(self):
    flags = ''
    for dirname in self.includeDirs:
      self.checkIncludeDirectory(dirname)
      if dirname[0] == '-': flags += ' '+dirname
      else:                 flags += ' -I'+dirname
    return flags

  def getDefines(self):
    flags = ''
    for define in self.defines:
      if type(define) == types.TupleType:
        flags += ' -D'+define[0]+'='+define[1]
      else:
        flags += ' -D'+define
    return flags

  def getLibraryName(self, source = None):
    return self.library[0]

  def compile(self, source):
    library = self.getLibraryName(source)
    self.debugPrint('Compiling '+source+' into '+library, 3, 'compile')
    # Compile file
    command  = self.compiler
    flags    = self.constructFlags(source, self.flags)+self.getDefines()+self.getIncludeFlags()
    command += ' '+flags
    object   = self.getIntermediateFileName(source)
    if (object): command += ' -o '+object
    command += ' '+source
    output   = self.executeShellCommand(command, self.errorHandler)
    # Update source DB if it compiled successfully
    if self.updateType == 'immediate':
      self.sourceDB.updateSource(source)
    elif self.updateType == 'deferred':
      self.deferredUpdates.append(source)
    # Archive file
    if self.archiver:
      command = self.archiver+' '+self.archiverFlags+' '+library+' '+object
      output  = self.executeShellCommand(command, self.errorHandler)
      os.remove(object)
    return object

  def compileSet(self, set):
    if len(set) == 0: return set
    library = self.getLibraryName()
    self.debugPrint('Compiling '+self.debugFileSetStr(set)+' into '+library, 3, 'compile')
    # Compile files
    command  = self.compiler
    flags    = self.constructFlags(set, self.flags)+self.getDefines()+self.getIncludeFlags()
    command += ' '+flags
    for source in set:
      command += ' '+source
    output   = self.executeShellCommand(command, self.errorHandler)
    # Update source DB if it compiled successfully
    for source in set:
      if self.updateType == 'immediate':
        self.sourceDB.updateSource(source)
      elif self.updateType == 'deferred':
        self.deferredUpdates.append(source)
    # Archive files
    if self.archiver:
      objects = []
      for source in set:
        objects.append(self.getIntermediateFileName(source))
      command = self.archiver+' '+self.archiverFlags+' '+library
      if self.archiverRoot:
        dirs = string.split(self.archiverRoot, os.sep)
        for object in objects:
          odirs = string.split(object, os.sep)
          if not dirs == odirs[:len(dirs)]: raise RuntimeError('Invalid object '+object+' not under root '+self.archiverRoot)
          command += ' -C '+self.archiverRoot+' '+string.join(odirs[len(dirs):], os.sep)
      else:
        for object in objects:
          command += ' '+object
      output  = self.executeShellCommand(command, self.errorHandler)
      for object in objects:
        os.remove(object)
    return set

  def rebuild(self, source):
    if self.rebuildAll: return 1
    library = self.getLibraryName(source)
    if library:
      if not os.path.exists(library):
        if self.sourceDB.has_key(library): del self.sourceDB[library]
        (dir, file) = os.path.split(library)
        if not os.path.exists(dir):
          os.makedirs(dir)
        return 1
      elif not self.sourceDB.has_key(library):
        return 1
    return 0

  def setExecute(self, set):
    if set.tag == self.tag:
      transform.Transform.setExecute(self, set)
    elif set.tag == 'old '+self.tag:
      redo = fileset.FileSet()
      redo.extend(filter(self.rebuild, set))
      if len(redo) > 0: transform.Transform.setExecute(self, redo)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)

  def setAction(self, set):
    if set.tag == self.tag:
      self.func(set)
    elif set.tag == 'old '+self.tag:
      redo = fileset.FileSet()
      redo.extend(filter(self.rebuild, set))
      if len(redo) > 0: self.func(redo)
    else:
      if isinstance(self.products, fileset.FileSet):
        self.products = [self.products]
      self.products.append(set)

  def execute(self):
    library = self.getLibraryName()
    if library:
      if not os.path.exists(library):
        self.rebuildAll = 1
        if self.sourceDB.has_key(library): del self.sourceDB[library]
        (dir, file) = os.path.split(library)
        if not os.path.exists(dir):
          os.makedirs(dir)
      elif not self.sourceDB.has_key(library):
        self.rebuildAll = 1
    return action.Action.execute(self)

class TagC (transform.GenericTag):
  def __init__(self, sourceDB, tag = 'c', ext = 'c', sources = None, extraExt = 'h', root = None):
    transform.GenericTag.__init__(self, sourceDB, tag, ext, sources, extraExt, root)

class CompileC (Compile):
  def __init__(self, sourceDB, library, sources = None, tag = 'c', compiler = 'gcc', compilerFlags = '-g -Wall -Wundef -Wpointer-arith -Wbad-function-cast -Wcast-align -Wwrite-strings -Wconversion -Wsign-compare -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations -Wmissing-noreturn -Wredundant-decls -Wnested-externs -Winline', archiver = 'ar', archiverFlags = 'crv'):
    Compile.__init__(self, sourceDB, library, tag, sources, compiler, '-c '+compilerFlags, archiver, archiverFlags, 0)
    self.includeDirs.append('.')
    self.errorHandler = self.handleCErrors

  def handleCErrors(self, command, status, output):
    if status:
      raise RuntimeError('Could not execute \''+command+'\':\n'+output)
    elif output.find('warning') >= 0:
      print('\''+command+'\': '+output)

class CompilePythonC (CompileC):
  def __init__(self, sourceDB, sources = None, tag = 'c', compiler = 'gcc', compilerFlags = '-g -Wall -Wundef -Wpointer-arith -Wbad-function-cast -Wcast-align -Wwrite-strings -Wconversion -Wsign-compare -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations -Wmissing-noreturn -Wredundant-decls -Wnested-externs -Winline', archiver = 'ar', archiverFlags = 'crv'):
    CompileC.__init__(self, sourceDB, None, sources, tag, compiler, compilerFlags, archiver, archiverFlags)
    self.products = []

  def getLibraryName(self, source = None):
    if not source: return None
    (dir, file) = os.path.split(source)
    (base, ext) = os.path.splitext(file)
    if not base[-7:] == '_Module' or not ext == '.c':
      # Stupid Babel hack
      if base == 'SIDLObjA' or base == 'SIDLPyArrays' or base == 'OpaqueObject':
        package     = base
        libraryName = os.path.join(dir, package+'.a')
      else:
        raise RuntimeError('Invalid Python extension module: '+source)
    else:
      package     = base[:-7]
      libraryName = os.path.join(dir, package+'module.a')
    return libraryName

  def setExecute(self, set):
    Compile.setExecute(self, set)
    if set.tag == self.tag or set.tag == 'old '+self.tag:
      for source in set:
        self.products.append(fileset.FileSet([self.getLibraryName(source)]))


class TagCxx (transform.GenericTag):
  def __init__(self, sourceDB, tag = 'cxx', ext = 'cc', sources = None, extraExt = 'hh', root = None):
    transform.GenericTag.__init__(self, sourceDB, tag, ext, sources, extraExt, root)

class CompileCxx (Compile):
  def __init__(self, sourceDB, library, sources = None, tag = 'cxx', compiler = 'g++', compilerFlags = '-g -Wall -Wundef -Wpointer-arith -Wbad-function-cast -Wcast-align -Wwrite-strings -Wconversion -Wsign-compare -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations -Wmissing-noreturn -Wnested-externs -Winline', archiver = 'ar', archiverFlags = 'crv'):
    Compile.__init__(self, sourceDB, library, tag, sources, compiler, '-c '+compilerFlags, archiver, archiverFlags, 0)
    self.includeDirs.append('.')
    self.errorHandler = self.handleCxxErrors
    self.checkCompiler()

  def checkCompiler(self):
    # Make sure g++ is recent enough
    (status,output) = commands.getstatusoutput('g++ -dumpversion')
    if not status == 0:
      raise RuntimeError('g++ is not in your path; please make sure that you have a g++ of at least version 3 installed in your path. Get gcc/g++ at http://gcc.gnu.com')
    version = output.split('.')[0]
    if not version == '3':
      raise RuntimeError('The g++ in your path is version '+version+'; please install a g++ of at least version 3 or fix your path. Get gcc/g++ at http://gcc.gnu.com')
    return

  def handleCxxErrors(self, command, status, output):
    if status:
      raise RuntimeError('Could not execute \''+command+'\':\n'+output)
    elif output.find('warning') >= 0:
      print('\''+command+'\': '+output)

class CompileMatlabCxx (CompileCxx):
  def __init__(self, sourceDB, sources = None, tag = 'cxx', compiler = 'g++', compilerFlags = '-g -Wall -Wundef -Wpointer-arith -Wbad-function-cast -Wcast-align -Wwrite-strings -Wconversion -Wsign-compare -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations -Wmissing-noreturn -Wredundant-decls -Wnested-externs -Winline', archiver = 'ar', archiverFlags = 'crv'):
    CompileCxx.__init__(self, sourceDB, None, sources, tag, compiler, compilerFlags, archiver, archiverFlags)
    self.includeDirs.append(self.argDB['MATLAB_INCLUDE'])
    self.products = []

  def getLibraryName(self, source = None):
    if not source: return None
    (dir, file) = os.path.split(source)
    (base, ext) = os.path.splitext(file)
    libraryName = os.path.join(dir, base+'.a')
    return libraryName

  def setExecute(self, set):
    Compile.setExecute(self, set)
    if set.tag == self.tag or set.tag == 'old '+self.tag:
      for source in set:
        self.products.append(fileset.FileSet([self.getLibraryName(source)]))

class TagF77 (transform.GenericTag):
  def __init__(self, sourceDB, tag = 'f77', ext = 'f', sources = None, extraExt = '', root = None):
    transform.GenericTag.__init__(self, sourceDB, tag, ext, sources, extraExt, root)

class CompileF77 (Compile):
  def __init__(self, sourceDB, library, sources = None, tag = 'f77', compiler = 'g77', compilerFlags = '-g', archiver = 'ar', archiverFlags = 'crv'):
    Compile.__init__(self, sourceDB, library, tag, sources, compiler, '-c '+compilerFlags, archiver, archiverFlags, 0)

class TagF90 (transform.GenericTag):
  def __init__(self, sourceDB, tag = 'f90', ext = 'f90', sources = None, extraExt = '', root = None):
    transform.GenericTag.__init__(self, sourceDB, tag, ext, sources, extraExt, root)

class CompileF90 (Compile):
  def __init__(self, sourceDB, library, sources = None, tag = 'f90', compiler = 'ifc', compilerFlags = '-g', archiver = 'ar', archiverFlags = 'crv'):
    Compile.__init__(self, sourceDB, library, tag, sources, compiler, '-c '+compilerFlags, archiver, archiverFlags, 0)

class TagJava (transform.GenericTag):
  def __init__(self, sourceDB, tag = 'java', ext = 'java', sources = None, extraExt = '', root = None):
    transform.GenericTag.__init__(self, sourceDB, tag, ext, sources, extraExt, root)

class CompileJava (Compile):
  def __init__(self, sourceDB, library, sources = None, tag = 'java', compiler = 'javac', compilerFlags = '-g', archiver = 'jar', archiverFlags = 'cf'):
    Compile.__init__(self, sourceDB, library, tag, sources, compiler, compilerFlags, archiver, archiverFlags, 1, 'deferred')
    self.includeDirs.append('.')
    self.errorHandler = self.handleJavaErrors

  def getDefines(self):
    return ''

  def checkJavaInclude(self, dirname):
    ext = os.path.splitext(dirname)[1]
    if ext == '.jar':
      if not os.path.isfile(dirname):
        raise RuntimeError('Jar file '+dirname+' does not exist')
    else:
      self.checkIncludeDirectory(dirname)

  def getIncludeFlags(self):
    flags = ' -classpath '
    for dirname in self.includeDirs:
      self.checkJavaInclude(dirname)
      flags += dirname+":"
    return flags[:-1]

  def getIntermediateFileName(self, source):
    (base, ext) = os.path.splitext(source)
    if not ext == '.java': raise RuntimeError('Invalid Java file: '+source)
    return base+'.class'

  def handleJavaErrors(self, command, status, output):
    if status:
      raise RuntimeError('Could not execute \''+command+'\':\n'+output)
    elif output.find('warning') >= 0:
      print('\''+command+'\': '+output)

  def execute(self):
    retval = Compile.execute(self)
    # Need to update JAR files since they do not get linked
    library = self.getLibraryName()
    if os.path.exists(library): self.sourceDB.updateSource(library)
    return retval

class TagEtags (transform.GenericTag):
  def __init__(self, sourceDB, tag = 'etags', ext = ['c', 'h', 'cc', 'hh', 'py'], sources = None, extraExt = '', root = None):
    transform.GenericTag.__init__(self, sourceDB, tag, ext, sources, extraExt, root)

class CompileEtags (Compile):
  def __init__(self, sourceDB, tagsFile, sources = None, tag = 'etags', compiler = 'etags', compilerFlags = '-a'):
    Compile.__init__(self, sourceDB, tagsFile, tag, sources, compiler, compilerFlags, '', '', 1, 'deferred')
    return

  def checkCompiler(self):
    '''If "self.compiler --help" fails, we assume etags is absent'''
    (status, output) = commands.getstatusoutput(self.compiler+' --help')
    return not status

  def constructFlags(self, source, baseFlags):
    return baseFlags+' -f '+self.getLibraryName()

  def getIntermediateFileName(self, source):
    return None
