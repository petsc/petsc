import args
import config.compile.processor
import config.compile.C
import config.framework
import config.libraries
import os
import sys

import config.setsOrdered as sets

class Preprocessor(config.compile.processor.Processor):
  '''The CUDA preprocessor'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'CUDAPP', 'CUDAPPFLAGS', '.cpp', '.c')
    self.language        = 'CUDA'
    self.includeDirectories = sets.Set()
    return

class Compiler(config.compile.processor.Processor):
  '''The CUDA compiler'''
  def __init__(self, argDB, usePreprocessorFlags = True):
    config.compile.processor.Processor.__init__(self, argDB, 'CUDAC', 'CUDAFLAGS', '.cu', '.o')
    self.language        = 'CUDA'
    self.requiredFlags[-1]  = '-c'
    self.outputFlag         = '-o'
    self.includeDirectories = sets.Set()
    if usePreprocessorFlags:
      self.flagsName.extend(Preprocessor(argDB).flagsName)
    return

  def getTarget(self, source):
    '''Return None for header files'''
    import os

    base, ext = os.path.splitext(source)
    if ext == '.h':
      return None
    return base+'.o'

  def getCommand(self, sourceFiles, outputFile = None):
    '''If no outputFile is given, do not execute anything'''
    if outputFile is None:
      return 'true'
    return config.compile.processor.Processor.getCommand(self, sourceFiles, outputFile)

class Linker(config.compile.C.Linker):
  '''The CUDA linker'''
  def __init__(self, argDB):
    self.compiler        = Compiler(argDB, usePreprocessorFlags = False)
    self.configLibraries = config.libraries.Configure(config.framework.Framework(clArgs = '', argDB = argDB, tmpDir = os.getcwd()))
    config.compile.processor.Processor.__init__(self, argDB, [self.compiler.name], ['CUDAC_LINKER_FLAGS'], '.o', '.a')
    self.language   = 'CUDA'
    self.outputFlag = '-o'
    self.libraries  = sets.Set()
    return

  def getExtraArguments(self):
    if not hasattr(self, '_extraArguments'):
      return ''
    return self._extraArguments
  extraArguments = property(getExtraArguments, config.compile.processor.Processor.setExtraArguments, doc = 'Optional arguments for the end of the command')

class SharedLinker(config.compile.C.SharedLinker):
  '''The CUDA shared linker: Just use regular linker for now'''
  def __init__(self, argDB):
    config.compile.C.SharedLinker.__init__(self, argDB)
    self.language = 'CUDA'
    return

class StaticLinker(config.compile.C.StaticLinker):
  '''The CUDA static linker, just use C for now'''
  def __init__(self, argDB):
    config.compile.C.StaticLinker.__init__(self, argDB)
    self.language = 'CUDA'
    return

class DynamicLinker(config.compile.C.DynamicLinker):
  '''The CUDA dynamic linker, just use C for now'''
  def __init__(self, argDB):
    config.compile.C.DynamicLinker.__init__(self, argDB)
    self.language = 'CUDA'
    return
