import args
import config.compile.processor
import config.compile.C
import config.framework
import config.libraries
import os
import sys

import config.setsOrdered as sets

'''
HIP is a C++ compiler with extensions to support the HIP programming model.
Because of its slowness, and in some ways the extensions make it a new language,
we have a separate compiler for it.
But we use the extension .hip.cpp to denote these files similar to what is done
for SYCL and following the recommendations of AMD
'''
class Preprocessor(config.compile.processor.Processor):
  '''The HIP preprocessor'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'HIPPP', 'HIPPPFLAGS', '.hip.cpp', '.hip.cpp')
    self.language        = 'HIP'
    self.includeDirectories = sets.Set()
    return

class Compiler(config.compile.processor.Processor):
  '''The HIP compiler'''
  def __init__(self, argDB, usePreprocessorFlags = True):
    config.compile.processor.Processor.__init__(self, argDB, 'HIPC', 'HIPFLAGS', '.hip.cpp', '.o')
    self.language        = 'HIP'
    self.requiredFlags[-1]  = '-c'
    self.outputFlag         = '-o'
    self.includeDirectories = sets.Set()
    if usePreprocessorFlags:
      self.flagsName.extend(Preprocessor(argDB).flagsName)

    return

  def getTarget(self, source):
    '''Return None for header files'''
    import os

    # HIP files are foo.hip.cpp
    base1, ext1 = os.path.splitext(source)
    base2, ext2 = os.path.splitext(base1)
    if ext1 == '.h':
      return None
    # If there is no .hip, then not a hip file
    if not ext2:
        return None
    return base2+'.o'

  def getCommand(self, sourceFiles, outputFile = None):
    '''If no outputFile is given, do not execute anything'''
    if outputFile is None:
      return 'true'
    return config.compile.processor.Processor.getCommand(self, sourceFiles, outputFile)

class Linker(config.compile.C.Linker):
  '''The HIP linker'''
  def __init__(self, argDB):
    self.compiler        = Compiler(argDB, usePreprocessorFlags = False)
    self.configLibraries = config.libraries.Configure(config.framework.Framework(clArgs = '', argDB = argDB, tmpDir = os.getcwd()))
    config.compile.processor.Processor.__init__(self, argDB, [self.compiler.name], ['HIPC_LINKER_FLAGS'], '.o', '.a')
    self.language   = 'HIP'
    self.outputFlag = '-o'
    self.libraries  = sets.Set()
    return

  def getExtraArguments(self):
    if not hasattr(self, '_extraArguments'):
      return ''
    return self._extraArguments
  extraArguments = property(getExtraArguments, config.compile.processor.Processor.setExtraArguments, doc = 'Optional arguments for the end of the command')

class SharedLinker(config.compile.C.SharedLinker):
  '''The HIP shared linker: Just use Cxx linker for now'''
  def __init__(self, argDB):
    config.compile.Cxx.SharedLinker.__init__(self, argDB)
    self.language = 'HIP'
    return

class StaticLinker(config.compile.C.StaticLinker):
  '''The HIP static linker, just use Cxx for now'''
  def __init__(self, argDB):
    config.compile.Cxx.StaticLinker.__init__(self, argDB)
    self.language = 'HIP'
    return

class DynamicLinker(config.compile.C.DynamicLinker):
  '''The HIP dynamic linker, just use Cxx for now'''
  def __init__(self, argDB):
    config.compile.Cxx.DynamicLinker.__init__(self, argDB)
    self.language = 'HIP'
    return
