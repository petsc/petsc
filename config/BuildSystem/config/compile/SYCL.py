import args
import config.compile.processor
import config.compile.C
import config.framework
import config.libraries
import os
import sys

import config.setsOrdered as sets

'''
SYCL is a C++ compiler with extensions to support the SYCL programming model.
Because of it's slowness, and in some ways the extensions make it a new language,
we have a separate compiler for it.
We use the extension .sycl.cxx to denote these files similar to what is done
for HIP (which is also C++ and has similar issue).
'''
class Preprocessor(config.compile.processor.Processor):
  '''The SYCL preprocessor'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'SYCLPP', 'SYCLPPFLAGS', '.sycl.cxx', '.sycl.cxx')
    self.language        = 'SYCL'
    self.includeDirectories = sets.Set()
    return

class Compiler(config.compile.processor.Processor):
  '''The SYCL compiler'''
  def __init__(self, argDB, usePreprocessorFlags = True):
    config.compile.processor.Processor.__init__(self, argDB, 'SYCLC', 'SYCLFLAGS', '.sycl.cxx', '.o')
    self.language        = 'SYCL'
    self.requiredFlags[-1]  = '-c'
    self.outputFlag         = '-o'
    self.includeDirectories = sets.Set()
    if usePreprocessorFlags:
      self.flagsName.extend(Preprocessor(argDB).flagsName)

    return

  def getTarget(self, source):
    '''Return the object file name for 'source'; None if 'source' is a header file'''
    import os

    # SYCL files are foo.sycl.cxx
    base1, ext1 = os.path.splitext(source)
    base2, ext2 = os.path.splitext(base1)
    if ext1 in ['.h', '.hh', '.hpp']:
      return None
    # If there is no .sycl, then not a sycl file
    if ext2 != '.sycl':
        return None
    return base2+'.o'

  def getCommand(self, sourceFiles, outputFile = None):
    '''If no outputFile is given, do not execute anything'''
    if outputFile is None:
      return 'true'
    return config.compile.processor.Processor.getCommand(self, sourceFiles, outputFile)

class Linker(config.compile.C.Linker):
  '''The SYCL linker'''
  def __init__(self, argDB):
    self.compiler        = Compiler(argDB, usePreprocessorFlags = False)
    self.configLibraries = config.libraries.Configure(config.framework.Framework(clArgs = '', argDB = argDB, tmpDir = os.getcwd()))
    config.compile.processor.Processor.__init__(self, argDB,
                                                [self.compiler.name], ['SYCLC_LINKER_FLAGS'], '.o', '.a')
    self.language   = 'SYCL'
    self.outputFlag = '-o'
    self.libraries  = sets.Set()
    return

  def getExtraArguments(self):
    if not hasattr(self, '_extraArguments'):
      return ''
    return self._extraArguments
  extraArguments = property(getExtraArguments, config.compile.processor.Processor.setExtraArguments, doc = 'Optional arguments for the end of the command')

class SharedLinker(config.compile.C.SharedLinker):
  '''The SYCL shared linker: Just use Cxx linker for now'''
  def __init__(self, argDB):
    config.compile.Cxx.SharedLinker.__init__(self, argDB)
    self.language = 'SYCL'
    return

class StaticLinker(config.compile.C.StaticLinker):
  '''The SYCL static linker, just use Cxx for now'''
  def __init__(self, argDB):
    config.compile.Cxx.StaticLinker.__init__(self, argDB)
    self.language = 'SYCL'
    return

class DynamicLinker(config.compile.C.DynamicLinker):
  '''The SYCL dynamic linker, just use Cxx for now'''
  def __init__(self, argDB):
    config.compile.Cxx.DynamicLinker.__init__(self, argDB)
    self.language = 'SYCL'
    return
