import args
import config.compile.processor
import config.framework
import config.libraries
import os
import sys


try:
  import sets
except ImportError:
  import config.setsBackport as sets

class Preprocessor(config.compile.processor.Processor):
  '''The C preprocessor'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'CPP', 'CPPFLAGS', '.cpp', '.c')
    return

class Compiler(config.compile.processor.Processor):
  '''The C compiler'''
  def __init__(self, argDB):
    config.compile.processor.Processor.__init__(self, argDB, 'CC', 'CFLAGS', '.c', '.o')
    self.requiredFlags[-1]  = '-c'
    self.outputFlag         = '-o'
    self.includeDirectories = sets.Set()
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

class Linker(config.compile.processor.Processor):
  '''The C linker'''
  def __init__(self, argDB):
    self.compiler        = Compiler(argDB)
    self.configLibraries = config.libraries.Configure(config.framework.Framework(argDB = argDB))
    config.compile.processor.Processor.__init__(self, argDB, ['CC_LD', 'LD', self.compiler.name], 'LDFLAGS', '.o', '.a')
    self.outputFlag = '-o'
    self.libraries  = sets.Set()
    return

  def setArgDB(self, argDB):
    args.ArgumentProcessor.setArgDB(self, argDB)
    self.configLibraries.argDB           = argDB
    self.configLibraries.framework.argDB = argDB
    return
  argDB = property(args.ArgumentProcessor.getArgDB, setArgDB, doc = 'The RDict argument database')

  def getFlags(self):
    '''Returns a string with the flags specified for running this processor.'''
    if not hasattr(self, '_flags'):
      flagsName = self.flagsName[:]
      if self.name == self.compiler.name:
        flagsName.extend(self.compiler.flagsName)
      if hasattr(self, 'configCompilers'):
        flags = ' '.join([getattr(self.configCompilers, name) for name in flagsName])
      else:
        flags = ' '.join([self.argDB[name] for name in flagsName])
      return flags
    return self._flags
  flags = property(getFlags, config.compile.processor.Processor.setFlags, doc = 'The flags for the executable')

  def getExtraArguments(self):
    if not hasattr(self, '_extraArguments'):
      if not 'LIBS' in self.argDB:
        return ''
      return self.argDB['LIBS']
    return self._extraArguments
  extraArguments = property(getExtraArguments, config.compile.processor.Processor.setExtraArguments, doc = 'Optional arguments for the end of the command')

  def getTarget(self, source, shared):
    import os
    import sys

    base, ext = os.path.splitext(source)
    if sys.platform[:3] == 'win' or sys.platform == 'cygwin':
      return base+'.exe'
    return base

class SharedLinker(config.compile.processor.Processor):
  '''The C linker'''
  def __init__(self, argDB):
    self.compiler = Compiler(argDB)
    self.configLibraries = config.libraries.Configure(config.framework.Framework(argDB = argDB))
    config.compile.processor.Processor.__init__(self, argDB, ['LD_SHARED', self.compiler.name], 'LDFLAGS', '.o', None)
    self.outputFlag = '-o'
    self.libraries  = sets.Set()
    return

  def getFlags(self):
    '''Returns a string with the flags specified for running this processor.'''
    if not hasattr(self, '_flags'):
      flagsName = self.flagsName[:]
      if self.name == self.compiler.name:
        flagsName.extend(self.compiler.flagsName)
      if hasattr(self, 'configCompilers'):
        flags = [getattr(self.configCompilers, name) for name in flagsName]
        flags.extend(self.configCompilers.setCompilers.sharedLibraryFlags)
      else:
        flags = [self.argDB[name] for name in flagsName]
      return ' '.join(flags)
    return self._flags
  flags = property(getFlags, config.compile.processor.Processor.setFlags, doc = 'The flags for the executable')

  def getExtraArguments(self):
    if not hasattr(self, '_extraArguments'):
      if not 'LIBS' in self.argDB:
        return ''
      return self.argDB['LIBS']
    return self._extraArguments
  extraArguments = property(getExtraArguments, config.compile.processor.Processor.setExtraArguments, doc = 'Optional arguments for the end of the command')

  def getTarget(self, source, shared):
    base, ext = os.path.splitext(source)
    if hasattr(self, 'configCompilers'):
      base += '.'+self.configCompilers.setCompilers.sharedLibraryExt
    else:
      base += '.'+self.argDB['LD_SHARED_SUFFIX']
    return base

class StaticLinker(SharedLinker):
  '''The C static linker, which is not really a linker, but we are hacking it in here'''
  def getOutputFlag(self):
    if hasattr(self, 'configCompilers'):
      return self.configCompilers.AR_FLAGS
    return self.argDB['AR_FLAGS']

  def getRanlib(self):
    '''Returns the processor executable'''
    if hasattr(self, 'configCompilers'):
      return self.configCompilers.RANLIB
    return self.argDB['RANLIB']

  def getCommand(self, sourceFiles, outputFile):
    self.outputFlag = self.getOutputFlag()
    self.libraries.clear()
    archiveCmd = SharedLinker.getCommand(self, sourceFiles, outputFile)
    ranlibCmd = ' '.join([';', self.getRanlib(), outputFile])
    return archiveCmd+ranlibCmd

  def getTarget(self, source, shared):
    base, ext = os.path.splitext(source)
    if hasattr(self,'configCompilers'):
      base += '.'+self.configCompilers.AR_LIB_SUFFIX
    else:
      base += '.'+self.argDB['AR_LIB_SUFFIX']
    return base
