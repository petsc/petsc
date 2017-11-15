import args
import logger

class Processor(logger.Logger):
  '''This class is intended to provide a basis for language operations, such as compiling and linking. Each operation will have a Processor.'''
  def __init__(self, argDB, name, flagsName, sourceExtension, targetExtension):
    logger.Logger.__init__(self, None, argDB)
    self.language        = 'C'
    self.name            = name
    if isinstance(flagsName, list):
      self.flagsName     = flagsName
    else:
      self.flagsName = [flagsName]
    self.requiredFlags   = ['']
    self.outputFlag      = ''
    self.sourceExtension = sourceExtension
    self.targetExtension = targetExtension
    return

  def copy(self, other):
    other.language = self.language
    other.name = self.name
    other.flagsName = self.flagsName[:]
    other.requiredFlags = self.requiredFlags[:]
    other.outputFlag = self.outputFlag
    other.sourceExtension = self.sourceExtension
    other.targetExtension = self.targetExtension
    return

  def setArgDB(self, argDB):
    args.ArgumentProcessor.setArgDB(self, argDB)
    if hasattr(self, 'configCompilers'):
      if not self.configCompilers.argDB == argDB:
        self.configCompilers.argDB = argDB
      if not self.configCompilers.framework.argDB == argDB:
        self.configCompilers.framework.argDB = argDB
    if hasattr(self, 'configLibraries'):
      if not self.configLibraries.argDB == argDB:
        self.configLibraries.argDB = argDB
      if not self.configLibraries.framework.argDB == argDB:
        self.configLibraries.framework.argDB = argDB
    if hasattr(self, 'versionControl'):
      self.versionControl.argDB = argDB
    return
  argDB = property(args.ArgumentProcessor.getArgDB, setArgDB, doc = 'The RDict argument database')

  def getName(self):
    if not hasattr(self, '_name'):
      raise RuntimeError('No valid argument name set for '+self.language+' '+self.__class__.__name__.lower())
    if isinstance(self._name, list):
      for n in self._name:
        if hasattr(self, 'configCompilers') and hasattr(self.configCompilers, n):
          self._name = n
          break
        if n in self.argDB:
          self._name = n
          break
      if isinstance(self._name, list):
        if hasattr(self, 'configCompilers'):
          raise RuntimeError('Name '+str(self._name)+' was not found in RDict or configure data')
        else:
          raise RuntimeError('Name '+str(self._name)+' was not found in RDict')
    return self._name

  def setName(self, name):
    self._name = name
    return
  name = property(getName, setName, doc = 'The name of the processor in RDict')

  def pushRequiredFlags(self, flags):
    self.requiredFlags.append(flags)
    return
  def popRequiredFlags(self):
    self.requiredFlags.pop()
    return

  def checkSetup(self):
    '''Check that this program has been specified. We assume that configure has checked its viability.'''
    if hasattr(self, 'configCompilers') and hasattr(self.configCompilers, self.name):
      pass
    elif not self.name in self.argDB:
      raise RuntimeError('Could not find a '+self.language+' '+self.__class__.__name__.lower()+'. Please set with the option --with-'+self.name.lower()+' or -'+self.name+' and load the config.compilers module.')
    return

  def getProcessor(self):
    '''Returns the processor executable'''
    if hasattr(self, 'configCompilers'):
      return getattr(self.configCompilers, self.name)
    return self.argDB[self.name]

  def getFlags(self):
    '''Returns a string with the flags specified for running this processor.'''
    # can't change _flags - as this broke triangle build
    if not hasattr(self, '_flags'):
      if hasattr(self, 'configCompilers'):
        flags = ' '.join([getattr(self.configCompilers, name) for name in self.flagsName])
      else:
        flags = ' '.join([self.argDB[name] for name in self.flagsName])
      return flags
    return self._flags
  def setFlags(self, flags):
    self._flags = flags
  flags = property(getFlags, setFlags, doc = 'The flags for the executable')

  def getExtraArguments(self):
    '''Returns a string which should be appended to every command'''
    if not hasattr(self, '_extraArguments'):
      return ''
    return self._extraArguments
  def setExtraArguments(self, extraArguments):
    self._extraArguments = extraArguments
    return
  extraArguments = property(getExtraArguments, setExtraArguments, doc = 'Optional arguments for the end of the command')

  def getTarget(self, source, shared = 0):
    '''Returns the default target for the given source file, or None'''
    return None

  def getCommand(self, sourceFiles, outputFile = None):
    '''Returns a shell command as a string which will invoke the processor on sourceFiles, producing outputFile if given'''
    if isinstance(sourceFiles, str):
      sourceFiles = [sourceFiles]
    cmd = [self.getProcessor()]
    cmd.append(self.requiredFlags[-1])
    if not outputFile is None:
      cmd.extend([self.outputFlag, outputFile])
    if hasattr(self, 'includeDirectories'):
      cmd.extend(['-I'+inc for inc in self.includeDirectories])
    cmd.append(self.flags)
    cmd.extend(sourceFiles)
    cmd.append(self.extraArguments)
    if hasattr(self, 'libraries') and hasattr(self, 'configLibraries'):
      self.configLibraries.pushLanguage(self.language)
      cmd.extend([self.configLibraries.getLibArgument(lib) for lib in self.libraries])
      self.configLibraries.popLanguage()
    return ' '.join(cmd)
