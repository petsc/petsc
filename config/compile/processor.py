class Processor(object):
  '''This class is intended to provide a basis for language operations, such as compiling and linking. Each operation will have a Processor.'''
  def __init__(self, argDB, name, flagsName, sourceExtension, targetExtension):
    self.language        = 'C'
    self.argDB           = argDB
    if isinstance(name, list):
      for n in name:
        if n in self.argDB:
          self.name      = n
          break
    else:
      self.name          = name
    if isinstance(flagsName, list):
      self.flagsName     = flagsName
    else:
      self.flagsName = [flagsName]
    self.requiredFlags   = ['']
    self.outputFlag      = ''
    self.sourceExtension = sourceExtension
    self.targetExtension = targetExtension
    return

  def pushRequiredFlags(self, flags):
    self.requiredFlags.append(flags)
    return
  def popRequiredFlags(self):
    self.requiredFlags.pop()
    return

  def checkSetup(self):
    '''Check that this program has been specified. We assume that configure has checked its viability.'''
    if not hasattr(self, 'name'):
      raise RuntimeError('No valid argument name set for '+self.language+' '+self.__class__.__name__.lower())
    if not self.name in self.argDB:
      raise RuntimeError('Could not find a '+self.language+' '+self.__class__.__name__.lower()+'. Please set with the option --with-'+self.name.lower()+' or -'+self.name+' and load the config.compilers module.')
    return

  def getFlags(self):
    '''Returns a string with the flags specified for running this processor.'''
    if not hasattr(self, '_flags'):
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

  def getTarget(self, source):
    '''Returns the default target for the given source file, or None'''
    return None

  def getCommand(self, sourceFiles, outputFile = None):
    '''Returns a shell command as a string which will invoke the processor on sourceFiles, producing outputFile if given'''
    if isinstance(sourceFiles, str):
      sourceFiles = [sourceFiles]
    cmd = [self.argDB[self.name], self.requiredFlags[-1]]
    if not outputFile is None:
      cmd.extend([self.outputFlag, outputFile])
    cmd.append(self.flags)
    cmd.extend(sourceFiles)
    cmd.append(self.extraArguments)
    return ' '.join(cmd)
