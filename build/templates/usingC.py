import base

import os

class UsingC (base.Base):
  def __init__(self, sourceDB, project, usingSIDL):
    import config.base

    base.Base.__init__(self)
    self.language    = 'C'
    self.sourceDB    = sourceDB
    self.project     = project
    self.usingSIDL   = usingSIDL
    self.language    = 'C'
    self.linker      = None
    self.linkerFlags = None
    self.configBase  = config.base.Configure(self)
    self.configBase.setLanguage(self.language)
    return

  def isCompiled(self):
    '''Returns True is source needs to be compiled in order to execute'''
    return 1

  def getCompileSuffix(self):
    '''Return the suffix for compilable files (.c)'''
    return '.c'

  def getLinker(self):
    if self._linker is None:
      return self.configBase.getLinker()
    return self._linker

  def setLinker(self, linker):
    self._linker = linker
  linker = property(getLinker, setLinker, doc = 'The linker corresponding to the C compiler')

  def getLinkerFlags(self):
    if self._linkerFlags is None:
      self.configBase.getLinker()
      return self.configBase.linkerFlags
    return self._linkerFlags

  def setLinkerFlags(self, flags):
    self._linkerFlags = flags
  linkerFlags = property(getLinkerFlags, setLinkerFlags, doc = 'The flags for the C linker')
