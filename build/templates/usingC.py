import base

import os

class UsingC (base.Base):
  def __init__(self, sourceDB, project, usingSIDL):
    base.Base.__init__(self)
    self.sourceDB  = sourceDB
    self.project   = project
    self.usingSIDL = usingSIDL
    self.language  = 'C'
    self.linker    = None
    return

  def isCompiled(self):
    '''Returns True is source needs to be compiled in order to execute'''
    return 1

  def getCompileSuffix(self):
    '''Return the suffix for compilable files (.c)'''
    return '.c'

  def getLinker(self):
    if self._linker is None:
      if 'CC_LD' in self.argDB:
        return self.argDB['CC_LD']
      elif 'LD' in self.argDB:
        return self.argDB['LD']
      else:
        return self.argDB['CC']
    return self._linker

  def setLinker(self, linker):
    self._linker = linker
  linker = property(getLinker, setLinker, doc = 'The linker corresponding to the C compiler')
