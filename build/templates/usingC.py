import base

import os

class UsingC (base.Base):
  def __init__(self, sourceDB, project, usingSIDL):
    base.Base.__init__(self)
    self.sourceDB  = sourceDB
    self.project   = project
    self.usingSIDL = usingSIDL
    self.language  = 'C'
    return

  def isCompiled(self):
    '''Returns True is source needs to be compiled in order to execute'''
    return 1

  def getCompileSuffix(self):
    '''Return the suffix for compilable files (.c)'''
    return '.c'
