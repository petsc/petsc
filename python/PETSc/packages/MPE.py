import config.base

import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.argDB        = framework.argDB
    self.framework.require('PETSc.packages.MPI', self)
    return

  def setOutput(self):
    '''Only null values now'''
    self.addSubstitution('MPE_INCLUDE', '')
    self.addSubstitution('MPE_LIB',     '')
    return

  def configure(self):
    self.setOutput()
    return
