import config.base
import os
import re

class Package(config.base.Configure):
  def __init__(self, framework,name):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',self)
    self.libraries    = self.framework.require('config.libraries',self)
    self.arch         = self.framework.require('PETSc.utilities.arch', self)
    self.mpi          = self.framework.require('PETSc.packages.MPI',self)
    self.blasLapack   = self.framework.require('PETSc.packages.BlasLapack',self)
    self.functions    = self.framework.require('config.functions',         self)
    self.found        = 0
    self.lib          = []
    self.include      = []
    self.name         = name
    self.PACKAGE      = self.name.upper()
    self.package      = self.name.lower()


  def configure(self):
    if self.framework.argDB['download-'+self.package]:
      self.framework.argDB['with-'+self.package] = 1
    if 'with-'+self.package+'-dir' in self.framework.argDB:
      self.framework.argDB['with-'+self.package] = 1
    if self.framework.argDB['with-'+self.package]:
      if self.mpi.usingMPIUni:
        raise RuntimeError('Cannot use '+self.name+' with MPIUNI, you need a real MPI')
      if self.framework.argDB['with-64-bit-ints']:
        raise RuntimeError('Cannot use '+self.name+' with 64 bit integers, it is not coded for this capability')    
      self.executeTest(self.configureLibrary)
    return

