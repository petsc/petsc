#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.arch          = self.framework.require('PETSc.utilities.arch', self)
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-lib-debug=<bool>',            nargs.ArgBool(None, 1, 'Activate debugging code in PETSc'))
    help.addArgument('PETSc', '-with-lib-log=<bool>',              nargs.ArgBool(None, 1, 'Activate logging code in PETSc'))
    help.addArgument('PETSc', '-with-lib-stack=<bool>',            nargs.ArgBool(None, 1, 'Activate manual stack tracing code in PETSc'))
    help.addArgument('PETSc', '-with-lib-ctable=<bool>',           nargs.ArgBool(None, 1, 'Use CTABLE hashing for certain search functions - to conserve memory'))
    help.addArgument('PETSc', '-with-lib-fortran-kernels=<bool>',  nargs.ArgBool(None, 0, 'Use Fortran for linear algebra kernels'))
    help.addArgument('PETSc', '-with-64-bit-ints=<bool>',      nargs.ArgBool(None, 0, 'Use 64 bit integers (long long) for indexing in vectors and matrices'))
    return

  def configureLibraryOptions(self):
    '''Sets PETSC_USE_DEBUG, PETSC_USE_LOG, PETSC_USE_STACK, PETSC_USE_CTABLE and PETSC_USE_FORTRAN_KERNELS'''
    self.useDebug = self.framework.argDB['with-lib-debug']
    self.addDefine('USE_DEBUG', self.useDebug)
    self.useLog   = self.framework.argDB['with-lib-log']
    self.addDefine('USE_LOG',   self.useLog)
    self.useStack = self.framework.argDB['with-lib-stack']
    self.addDefine('USE_STACK', self.useStack)
    self.useCtable = self.framework.argDB['with-lib-ctable']
    self.addDefine('USE_CTABLE', self.useCtable)
    self.useFortranKernels = self.framework.argDB['with-lib-fortran-kernels']
    self.addDefine('USE_FORTRAN_KERNELS', self.useFortranKernels)
    if self.framework.argDB['with-64-bit-ints']:
      self.addDefine('USE_64BIT_INT', 1)
    else:
      self.addDefine('USE_32BIT_INT', 1)
    return


  def configure(self):
    self.executeTest(self.configureLibraryOptions)
    return
