#!/usr/bin/env python
from __future__ import generators
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def __str1__(self):
    desc = []
    if hasattr(self, 'integerSize'):
      desc.append('  Integer size: ' + str(self.integerSize//8) + ' bytes')
    return '\n'.join(desc)+'\n'

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-64-bit-indices=<bool>',   nargs.ArgBool(None, 0, 'Use 64 bit integers (long long) for indexing in vectors and matrices'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers', self)
    self.libraries    = framework.require('config.libraries', None)
    self.compilers    = framework.require('config.compilers', None)
    return

  def fortranPromoteInteger(self):
      self.pushLanguage('FC')
      flags = self.getCompilerFlags()
      self.popLanguage()
      #   ifort compiler flag             gfortran compiler flag
      if '-integer-size 64' in flags or '-fdefault-integer-8' in flags:
        return 1
      return 0

  def configureIndexSize(self):
    if self.framework.argDB['with-64-bit-indices']:
      self.integerSize = 64
      self.addDefine('USE_64BIT_INDICES', 1)
      if self.libraries.check('-lgcc_s.1', '__floatdidf'):
        self.compilers.LIBS += ' '+self.libraries.getLibArgument('-lgcc_s.1')
      self.addMakeMacro('PETSC_INDEX_SIZE', '64')
      if self.fortranPromoteInteger():
        self.addDefine('PROMOTE_FORTRAN_INTEGER', 1)
        self.logPrintBox('Warning: you have a Fortran compiler option to promote integer to 8 bytes.\nThis is fragile and not supported by the MPI standard.\nYou must ensure in your code that all calls to MPI routines pass 4-byte integers.')
    else:
      self.integerSize = 32
      self.addMakeMacro('PETSC_INDEX_SIZE', '32')
      if self.fortranPromoteInteger():
        raise RuntimeError('Fortran compiler flag to promote integers to 8 bytes has been set, but PETSc is being built with 4-byte integers.')
    return

  def configure(self):
    self.executeTest(self.configureIndexSize)
    return
