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
    return

  def __str__(self):
    return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-log=<bool>',              nargs.ArgBool(None, 1, 'Activate logging code in PETSc'))
    help.addArgument('PETSc', '-with-info=<bool>',             nargs.ArgBool(None, 1, 'Activate PetscInfo() (i.e. -info)  code in PETSc'))
    help.addArgument('PETSc', '-with-ctable=<bool>',           nargs.ArgBool(None, 1, 'Activate CTABLE hashing for certain search functions - to conserve memory'))
    help.addArgument('PETSc', '-with-fortran-kernels=<bool>',  nargs.ArgBool(None, 0, 'Use Fortran for linear algebra kernels'))
    help.addArgument('PETSc', '-with-64-bit-indices=<bool>',   nargs.ArgBool(None, 0, 'Use 64 bit integers (long long) for indexing in vectors and matrices'))
    help.addArgument('PETSc', '-with-is-color-value-type=<char,short>',nargs.ArgString(None, 'short', 'char, short can store 256, 65536 colors'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.debugging   = framework.require('PETSc.utilities.debugging', self)
    self.compilers   = framework.require('config.compilers', self)
    self.libraries   = framework.require('config.libraries', self)
    self.types       = framework.require('config.types', self)
    self.scalarTypes = framework.require('PETSc.utilities.scalarTypes', self)
    return

  def isBGL(self):
    '''Returns true if compiler is IBM cross compiler for BGL'''
    if not hasattr(self, '_isBGL'):
      self.logPrint('**********Checking if running on BGL/IBM detected')
      if (self.libraries.check('', 'bgl_perfctr_void') or self.libraries.check('','ADIOI_BGL_Open')) and self.libraries.check('', '_xlqadd'):
        self.logPrint('*********BGL/IBM detected')
        self._isBGL = 1
      else:
        self.logPrint('*********BGL/IBM test failure')
        self._isBGL = 0
    return self._isBGL

  def configureLibraryOptions(self):
    '''Sets PETSC_USE_DEBUG, PETSC_USE_INFO, PETSC_USE_LOG, PETSC_USE_CTABLE and PETSC_USE_FORTRAN_KERNELS'''
    '''Also sets PETSC_AssertAlignx() in Fortran and PETSC_Alignx() in C for IBM BG/P compiler '''
    self.useLog   = self.framework.argDB['with-log']
    self.addDefine('USE_LOG',   self.useLog)

    if self.debugging.debugging:
      self.addDefine('USE_DEBUG', '1')

    self.useInfo   = self.framework.argDB['with-info']
    self.addDefine('USE_INFO',   self.useInfo)

    self.useCtable = self.framework.argDB['with-ctable']
    if self.useCtable:
      self.addDefine('USE_CTABLE', '1')

    # used in src/mat/impls/sbaij/seq/relax.h
    if not self.isBGL():
      self.addDefine('PETSC_USE_BACKWARD_LOOP','1')

    self.useFortranKernels = self.framework.argDB['with-fortran-kernels']
    if not hasattr(self.compilers, 'FC') and self.useFortranKernels:
      raise RuntimeError('Cannot use fortran kernels without a Fortran compiler')
    if self.useFortranKernels:
      self.addDefine('USE_FORTRAN_KERNELS', 1)
      if self.isBGL():
        self.addDefine('AssertAlignx(a,b)','call ALIGNX(a,b)')
      else:
        self.addDefine('AssertAlignx(a,b)','  ')

    if self.isBGL():
      self.addDefine('Alignx(a,b)','__alignx(a,b)')
    else:
      self.addDefine('Alignx(a,b)','  ')

    if self.framework.argDB['with-64-bit-indices']:
      self.integerSize = 64
      self.addDefine('USE_64BIT_INDICES', 1)
      if self.libraries.check('-lgcc_s.1', '__floatdidf'):
        self.compilers.LIBS += ' '+self.libraries.getLibArgument('-lgcc_s.1')
    else:
      self.integerSize = 32
    return

  def configureISColorValueType(self):
    '''Sets PETSC_IS_COLOR_VALUE_TYPE, MPIU_COLORING_VALUE, IS_COLORING_MAX required by ISColor'''
    self.isColorValueType  = self.framework.argDB['with-is-color-value-type']
    if self.isColorValueType != 'char' and self.isColorValueType != 'short':
      raise RuntimeError('Incorrect --with-is-color-value-type value specified. Can be either char or short. Specified value is :'+self.isColorValueType)
    if self.isColorValueType == 'char':
      max = pow(2,self.types.sizes['known-sizeof-char']*self.types.bits_per_byte)-1
      mpi_type = 'MPI_UNSIGNED_CHAR'
    else:
      max = pow(2,self.types.sizes['known-sizeof-short']*self.types.bits_per_byte)-1
      mpi_type = 'MPI_UNSIGNED_SHORT'

    self.framework.addDefine('MPIU_COLORING_VALUE',mpi_type)
    self.framework.addDefine('IS_COLORING_MAX',max)
    self.addDefine('IS_COLOR_VALUE_TYPE', self.isColorValueType)
    return

  def configure(self):
    self.executeTest(self.configureLibraryOptions)
    self.executeTest(self.configureISColorValueType)
    return
