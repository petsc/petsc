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
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.arch           = framework.require('PETSc.options.arch', self)
    self.scalartypes    = framework.require('PETSc.options.scalarTypes', self)
    self.indextypes     = framework.require('PETSc.options.indexTypes', self)
    self.datafilespath  = framework.require('PETSc.options.dataFilesPath', self)
    self.compilers      = framework.require('config.compilers', self)
    self.mpi            = framework.require('config.packages.MPI', self)
    self.elemental      = framework.require('config.packages.elemental', self)
    self.superlu_dist   = framework.require('config.packages.SuperLU_DIST', self)
    self.x              = framework.require('config.packages.X', self)
    self.fortrancpp     = framework.require('PETSc.options.fortranCPP', self)
    self.libraryOptions = framework.require('PETSc.options.libraryOptions', self)
    return

  def configureRegression(self):
    '''Output a file listing the jobs that should be run by the PETSc buildtest'''
    jobs  = []    # Jobs can be run always
    rjobs = []    # Jobs can only be run with real numbers; i.e. NOT  complex
    ejobs = []    # Jobs that require an external package install (also cannot work with complex)
    if self.mpi.usingMPIUni:
      jobs.append('C_X_MPIUni')
      if hasattr(self.compilers, 'FC'):
        jobs.append('Fortran_MPIUni')
    else:
      jobs.append('C')
      if self.libraryOptions.useInfo:
        jobs.append('C_Info')
      if not self.scalartypes.precision == 'single':
        jobs.append('C_NotSingle')
      if hasattr(self.compilers, 'CXX'):
        rjobs.append('Cxx')
      if self.x.found:
        jobs.append('C_X')
      if hasattr(self.compilers, 'FC'):
        jobs.append('Fortran')
        if not self.scalartypes.precision == 'single':
          jobs.append('Fortran_NotSingle')
        if self.compilers.fortranIsF90FreeForm and self.compilers.fortranIsF90:
          rjobs.append('F90')
          if self.datafilespath.datafilespath and self.scalartypes.precision == 'double' and self.indextypes.integerSize == 32:
            jobs.append('F90_DataTypes')
          if self.libraryOptions.useThreadSafety:
            jobs.append('F90_Threadsafety')
          if not self.scalartypes.precision == 'single':
            jobs.append('F90_NotSingle')
          if self.scalartypes.scalartype.lower() == 'complex':
            rjobs.append('F90_Complex')
          else:
            rjobs.append('F90_NoComplex')
        if self.compilers.fortranIsF90FreeForm and self.compilers.fortranIsF2003:
          rjobs.append('F2003')
        if self.scalartypes.scalartype.lower() == 'complex':
          rjobs.append('Fortran_Complex')
        else:
          rjobs.append('Fortran_NoComplex')
          if not self.scalartypes.precision == 'single':
            jobs.append('Fortran_NoComplex_NotSingle')
      if self.scalartypes.scalartype.lower() == 'complex':
        rjobs.append('C_Complex')
        if self.datafilespath.datafilespath and self.scalartypes.precision == 'double' and self.indextypes.integerSize == 32:
          for j in self.framework.packages:
            if j.PACKAGE in ['SUPERLU_DIST','ELEMENTAL']:
                ejobs.append(j.PACKAGE+'_COMPLEX_DATAFILESPATH')
      else:
        rjobs.append('C_NoComplex')
        if not self.scalartypes.precision == 'single':
          jobs.append('C_NoComplex_NotSingle')
        if self.datafilespath.datafilespath and self.scalartypes.precision == 'double' and self.indextypes.integerSize == 32:
          rjobs.append('DATAFILESPATH')
          if hasattr(self.compilers, 'CXX'):
            rjobs.append('Cxx_DATAFILESPATH')
          if hasattr(self.compilers, 'FC'):
            rjobs.append('Fortran_DATAFILESPATH')
          for j in self.framework.packages:
            if j.hastestsdatafiles:
                ejobs.append(j.PACKAGE+'_DATAFILESPATH')
        if self.scalartypes.precision == 'double' and self.indextypes.integerSize == 32:
          rjobs.append('DOUBLEINT32')
          if hasattr(self.compilers, 'FC'):
            rjobs.append('Fortran_DOUBLEINT32')
      # add jobs for each external package BUGBUGBUG may be run before all packages
      # Note: do these tests only for non-complex builds
      if self.scalartypes.scalartype.lower() != 'complex':
        for i in self.framework.packages:
          if i.hastests:
            ejobs.append(i.PACKAGE)
          # horrible python here
          if i.PACKAGE == 'MOAB':
            for j in self.framework.packages:
              if j.PACKAGE == 'HDF5':
                ejobs.append('MOAB_HDF5')
          if i.PACKAGE == 'TRIANGLE':
            for j in self.framework.packages:
              if j.PACKAGE == 'HDF5':
                ejobs.append('TRIANGLE_HDF5')
      else:
        for i in self.framework.packages:
          if i.PACKAGE in ['FFTW','SUPERLU_DIST']:
            jobs.append(i.PACKAGE+ '_COMPLEX')
          elif i.PACKAGE in ['STRUMPACK','ELEMENTAL']:
            jobs.append(i.PACKAGE)
      for i in self.framework.packages:
        if i.PACKAGE == 'CUDA':
          jobs.append('VECCUDA')
          if self.scalartypes.scalartype.lower() == 'complex':
            rjobs.append('VECCUDA_Complex')
          else:
            rjobs.append('VECCUDA_NoComplex')
            if self.datafilespath.datafilespath and self.scalartypes.precision == 'double' and self.indextypes.integerSize == 32:
              rjobs.append('VECCUDA_DATAFILESPATH')

    self.addMakeMacro('TEST_RUNS',' '.join(jobs)+' '+' '.join(ejobs)+' '+' '.join(rjobs))
    return


  def configure(self):
    self.executeTest(self.configureRegression)
    return
