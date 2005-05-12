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
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.arch          = framework.require('PETSc.utilities.arch', self)
    self.bmake         = framework.require('PETSc.utilities.bmakeDir', self)    
    self.datafilespath = framework.require('PETSc.utilities.dataFilesPath', self)
    self.compilers     = framework.require('config.compilers', self)
    self.mpi           = framework.require('PETSc.packages.MPI', self)
    self.x11           = framework.require('PETSc.packages.X11', self)        
    return

  def configureRegression(self):
    '''Output a file listing the jobs that should be run by the PETSc buildtest'''
    jobs  = []    # Jobs can be run on with alwaysw
    rjobs = []    # Jobs can only be run with real numbers; i.e. NOT  complex
    ejobs = []    # Jobs that require an external package install (also cannot work with complex)
    if self.mpi.usingMPIUni:
      jobs.append('C_X11_MPIUni')
      if hasattr(self.compilers, 'FC'):
        jobs.append('Fortran_MPIUni')
    else:
      jobs.append('C')
      if self.x11.foundX11:
        jobs.append('C_X11')
      if hasattr(self.compilers, 'FC'):
        jobs.append('Fortran')
        rjobs.append('Fortran_NoComplex')
      if self.datafilespath.datafilespath:
        rjobs.append('C_NoComplex')
      # add jobs for each external package BUGBUGBUG may be run before all packages
      for i in self.framework.packages:
        ejobs.append(i.name.upper())
    if os.path.isfile(os.path.join(self.bmake.bmakeDir, 'jobs')):
      try:
        os.unlink(os.path.join(self.bmake.bmakeDir, 'jobs'))
      except:
        raise RuntimeError('Unable to remove file '+os.path.join(self.bmake.bmakeDir, 'jobs')+'. Did a different user create it?')
    jobsFile  = file(os.path.abspath(os.path.join(self.bmake.bmakeDir, 'jobs')), 'w')
    jobsFile.write(' '.join(jobs)+'\n')
    jobsFile.close()
    self.framework.actions.addArgument('PETSc', 'File creation', 'Generated list of jobs for testing in '+os.path.join(self.bmake.bmakeDir,'jobs'))
    if os.path.isfile(os.path.join(self.bmake.bmakeDir, 'ejobs')):
      try:
        os.unlink(os.path.join(self.bmake.bmakeDir, 'ejobs'))
      except:
        raise RuntimeError('Unable to remove file '+os.path.join(self.bmake.bmakeDir, 'ejobs')+'. Did a different user create it?')
    ejobsFile = file(os.path.abspath(os.path.join(self.bmake.bmakeDir, 'ejobs')), 'w')
    ejobsFile.write(' '.join(ejobs)+'\n')
    ejobsFile.close()
    self.framework.actions.addArgument('PETSc', 'File creation', 'Generated list of jobs for testing in '+os.path.join(self.bmake.bmakeDir,'ejobs'))
    if os.path.isfile(os.path.join(self.bmake.bmakeDir, 'rjobs')):
      try:
        os.unlink(os.path.join(self.bmake.bmakeDir, 'rjobs'))
      except:
        raise RuntimeError('Unable to remove file '+os.path.join(self.bmake.bmakeDir, 'rjobs')+'. Did a different user create it?')
    rjobsFile = file(os.path.abspath(os.path.join(self.bmake.bmakeDir, 'rjobs')), 'w')
    rjobsFile.write(' '.join(rjobs)+'\n')
    rjobsFile.close()
    self.framework.actions.addArgument('PETSc', 'File creation', 'Generated list of jobs for testing in '+os.path.join(self.bmake.bmakeDir,'rjobs'))
    return

  def configure(self):
    self.executeTest(self.configureRegression)
    return
