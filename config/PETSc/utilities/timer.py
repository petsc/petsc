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
    help.addArgument('PETSc', '-with-timer=<default,mpi,ibm,dec,asci-red,nt>',  nargs.ArgString(None, None, 'Use high precision timer'))
    return

  def setupDependencies(self, framework):
    self.libraries = framework.require('config.libraries', self)
    return

  def isCrayMPI(self):
    '''Returns true if using Cray MPI'''
    if self.libraries.check('', 'MPI_CRAY_barrier'):
      self.logPrint('Cray-MPI detected')
      return 1
    else:
      self.logPrint('Cray-MPI test failure')
      return 0

  def configureTimers(self):
    '''Sets PETSC_HAVE_FAST_MPI_WTIME PETSC_USE_READ_REAL_TIME PETSC_USE_GETCLOCK PETSC_USE_DCLOCK PETSC_USE_NT_TIME.'''
    if 'with-timer' in self.framework.argDB:
      self.useTimer = self.framework.argDB['with-timer'].lower()
    elif self.isCrayMPI():
      self.useTimer = 'mpi'
    else:
      self.useTimer = 'default'

    # now check the timer
    if self.useTimer == 'mpi':
      self.addDefine('HAVE_FAST_MPI_WTIME', 1)
    elif self.useTimer == 'ibm':
      self.addDefine('USE_READ_REAL_TIME', 1)
    elif self.useTimer == 'dec':
      self.addDefine('USE_GETCLOCK', 1)
    elif self.useTimer == 'asci-red':
      self.addDefine('USE_DCLOCK', 1)
    elif self.useTimer == 'nt':
      self.addDefine('USE_NT_TIME', 1)
    elif self.useTimer != 'default':
      raise RuntimeError('Unknown Timer type specified :'+self.framework.argDB['with-timer'])
    return
  
  def configure(self):
    self.executeTest(self.configureTimers)
    return
