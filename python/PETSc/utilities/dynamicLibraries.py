#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.useDynamic   = 0
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-petsc-dynamic', nargs.ArgBool(None, 0, 'Make PETSc libraries dynamic'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.setCompilers = framework.require('config.setCompilers', self)
    self.shared       = framework.require('PETSc.utilities.sharedLibraries', self)
    return

  def configureDynamicLibraries(self):
    '''Checks whether dynamic libraries should be used, for which you must
      - Specify --with-petsc-dynamic
      - Have found a working dynamic linker (with dlfcn.h and libdl)
    Defines PETSC_USE_DYNAMIC_LIBRARIES if they are used'''
    self.useDynamic = self.argDB['with-petsc-dynamic'] and self.shared.useShared and self.setCompilers.dynamicLibraries
    if self.useDynamic:
      self.addDefine('USE_DYNAMIC_LIBRARIES', 1)
    else:
      self.logPrint('Dynamic libraries - disabled')
    return

  def configure(self):
    self.executeTest(self.configureDynamicLibraries)
    return
