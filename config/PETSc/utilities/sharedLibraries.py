#!/usr/bin/env python
from __future__ import generators
import user
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.useShared    = 0
    self.useDynamic   = 0
    return

  def __str1__(self):
    if not hasattr(self, 'useShared') or not hasattr(self, 'useDynamic'):
      return ''
    txt = ''
    if self.useShared:
      txt += '  shared libraries: enabled\n'
    else:
      txt += '  shared libraries: disabled\n'
    if self.useDynamic:
      txt += '  dynamic libraries: enabled\n'
    else:
      txt += '  dynamic libraries: disabled\n'
    return txt

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-shared', nargs.ArgBool(None, 0, 'Make PETSc libraries shared'))
    help.addArgument('PETSc', '-with-dynamic', nargs.ArgBool(None, 0, 'Make PETSc libraries dynamic'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.arch = framework.require('PETSc.utilities.arch', self)
    self.setCompilers = framework.require('config.setCompilers', self)
    return

  def configureSharedLibraries(self):
    '''Checks whether dynamic libraries should be used, for which you must
      - Specify --with-shared
      - Have found a working dynamic linker
    Defines PETSC_USE_SHARED_LIBRARIES if they are used'''
    self.useShared = (self.argDB['with-dynamic'] or self.argDB['with-shared']) and not self.setCompilers.staticLibraries
    if self.useShared:
      if config.setCompilers.Configure.isSolaris() and config.setCompilers.Configure.isGNU(self.framework.getCompiler()):
        self.addMakeRule('shared_arch','shared_'+self.arch.hostOsBase+'gnu')
      elif '-qmkshrobj' in self.setCompilers.sharedLibraryFlags:
        self.addMakeRule('shared_arch','shared_linux_ibm')
      else:
        self.addMakeRule('shared_arch','shared_'+self.arch.hostOsBase)
      self.addMakeMacro('BUILDSHAREDLIB','yes')
    else:
      self.addMakeRule('shared_arch','')
      self.addMakeMacro('BUILDSHAREDLIB','no')
    if self.setCompilers.sharedLibraries:
      self.addDefine('HAVE_SHARED_LIBRARIES', 1)
    if self.useShared:
      self.addDefine('USE_SHARED_LIBRARIES', 1)
    else:
      self.logPrint('Shared libraries - disabled')
    return

  def configureDynamicLibraries(self):
    '''Checks whether dynamic libraries should be used, for which you must
      - Specify --with-dynamic
      - Have found a working dynamic linker (with dlfcn.h and libdl)
    Defines PETSC_USE_DYNAMIC_LIBRARIES if they are used'''
    if self.setCompilers.dynamicLibraries:
      self.addDefine('HAVE_DYNAMIC_LIBRARIES', 1)
    self.useDynamic = self.argDB['with-dynamic'] and self.useShared and self.setCompilers.dynamicLibraries
    if self.useDynamic:
      self.addDefine('USE_DYNAMIC_LIBRARIES', 1)
    else:
      self.logPrint('Dynamic libraries - disabled')
    return

  def configure(self):
    self.executeTest(self.configureSharedLibraries)
    self.executeTest(self.configureDynamicLibraries)
    return
