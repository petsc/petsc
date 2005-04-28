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
    help.addArgument('PETSc', '-with-dynamic=<bool>', nargs.ArgBool(None, 0, 'Build dynamic libraries for PETSc'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers = framework.require('config.compilers', self)
    self.headers   = framework.require('config.headers', self)
    self.libraries = framework.require('config.libraries', self)
    self.arch      = framework.require('PETSc.utilities.arch', self)
    self.shared    = framework.require('PETSc.utilities.sharedLibraries', self)
    return

  def configureDynamicLibraries(self):
    '''Checks whether dynamic libraries should be used, for which you must
      - Specify --with-dynamic
      - Find dlfcn.h and libdl
    Defines PETSC_USE_DYNAMIC_LIBRARIES if they are used
    Also checks that dlopen() takes RTLD_GLOBAL, and defines PETSC_HAVE_RTLD_GLOBAL if it does'''
    self.useDynamic = 0
    self.useDynamic = self.shared.useShared and self.framework.argDB['with-dynamic'] and self.headers.check('dlfcn.h')
    if not self.useDynamic:
      self.logPrint('Dynamic libraries - disabled')
      return
    if not self.libraries.add('dl', ['dlopen', 'dlsym']):
      if not self.libraries.check('', ['dlopen', 'dlsym']):
        self.logPrint('The dynamic linking functions dlopen() and dlsym() were not found')
        self.useDynamic = 0
    if self.useDynamic:
      self.addDefine('USE_DYNAMIC_LIBRARIES', 1)
      if self.checkLink('#include <dlfcn.h>\nchar *libname;\n', 'dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);\n'):
        self.addDefine('HAVE_RTLD_GLOBAL', 1)
    return

  def configure(self):
    self.executeTest(self.configureDynamicLibraries)
    return
