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
      txt += '  dynamic loading: enabled\n'
    else:
      txt += '  dynamic loading: disabled\n'
    return txt

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-shared-libraries=<bool>', nargs.ArgBool(None, 0, 'Make PETSc libraries shared -- libpetsc.so (Unix/Linux) or libpetsc.dylib (Mac)'))
    #help.addArgument('PETSc', '-with-shared=<bool>', nargs.ArgBool(None, 0, 'Make PETSc libraries shared -- libpetsc.so (Unix/Linux) or libpetsc.dylib (Mac)', deprecated = '-with-shared-libraries'))
    help.addArgument('PETSc', '-with-dynamic-loading=<bool>', nargs.ArgBool(None, 0, 'Make PETSc libraries dynamic -- uses dlopen() to access libraries, rarely needed'))
    #help.addArgument('PETSc', '-with-dynamic=<bool>', nargs.ArgBool(None, 0, 'Make PETSc libraries dynamic -- uses dlopen() to access libraries, rarely needed', deprecated = '-with-dynamic-loading'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.arch = framework.require('PETSc.utilities.arch', self)
    self.setCompilers = framework.require('config.setCompilers', self)
    return

  def checkSharedDynamicPicOptions(self):
    # if user specifies inconsistant 'with-dynamic-loading with-shared-libraries with-pic' options - flag error.
    if self.framework.argDB['with-dynamic-loading'] and not self.framework.argDB['with-shared-libraries'] and 'with-shared-libraries' in self.framework.clArgDB:
      raise RuntimeError('If you use --with-dynamic-loading you cannot disable --with-shared-libraries')
    if self.framework.argDB['with-dynamic-loading'] and not self.framework.argDB['with-pic'] and 'with-pic' in self.framework.clArgDB:
      raise RuntimeError('If you use --with-dynamic-loading you cannot disable --with-pic')
    if self.framework.argDB['with-shared-libraries'] and not self.framework.argDB['with-pic'] and 'with-pic' in self.framework.clArgDB:
      raise RuntimeError('If you use --with-shared-libraries you cannot disable --with-pic')

    # default with-dynamic-loading=1 => with-shared-libraries=1 --with-pic=1
    # default with-shared-libraries=1 => --with-pic=1
    # Note: there is code in setCompilers.py that uses this as default.
    if self.framework.argDB['with-dynamic-loading'] and not self.framework.argDB['with-shared-libraries']: self.framework.argDB['with-shared-libraries'] = 1
    if self.framework.argDB['with-shared-libraries'] and not self.framework.argDB['with-pic']: self.framework.argDB['with-pic'] = 1
    return

  def configureSharedLibraries(self):
    '''Checks whether shared libraries should be used, for which you must
      - Specify --with-shared-libraries
      - Have found a working shared linker
    Defines PETSC_USE_SHARED_LIBRARIES if they are used'''
    import sys

    self.useShared = self.framework.argDB['with-shared-libraries'] and not self.setCompilers.staticLibraries
    
    if self.useShared:
      if config.setCompilers.Configure.isSolaris() and config.setCompilers.Configure.isGNU(self.framework.getCompiler()):
        self.addMakeRule('shared_arch','shared_'+sys.platform+'gnu')
      elif '-qmkshrobj' in self.setCompilers.sharedLibraryFlags:
        self.addMakeRule('shared_arch','shared_linux_ibm')
      else:
        self.addMakeRule('shared_arch','shared_'+sys.platform)
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
    '''Checks whether dynamic loading should be used, for which you must
      - Specify --with-dynamic-loading
      - Have found a working dynamic linker (with dlfcn.h and libdl)
    Defines PETSC_USE_DYNAMIC_LIBRARIES if they are used'''
    if self.setCompilers.dynamicLibraries:
      self.addDefine('HAVE_DYNAMIC_LIBRARIES', 1)
    self.useDynamic = self.framework.argDB['with-dynamic-loading'] and self.useShared and self.setCompilers.dynamicLibraries
    if self.useDynamic:
      self.addDefine('USE_DYNAMIC_LIBRARIES', 1)
    else:
      self.logPrint('Dynamic loading - disabled')
    return

  def configure(self):
    self.executeTest(self.checkSharedDynamicPicOptions)
    self.executeTest(self.configureSharedLibraries)
    self.executeTest(self.configureDynamicLibraries)
    return
