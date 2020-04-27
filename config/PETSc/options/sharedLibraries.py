#!/usr/bin/env python
from __future__ import generators
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.useShared    = 0
    return

  def __str1__(self):
    if not hasattr(self, 'useShared'):
      return ''
    txt = ''
    if self.useShared:
      txt += '  shared libraries: enabled\n'
    else:
      txt += '  shared libraries: disabled\n'
    return txt

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-shared-libraries=<bool>', nargs.ArgBool(None, 1, 'Make PETSc libraries shared -- libpetsc.so (Unix/Linux) or libpetsc.dylib (Mac)'))
    help.addArgument('PETSc', '-with-serialize-functions=<bool>', nargs.ArgBool(None, 0, 'Allows function pointers to be serialized to binary files with string representations'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.arch         = framework.require('PETSc.options.arch', self)
    self.debuggers    = framework.require('config.utilities.debuggers', self)
    self.setCompilers = framework.require('config.setCompilers', self)
    return

  def checkSharedDynamicPicOptions(self):
    # uf user specified 'with-shared' or 'with-dynamic' - flag an error
    if 'with-shared' in self.framework.argDB:
      raise RuntimeError('Option "--with-shared" no longer exists. Use "--with-shared-libraries".')
    if 'with-dynamic' in self.framework.argDB or 'with-dynamic-loading' in self.framework.argDB:
      raise RuntimeError('Option "--with-dynamic" and "--with-dynamic-loading" no longer exist.')
    # if user specifies inconsistant 'with-dynamic-loading with-shared-libraries with-pic' options - flag error
    if self.framework.argDB['with-shared-libraries'] and not self.framework.argDB['with-pic'] and 'with-pic' in self.framework.clArgDB:
      raise RuntimeError('If you use --with-shared-libraries you cannot disable --with-pic')

    # default with-shared-libraries=1 => --with-pic=1
    # Note: there is code in setCompilers.py that uses this as default.
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
      #if config.setCompilers.Configure.isSolaris(self.log) and config.setCompilers.Configure.isGNU(self.framework.getCompiler(),self.log):
      #  self.addMakeRule('shared_arch','shared_'+self.arch.hostOsBase+'gnu')
      #elif '-qmkshrobj' in self.setCompilers.sharedLibraryFlags:
      #  self.addMakeRule('shared_arch','shared_linux_ibm')
      #else:
      #  self.addMakeRule('shared_arch','shared_'+self.arch.hostOsBase)

      # Linux is the default
      if self.setCompilers.isDarwin(self.log):
        self.addMakeRule('shared_arch','shared_darwin')
        self.addMakeMacro('SONAME_FUNCTION', '$(1).$(2).dylib')
        self.addMakeMacro('SL_LINKER_FUNCTION', '-dynamiclib -install_name $(call SONAME_FUNCTION,$(1),$(2)) -compatibility_version $(2) -current_version $(3) -single_module -multiply_defined suppress -undefined dynamic_lookup')
      elif self.setCompilers.CC.find('win32fe') >=0:
        self.addMakeMacro('SONAME_FUNCTION', '$(1).dll')
        self.addMakeMacro('SL_LINKER_FUNCTION', '-LD')
        self.addMakeMacro('PETSC_DLL_EXPORTS', '1')
      else:
        # TODO: check that -Wl,-soname,${LIBNAME}.${SL_LINKER_SUFFIX} can be passed (might fail on Intel)
        # TODO: check whether to use -qmkshrobj or -shared (maybe we can just use self.setCompilers.sharedLibraryFlags)
        # TODO: check whether we need to specify dependent libraries on the link line (long test)
        self.addMakeRule('shared_arch','shared_linux')
        self.addMakeMacro('SONAME_FUNCTION', '$(1).so.$(2)')
        self.addMakeMacro('SL_LINKER_FUNCTION', '-shared -Wl,-soname,$(call SONAME_FUNCTION,$(notdir $(1)),$(2))')
        if config.setCompilers.Configure.isMINGW(self.framework.getCompiler(),self.log):
          self.addMakeMacro('PETSC_DLL_EXPORTS', '1')
      self.addMakeMacro('BUILDSHAREDLIB','yes')
    else:
      self.addMakeRule('shared_arch','')
      self.addMakeMacro('BUILDSHAREDLIB','no')
    if self.useShared:
      self.addDefine('USE_SHARED_LIBRARIES', 1)
    else:
      self.logPrint('Shared libraries - disabled')
    return

  def configureDynamicLibraries(self):
    '''Checks whether dynamic loading is available (with dlfcn.h and libdl)'''
    if self.setCompilers.dynamicLibraries:
      self.addDefine('HAVE_DYNAMIC_LIBRARIES', 1)
    return

  def configureSerializedFunctions(self):
    '''
    Defines PETSC_SERIALIZE_FUNCTIONS if they are used
    Requires shared libraries'''
    import sys

    if self.framework.argDB['with-serialize-functions'] and self.setCompilers.dynamicLibraries:
      self.addDefine('SERIALIZE_FUNCTIONS', 1)


  def configure(self):
    self.executeTest(self.checkSharedDynamicPicOptions)
    self.executeTest(self.configureSharedLibraries)
    self.executeTest(self.configureDynamicLibraries)
    self.executeTest(self.configureSerializedFunctions)
    return
