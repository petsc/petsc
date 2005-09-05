#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.functions     = ['impl_sidl_DLL__ctor']
    self.includes      = ['sidl.h']
    self.liblist       = [['libsidl.a']]
    self.version       = '0.10.8'


  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.shared     = framework.require('PETSc.utilities.sharedLibraries',self)
    self.dynamic    = framework.require('PETSc.utilities.dynamicLibraries',self)
    self.languages  = framework.require('PETSc.utilities.languages',self)
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by babel'''
    '''Normally you do not need to provide this method'''
    if not self.languages.clanguage == 'Cxx':
      raise RuntimeError('babel requires --with-clanguage=c++')
    if not self.shared.useShared:
      raise RuntimeError('babel requires --with-shared=1')
    PETSc.package.Package.configureLibrary(self)
    if self.framework.argDB['with-babel-dir']:
      self.getExecutable('babel', path = os.path.join(self.framework.argDB['with-babel-dir'],'bin'), getFullPath = 1,resultName = 'babel')
    else:
      self.getExecutable('babel', getFullPath = 1,resultName = 'babel')
    if not hasattr(self,'babel'):
      raise RuntimeError('Located Babel library and include file but could not find babel executable')
    self.addMakeMacro('BABEL_VERSION',self.version)
    return


if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
