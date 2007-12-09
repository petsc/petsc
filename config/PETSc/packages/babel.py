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
    self.version       = '0.10.12'

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.shared     = framework.require('PETSc.utilities.sharedLibraries',self)
    self.languages  = framework.require('PETSc.utilities.languages',self)
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by babel'''
    '''Normally you do not need to provide this method'''
    PETSc.package.Package.configureLibrary(self)
    # add in include/cxx path
    self.include.append(os.path.join(self.framework.argDB['with-babel-dir'],'include','cxx'))
    babel_bin_path = os.path.join(self.framework.argDB['with-babel-dir'],'bin')
    if self.framework.argDB['with-babel-dir']:
      self.getExecutable('babel', path = babel_bin_path, getFullPath = 1,resultName = 'babel')
    else:
      self.getExecutable('babel', getFullPath = 1,resultName = 'babel')
    if not hasattr(self,'babel'):
      raise RuntimeError('Located Babel library and include file but could not find babel executable')
    self.getExecutable('babel-config', path = babel_bin_path, getFullPath = 1, resultName = 'babel_config')
    if not hasattr(self,'babel_config'):
      raise RuntimeError('Located Babel library and include file but could not find babel-config executable')
    self.version = config.base.Configure.executeShellCommand(self.babel_config + ' --version')[0].rstrip()
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
