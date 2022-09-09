from __future__ import generators
import config.package

import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.PACKAGE      = 'MATLAB_ENGINE'
    self.package      = 'matlab-engine'
    self.precisions   = ['double']
    self.hastests     = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.matlab = framework.require('config.packages.Matlab',self)
    self.deps       = [self.matlab]
    return

  def configureLibrary(self):
    '''Find a Matlab installation and check if it can work with PETSc'''
    if 'with-matlab-engine-lib' in self.framework.clArgDB:
      self.lib = self.argDB['with-matlab-engine-lib']
      self.framework.packages.append(self)
      self.found = 1
      return

    self.lib = [self.setCompilers.CSharedLinkerFlag+os.path.join(self.matlab.matlab,'bin',self.matlab.matlab_arch),'-L'+os.path.join(self.matlab.matlab,'bin',self.matlab.matlab_arch),'-leng','-lmex','-lmx','-lmat']
    self.framework.packages.append(self)
    self.found = 1
    return

