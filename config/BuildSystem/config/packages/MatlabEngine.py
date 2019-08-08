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
    if 'with-matlabengine-lib' in self.framework.clArgDB:
      self.lib = self.argDB['with-matlabengine-lib']
      self.framework.packages.append(self)
      self.found = 1
      return

    if self.matlab.matlab_arch == 'mac':
      matlab_dl = [' -L'+os.path.join(self.matlab.matlab,'sys','os','mac'),' -ldl']
    else:
      matlab_dl = ['']
    # Matlab libraries require libstdc++-libc6.1-2.so.3 which they provide in the sys/os directory
    if self.matlab.matlab_arch == 'glnx86' or self.matlab.matlab_arch == 'glnxa64':
       matlab_sys = self.setCompilers.CSharedLinkerFlag+os.path.join(self.matlab.matlab,'sys','os',self.matlab.matlab_arch)
       matlab_sys += ':'+os.path.join(self.matlab.matlab,'bin',self.matlab.matlab_arch)+':'+os.path.join(self.matlab.matlab,'extern','lib',self.matlab.matlab_arch)
    else:
       matlab_sys = ''
    self.lib = [matlab_sys,'-L'+os.path.join(self.matlab.matlab,'bin',self.matlab.matlab_arch),'-L'+os.path.join(self.matlab.matlab,'extern','lib',self.matlab.matlab_arch),'-leng','-lmex','-lmx','-lmat','-lut','-licudata','-licui18n','-licuuc'] + matlab_dl
    if self.setCompilers.isDarwin(self.log):
      self.logPrintBox('You may need to set DYLD_FALLBACK_LIBRARY_PATH to \n'+os.path.join(self.matlab.matlab,'bin',self.matlab.matlab_arch)+':'+os.path.join(self.matlab.matlab,'sys','os',self.matlab.matlab_arch))
    self.framework.packages.append(self)
    self.found = 1
    return

