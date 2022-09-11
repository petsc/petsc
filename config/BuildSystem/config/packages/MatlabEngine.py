from __future__ import generators
import config.package

import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.PACKAGE      = 'MATLAB-ENGINE'
    self.package      = 'matlab-engine'
    self.precisions   = ['double']
    self.hastests     = 1
    return

  def configureLibrary(self):
    ''' Deprecated, now handled by Matlab.py since only standard library names that will always be available are needed'''
    pass

