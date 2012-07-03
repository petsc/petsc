from __future__ import generators
import config.base
import config.package
import os
import re
import sys
try:
  from hashlib import md5 as new_md5
except ImportError:
  from md5 import new as new_md5

import nargs

class NewPackage(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    # These are specified for the package
    self.double           = 1   # 1 means requires double precision
    self.complex          = 0   # 0 means cannot use complex
    self.requires32bitint = 1;  # 1 means that the package will not work with 64 bit integers
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.petscarch      = framework.require('PETSc.utilities.arch', self)
    self.languages      = framework.require('PETSc.utilities.languages', self)
    self.scalartypes    = framework.require('PETSc.utilities.scalarTypes',self)
    self.libraryOptions = framework.require('PETSc.utilities.libraryOptions', self)
    self.petscdir       = framework.require('PETSc.utilities.petscdir', self.setCompilers)
    return

  def consistencyChecks(self):
    config.package.Package.consistencyChecks(self)
    if self.framework.argDB['with-'+self.package]:
      if self.double and not self.scalartypes.precision.lower() == 'double':
        raise RuntimeError('Cannot use '+self.name+' withOUT double precision numbers, it is not coded for this capability')
      if not self.complex and self.scalartypes.scalartype.lower() == 'complex':
        raise RuntimeError('Cannot use '+self.name+' with complex numbers it is not coded for this capability')
      if self.libraryOptions.integerSize == 64 and self.requires32bitint:
        raise RuntimeError('Cannot use '+self.name+' with 64 bit integers, it is not coded for this capability')
    return
