#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.functions     = ['PAPI_library_init']
    self.includes      = ['papi.h']
    self.liblist       = [['libpapi.a','libperfctr.a']]

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.languages  = framework.require('PETSc.utilities.languages',self)
    return


if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
