#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.version  = '0.5.9'
    #self.liblist  = [['libccaffeine_'+self.version.replace('.','_')+'.a']]
    self.includes = ['ccafe-'+self.version+'/cmd/Cmd.h']
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.babel      = framework.require('PETSc.packages.babel',self)
    return

  def configureLibrary(self):
    '''Calls the regular package configureLibrary and then does an additional test needed by babel'''
    '''Normally you do not need to provide this method'''
    PETSc.package.Package.configureLibrary(self)
    self.addMakeMacro('CCAFE_HOME',self.framework.argDB['with-ccafe-dir'])
    self.addMakeMacro('CCAFE_VERSION',self.version)
    self.addMakeMacro('CCA_REPO','${CCAFE_HOME}/share/cca-spec-babel-0_7_8-babel-${BABEL_VERSION}/xml')
    self.addMakeMacro('HAVE_CCA','-DHAVE_CCA')

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
