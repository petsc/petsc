#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download          = ['svn://petsc4py.googlecode.com/svn/trunk/']
    self.functions         = []
    self.includes          = []
    self.liblist           = []
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.numpy      = framework.require('PETSc.packages.Numpy',self)
    self.petscdir   = framework.require('PETSc.utilities.petscdir',self)
    self.setCompilers  = framework.require('config.setCompilers',self)
    return

  def Install(self):
    return self.installDir

  def configureLibrary(self):
    self.checkDownload(1)
    pp = os.path.join(self.installDir,'lib','python*','site-packages')
    if self.setCompilers.isDarwin():
      apple = 'You may first need to (csh/tcsh) setenv MACOSX_DEPLOYMENT_TARGET 10.X\n (sh/bash) set  MACOSX_DEPLOYMENT_TARGET=10.X;export MACOSX_DEPLOYMENT_TARGET\n'
    else:
      apple = ''
    self.logClearRemoveDirectory()
    self.logPrintBox('After installing PETSc run:\ncd '+os.path.join(self.petscdir.externalPackagesDir,'petsc4py')+'\n python setup.py install --prefix='+self.installDir+'\n'+apple+'then add the following to your shell startup file (.cshrc, .bashrc etc)\n (csh/tcsh) setenv PYTHONPATH ${PYTHONPATH}:'+pp+'\n (sh/bash) set PYTHONPATH=${PYTHONPATH}:'+pp+';export PYTHONPATH' )
    self.logResetRemoveDirectory()

