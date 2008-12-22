#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download          = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpi4py-dev.tar.gz']
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
    self.setCompilers.pushLanguage('C')    
    dd = self.setCompilers.getCompiler()
    self.setCompilers.popLanguage()    
    if self.setCompilers.isDarwin():
      apple = 'You may first need to\n (csh/tcsh) setenv MACOSX_DEPLOYMENT_TARGET 10.X\n (sh/bash) MACOSX_DEPLOYMENT_TARGET=10.X; export MACOSX_DEPLOYMENT_TARGET\n'
    else:
      apple = ''
    self.logClearRemoveDirectory()
    self.logPrintBox('After installing PETSc run:\n (csh/tcsh) setenv MPICC '+dd+'\n (sh/bash) MPICC='+dd+'; export MPICC \ncd '+self.packageDir+'\n python setup.py install --prefix='+self.installDir+'\n'+apple+'then add the following to your shell startup file (.cshrc, .bashrc etc)\n (csh/tcsh) setenv PYTHONPATH ${PYTHONPATH}:'+pp+'\n (sh/bash) set PYTHONPATH=${PYTHONPATH}:'+pp+'; export PYTHONPATH' )
    self.logResetRemoveDirectory()

