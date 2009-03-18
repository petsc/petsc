#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download          = ['http://ftp.mcs.anl.gov/pub/petsc/externalpackages/mpi4py-dev.tar.gz']
    self.functions         = []
    self.includes          = []
    self.liblist           = []
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.numpy      = framework.require('PETSc.packages.Numpy',self)
    self.petscdir   = framework.require('PETSc.utilities.petscdir',self)
    self.setCompilers  = framework.require('config.setCompilers',self)
    self.sharedLibraries = framework.require('PETSc.utilities.sharedLibraries', self)    
    self.petscconfigure   = framework.require('PETSc.Configure',self)
    self.arch = framework.require('PETSc.utilities.arch', self)
    return

  def Install(self):
    pp = os.path.join(self.installDir,'lib','python*','site-packages')
    if self.setCompilers.isDarwin():
      apple = 'You may need to\n (csh/tcsh) setenv MACOSX_DEPLOYMENT_TARGET 10.X\n (sh/bash) MACOSX_DEPLOYMENT_TARGET=10.X; export MACOSX_DEPLOYMENT_TARGET\nbefore running make on PETSc'
    else:
      apple = ''
    self.logClearRemoveDirectory()
    self.logResetRemoveDirectory()
    if self.framework.argDB['prefix']:
      arch = ''
      self.addMakeRule('mpi4py_noinstall','')
    else:
      arch = self.arch.arch
      self.addMakeRule('mpi4py_noinstall','mpi4py')      
    self.addMakeRule('mpi4py','', \
                       ['@MPICC=${PCC}; export MPICC; cd '+self.packageDir+';python setup.py clean --all; python setup.py install --install-lib='+os.path.join(self.petscconfigure.installdir,'lib'),\
                          '@echo "====================================="',\
                          '@echo "To use mpi4py, add '+os.path.join(self.petscconfigure.installdir,'lib')+' to PYTHONPATH"',\
                          '@echo "====================================="'])
    
    return self.installDir

  def configureLibrary(self):
    self.checkDownload(1)
    if not self.sharedLibraries.useShared:
        raise RuntimeError('mpi4py requires PETSc be built with shared libraries; rerun with --with-shared')

  def alternateConfigureLibrary(self):
    self.addMakeRule('mpi4py','')   
    self.addMakeRule('mpi4py_noinstall','')
