#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download          = ['ssh://petsc@petsc.cs.iit.edu/petsc4py/petsc4py-dev']
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
    pp = os.path.join(self.installDir,'lib','python*','site-packages')
    if self.setCompilers.isDarwin():
      apple = 'You may first need to (csh/tcsh) setenv MACOSX_DEPLOYMENT_TARGET 10.X\n (sh/bash) set  MACOSX_DEPLOYMENT_TARGET=10.X;export MACOSX_DEPLOYMENT_TARGET\n'
    else:
      apple = ''
    self.logClearRemoveDirectory()
    self.logPrintBox('After installing PETSc run:\ncd '+self.packageDir+'\n python setup.py install --prefix='+self.installDir+'\n'+apple+'then add the following to your shell startup file (.cshrc, .bashrc etc)\n (csh/tcsh) setenv PYTHONPATH ${PYTHONPATH}:'+pp+'\n (sh/bash) set PYTHONPATH=${PYTHONPATH}:'+pp+';export PYTHONPATH' )
    self.logResetRemoveDirectory()
    return self.installDir

  def configureLibrary(self):
    if self.setCompilers.isDarwin():
      # The name of the Python library on Apple is Python which does not end in the expected .dylib
      # Thus see if the python library in the standard location points to the Python version
      import sys
      import os
      prefix = sys.exec_prefix
      if os.path.isfile(os.path.join(prefix,'Python')):
        if os.path.realpath('/usr/lib/libpython.dylib') == os.path.join(prefix,'Python'):
          self.addDefine('PYTHON_LIB','"'+os.path.join('/usr','lib','libpython.dylib')+'"')
          return
        raise RuntimeError('realpath of /usr/lib/libpython.dylib ('+os.path.realpath('/usr/lib/libpython.dylib')+') does not point to Python library path ('+os.path.join(prefix,'Python')+') for current Python;\n Are you not using the Apple python?')
      elif os.path.isfile(os.path.join(prefix,'lib','libpython.dylib')):
        self.addDefine('PYTHON_LIB','"'+os.path.join(prefix,'lib','libpython.dylib')+'"')
      else:
        raise RuntimeError('Unable to find Python dynamic library at prefix '+prefix)

