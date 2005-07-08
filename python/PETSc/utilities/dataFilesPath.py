from __future__ import generators
import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix  = 'PETSC'
    self.substPrefix   = 'PETSC'
    self.updated       = 0
    self.strmsg        = ''
    self.datafilespath = ''
    return

  def __str__(self):
    return self.strmsg
     
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-DATAFILESPATH=directory',                 nargs.Arg(None, None, 'Specifiy location of PETSc datafiles, e.g. test matrices'))    
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.petscdir = framework.require('PETSc.utilities.petscdir', self)
    return

  def getDatafilespath(self):
    '''Checks what DATAFILESPATH should be'''
    self.datafilespath = None
    if self.framework.argDB.has_key('DATAFILESPATH'):
      if os.path.isdir(self.framework.argDB['DATAFILESPATH']) & os.path.isdir(os.path.join(self.framework.argDB['DATAFILESPATH'], 'matrices')):
        self.datafilespath = str(self.framework.argDB['DATAFILESPATH'])
      else:
        raise RuntimeError('Path given with option -DATAFILES='+self.framework.argDB['DATAFILESPATH']+' is not a valid datafiles directory')
    elif os.path.isdir(os.path.join('/home','petsc','datafiles')) & os.path.isdir(os.path.join('/home','petsc','datafiles','matrices')):
      self.datafilespath = os.path.join('/home','petsc','datafiles')
    elif os.path.isdir(os.path.join(self.petscdir.dir, '..', 'datafiles')) &  os.path.isdir(os.path.join(self.petscdir.dir, '..', 'datafiles', 'matrices')):
      self.datafilespath = os.path.join(self.petscdir.dir, '..', 'datafiles')
    self.addMakeMacro('DATAFILESPATH',self.datafilespath)
    return

  def configure(self):
    self.executeTest(self.getDatafilespath)
    return
