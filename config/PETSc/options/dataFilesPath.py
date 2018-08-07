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
    help.addArgument('PETSc', '-DATAFILESPATH=<dir>',                 nargs.Arg(None, None, 'Specifiy location of PETSc datafiles, e.g. test matrices'))
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.petscdir = framework.require('PETSc.options.petscdir', self)
    return

  def getDatafilespath(self):
    '''Checks what DATAFILESPATH should be'''
    homeloc = os.path.join(os.getenv('HOME', '.'), 'datafiles')
    parentloc =  os.path.join(self.petscdir.dir,'..','datafiles')
    self.datafilespath = None

    if 'DATAFILESPATH' in self.framework.argDB:
      if os.path.isdir(self.framework.argDB['DATAFILESPATH']) and os.path.isdir(os.path.join(self.framework.argDB['DATAFILESPATH'], 'matrices')):
        self.datafilespath = str(self.framework.argDB['DATAFILESPATH'])
      else:
        raise RuntimeError('Path given with option -DATAFILES='+self.framework.argDB['DATAFILESPATH']+' is not a valid datafiles directory')
    elif os.path.isdir(homeloc) and os.path.isdir(os.path.join(homeloc,'matrices')):
      self.datafilespath = homeloc
    elif os.path.isdir(parentloc) and  os.path.isdir(os.path.join(parentloc,'matrices')):
      self.datafilespath = parentloc
    elif os.path.isdir(os.path.join(self.petscdir.dir, '..', '..','Datafiles')) &  os.path.isdir(os.path.join(self.petscdir.dir, '..','..', 'Datafiles', 'Matrices')):
      self.datafilespath = os.path.join(self.petscdir.dir, '..','..', 'Datafiles')
    if self.datafilespath:
      self.addMakeMacro('DATAFILESPATH',self.datafilespath)
    return

  def configure(self):
    self.executeTest(self.getDatafilespath)
    return
