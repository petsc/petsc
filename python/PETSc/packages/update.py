from __future__ import generators
import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.updated      = 0
    self.strmsg       = ''
    self.hasdatafiles = 0
    self.arch         = self.framework.require('PETSc.packages.arch', self)
    return

  def __str__(self):
    return self.strmsg
     
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-default-arch=<bool>',                nargs.ArgBool(None, 1, 'Allow using the last configured arch without setting PETSC_ARCH'))
    help.addArgument('PETSc', '-with-default-language=<c,c++,complex,0>', nargs.Arg(None, 'c', 'Specifiy default language of libraries. 0 indicates no default'))
    help.addArgument('PETSc', '-with-default-optimization=<g,O,0>',       nargs.Arg(None, 'g', 'Specifiy default optimization of libraries. 0 indicates no default'))
    help.addArgument('PETSc', '-DATAFILESPATH=directory',                 nargs.Arg(None, None, 'Specifiy location of PETSc datafiles, e.g. test matrices'))    
    return

  def configureDirectories(self):
    '''Verifies that PETSC_DIR is acceptable'''
    if not os.path.samefile(self.arch.dir, os.getcwd()):
      raise RuntimeError('  Wrong PETSC_DIR option specified: '+ self.framework.argDB['PETSC_DIR'] + '\n  Configure invoked in: '+ os.path.realpath(os.getcwd()))
    if not os.path.exists(os.path.join(self.arch.dir, 'include', 'petscversion.h')):
      raise RuntimeError('Invalid PETSc directory '+str(self.arch.dir)+' it may not exist?')
    return

  def configureArchitecture(self):
    '''Setup a default architecture; so one need not set PETSC_ARCH'''
    if self.framework.argDB['with-default-arch']:
      fd = file(os.path.join('bmake', 'petscconf'), 'w')
      fd.write('PETSC_ARCH='+self.arch.arch+'\n')
      fd.write('include '+os.path.join('${PETSC_DIR}','bmake',self.arch.arch,'petscconf')+'\n')
      fd.close()
      self.framework.actions.addArgument('PETSc', 'Build', 'Set default architecture to '+self.arch.arch+' in bmake/petscconf')
    else:
      os.unlink(os.path.join('bmake', 'petscconf'))
    return

  def datafilespath(self):
    '''Checks what DATAFILESPATH should be'''
    datafilespath = None
    if self.framework.argDB.has_key('DATAFILESPATH'):
      if os.path.isdir(self.framework.argDB['DATAFILESPATH']) & os.path.isdir(os.path.join(self.framework.argDB['DATAFILESPATH'], 'matrices')):
        datafilespath = self.framework.argDB['DATAFILESPATH']
      else:
        raise RuntimeError('Path given with option -DATAFILES='+self.framework.argDB['DATAFILESPATH']+' is not a valid datafiles directory')
    elif os.path.isdir(os.path.join('/home','petsc','datafiles')) & os.path.isdir(os.path.join('/home','petsc','datafiles','matrices')):
      datafilespath = os.path.join('/home','petsc','datafiles')
    elif os.path.isdir(os.path.join(self.arch.dir, '..', 'datafiles')) &  os.path.isdir(os.path.join(self.arch.dir, '..', 'datafiles', 'matrices')):
      datafilespath = os.path.join(self.arch.dir, '..', 'datafiles')
    if datafilespath:
      self.framework.addSubstitution('SET_DATAFILESPATH', 'DATAFILESPATH ='+datafilespath)
      self.hasdatafiles = 1
    else:
      self.framework.addSubstitution('SET_DATAFILESPATH', '')
    return

  def configure(self):
    self.executeTest(self.configureDirectories)
    self.executeTest(self.configureArchitecture)
    self.executeTest(self.datafilespath)
    return
