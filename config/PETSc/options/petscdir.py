import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.isPetsc      = 1
    return

  def __str1__(self):
    if hasattr(self, 'dir'):
      return '  PETSC_DIR: '+str(self.dir)+'\n'
    return ''

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-PETSC_DIR=<root-dir>',                        nargs.Arg(None, None, 'The root directory of the PETSc installation'))
    return

  def configureDirectories(self):
    '''Checks PETSC_DIR and sets if not set'''
    if 'PETSC_DIR' in self.framework.argDB:
      self.dir = os.path.normpath(self.framework.argDB['PETSC_DIR'])
      if self.dir == 'pwd':
        raise RuntimeError('You have set -PETSC_DIR=pwd, you need to use back quotes around the pwd\n  like -PETSC_DIR=`pwd`')
      if not os.path.isdir(self.dir):
        raise RuntimeError('The value you set with -PETSC_DIR='+self.dir+' is not a directory')
    elif 'PETSC_DIR' in os.environ:
      self.dir = os.path.normpath(os.environ['PETSC_DIR'])
      if self.dir == 'pwd':
        raise RuntimeError('''
The environmental variable PETSC_DIR is set incorrectly. Please use the following: [notice backquotes]
  For sh/bash  : PETSC_DIR=`pwd`; export PETSC_DIR
  for csh/tcsh : setenv PETSC_DIR `pwd`''')
      elif not os.path.isdir(self.dir):
        raise RuntimeError('The environmental variable PETSC_DIR '+self.dir+' is not a directory')
    else:
      self.dir = os.getcwd()
    if self.isPetsc and not os.path.realpath(self.dir) == os.path.realpath(os.getcwd()):
      raise RuntimeError('The environmental variable PETSC_DIR '+self.dir+' MUST be the current directory '+os.getcwd())
    self.version  = 'Unknown'
    versionHeader = os.path.join(self.dir, 'include', 'petscversion.h')
    versionInfo = []
    if os.path.exists(versionHeader):
      f = open(versionHeader)
      for line in f:
        if line.find('define PETSC_VERSION') >= 0:
          versionInfo.append(line[:-1])
      f.close()
    else:
      raise RuntimeError('Invalid PETSc directory '+str(self.dir)+'. Could not locate '+versionHeader)
    self.versionRelease = True if versionInfo[0].split(' ')[-1] == '1' else False
    if self.versionRelease:
      self.version = '.'.join([line.split(' ')[-1] for line in versionInfo[1:4]])
    else:
      self.version = '.'.join([line.split(' ')[-1] for line in versionInfo[1:3]])
      self.version += '.99'
    self.logPrint('Version Information:')
    for line in versionInfo:
      self.logPrint(line)
    self.framework.argDB['with-executables-search-path'].append(os.path.join(self.dir, 'lib','petsc','bin', 'win32fe'))

    return

  def configure(self):
    self.executeTest(self.configureDirectories)
    return
