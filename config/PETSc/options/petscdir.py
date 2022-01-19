import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
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
      msg1 = 'The configure option'
      msg2 = ''
    elif 'PETSC_DIR' in os.environ:
      self.dir = os.path.normpath(os.environ['PETSC_DIR'])
      msg1 = 'The environmental variable'
      msg2 = 'export'
    else:
      self.dir = os.getcwd()
      msg1 = ''
      msg2 = ''

    if self.dir == 'pwd':
      raise RuntimeError('{0} PETSC_DIR=pwd is incorrect. You need to use back quotes around the pwd - i.e: {1} PETSC_DIR=`pwd`'.format(msg1, msg2))
    elif self.dir.find(' ') > -1:
      raise RuntimeError('{0} PETSC_DIR="{1}" has spaces in it; this is not allowed. Change the directory with PETSc to not have spaces in it'.format(msg1, self.dir))
    elif not os.path.isabs(self.dir):
      raise RuntimeError('{0} PETSC_DIR={1} is a relative path. Use absolute path - i.e: {2} PETSC_DIR={3}'.format(msg1, self.dir, msg2, os.path.abspath(self.dir)))
    elif not os.path.isdir(self.dir):
      raise RuntimeError('{0} PETSC_DIR={1} is not a directory'.format(msg1, self.dir))
    elif os.path.realpath(self.dir) != os.path.realpath(os.getcwd()):
      raise RuntimeError('{0} PETSC_DIR={1} MUST be the current directory {2}'.format(msg1, self.dir, os.getcwd()))

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
