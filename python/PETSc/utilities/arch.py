import config.base
import os
import re

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    return

  def __str__(self):
    desc = ['PETSc:']
    desc.append('  PETSC_ARCH: '+str(self.framework.argDB['PETSC_ARCH']))
    desc.append('  PETSC_DIR: '+str(self.framework.argDB['PETSC_DIR']))
    return '\n'.join(desc)+'\n'

  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', 'PETSC_DIR',                        nargs.Arg(None, None, 'The root directory of the PETSc installation'))
    help.addArgument('PETSc', 'PETSC_ARCH',                       nargs.Arg(None, None, 'The machine architecture'))
    help.addArgument('PETSc', '-with-external-packages-dir=<dir>',nargs.Arg(None, os.path.join('$PETSC_DIR','externalpackages'),'Location to installed downloaded packages'))
    return

  def configureDirectories(self):
    '''Checks PETSC_DIR and sets if not set'''
    if 'PETSC_DIR' in self.framework.argDB:
      self.dir = self.framework.argDB['PETSC_DIR']
      if self.dir == 'pwd':
        raise RuntimeError('You have set -PETSC_DIR=pwd, you need to use back quotes around the pwd\n  like -PETSC_DIR=`pwd`')
      if not os.path.isdir(self.dir):
        raise RuntimeError('The value you set with -PETSC_DIR='+self.dir+' is not a directory')
    else:
      if 'PETSC_DIR' in os.environ:
        self.dir = os.environ['PETSC_DIR']
        if not os.path.isdir(self.dir):
          raise RuntimeError('The environmental variable PETSC_DIR '+self.dir+' is not a directory')
      else:
        self.dir = os.getcwd()
    if self.dir[1] == ':':
      try:
        (self.dir, error, status) = self.executeShellCommand('cygpath -au '+self.dir)
      except RuntimeError:
        pass
    if os.path.exists(os.path.join(self.dir, 'include', 'petscversion.h')):
      try:
        (version_info,error,status) = self.executeShellCommand('grep "define PETSC_VERSION" '+ os.path.join(self.dir, 'include', 'petscversion.h'))
      except RuntimeError,e:
        raise RuntimeError('Error running grep on petscversion.h: '+str(e))
    else:
      raise RuntimeError('Invalid PETSc directory '+str(self.dir)+' it may not exist?')

    self.addMakeMacro('DIR', self.dir)
    self.addDefine('DIR', self.dir)
    self.framework.argDB['PETSC_DIR'] = self.dir
    return

  def configureArchitecture(self):
    '''Checks PETSC_ARCH and sets if not set'''
    import sys

    auxDir = None
    for dir in [os.path.join(self.dir, 'config'), os.path.join(self.dir, 'bin', 'config')] + sys.path:
      if os.path.isfile(os.path.join(dir, 'config.sub')):
        auxDir      = dir
        configSub   = os.path.join(auxDir, 'config.sub')
        configGuess = os.path.join(auxDir, 'config.guess')
        break
    if auxDir is None:
      raise RuntimeError('Unable to locate config.sub in order to determine architecture.Your PETSc directory is incomplete.\n Get PETSc again')
    try:
      host   = config.base.Configure.executeShellCommand(self.shell+' '+configGuess, log = self.framework.log)[0]
      output = config.base.Configure.executeShellCommand(self.shell+' '+configSub+' '+host, log = self.framework.log)[0]
    except RuntimeError, e:
      fd = open(configGuess)
      data = fd.read()
      fd.close()
      if data.find('\r\n') >= 0:
        raise RuntimeError('''It appears petsc.tar.gz is uncompressed on Windows (perhaps with Winzip)
          and files copied over to Unix/Linux. Windows introduces LF characters which are
          inappropriate on other systems. Please use gunzip/tar on the install machine.\n''')
      raise RuntimeError('Unable to determine host type using '+configSub+': '+str(e))
    m = re.match(r'^(?P<cpu>[^-]*)-(?P<vendor>[^-]*)-(?P<os>.*)$', output)
    if not m:
      raise RuntimeError('Unable to parse output of '+configSub+': '+output)
    self.framework.host_cpu    = m.group('cpu')
    self.framework.host_vendor = m.group('vendor')
    self.framework.host_os     = m.group('os')

    if 'PETSC_ARCH' in self.framework.argDB:
      self.arch = self.framework.argDB['PETSC_ARCH']
    else:
      if 'PETSC_ARCH' in os.environ:
        self.arch = os.environ['PETSC_ARCH']
      else:
        self.arch = self.framework.host_os
    self.archBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.arch)
    self.hostOsBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.framework.host_os)
    self.addDefine('ARCH', self.hostOsBase)
    self.addDefine('ARCH_NAME', '"'+self.arch+'"')
    self.framework.argDB['PETSC_ARCH']      = self.arch
    self.framework.argDB['PETSC_ARCH_BASE'] = self.hostOsBase
    self.addArgumentSubstitution('ARCH', 'PETSC_ARCH')
    return


  def configure(self):
    self.executeTest(self.configureDirectories)
    self.executeTest(self.configureArchitecture)
    if self.framework.argDB['with-external-packages-dir'].startswith('$'):
      self.framework.argDB['with-external-packages-dir'] = os.path.join(self.framework.argDB['PETSC_DIR'], 'externalpackages')
    return
