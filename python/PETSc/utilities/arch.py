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
    help.addArgument('PETSc', '-with-default-arch=<bool>',        nargs.ArgBool(None, 1, 'Allow using the last configured arch without setting PETSC_ARCH'))
    help.addArgument('PETSc', '-with-external-packages-dir=<dir>',nargs.Arg(None, os.path.join('$PETSC_DIR','externalpackages'),'Location to installed downloaded packages'))

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
    if not os.path.samefile(self.dir, os.getcwd()):
      raise RuntimeError('Wrong PETSC_DIR option specified: '+str(self.dir) + '\n  Configure invoked in: '+os.path.realpath(os.getcwd()))
    if not os.path.exists(os.path.join(self.dir, 'include', 'petscversion.h')):
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
      raise RuntimeError('Unable to determine host type using '+configSub+': '+str(e))
    m = re.match(r'^(?P<cpu>[^-]*)-(?P<vendor>[^-]*)-(?P<os>.*)$', output)
    if not m:
      raise RuntimeError('Unable to parse output of '+configSub+': '+output)
    self.framework.host_cpu = m.group('cpu')
    self.host_vendor        = m.group('vendor')
    self.host_os            = m.group('os')

    if 'PETSC_ARCH' in self.framework.argDB:
      self.arch = self.framework.argDB['PETSC_ARCH']
    else:
      if 'PETSC_ARCH' in os.environ:
        self.arch = os.environ['PETSC_ARCH']
      else:
        self.arch = self.host_os
    self.archBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.arch)
    self.addDefine('ARCH', self.archBase)
    self.addDefine('ARCH_NAME', '"'+self.arch+'"')
    self.framework.argDB['PETSC_ARCH']      = self.arch
    self.framework.argDB['PETSC_ARCH_BASE'] = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.host_os)
    self.addArgumentSubstitution('ARCH', 'PETSC_ARCH')
    if self.framework.argDB['with-default-arch']:
      fd = file(os.path.join('bmake', 'petscconf'), 'w')
      fd.write('PETSC_ARCH='+self.arch+'\n')
      fd.write('include '+os.path.join('${PETSC_DIR}','bmake',self.arch,'petscconf')+'\n')
      fd.close()
      self.framework.actions.addArgument('PETSc', 'Build', 'Set default architecture to '+self.arch+' in bmake/petscconf')
    else:
      os.unlink(os.path.join('bmake', 'petscconf'))
    return


  def configure(self):
    self.executeTest(self.configureDirectories)
    self.executeTest(self.configureArchitecture)
    if self.framework.argDB['with-external-packages-dir'].startswith('$'):
      self.framework.argDB['with-external-packages-dir'] = os.path.join(self.framework.argDB['PETSC_DIR'], 'externalpackages')
    return
