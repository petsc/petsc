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
    return ''

  def configureDirectories(self):
    '''Checks PETSC_DIR and sets if not set'''
    if 'PETSC_DIR' in self.framework.argDB:
      self.dir = self.framework.argDB['PETSC_DIR']
    else:
      if 'PETSC_DIR' in os.environ:
        self.dir = os.environ['PETSC_DIR']
      else:
        self.dir = os.getcwd()
    if self.dir[1] == ':':
      try:
        dir = self.dir.replace('\\','/')
        (dir, error, status) = self.executeShellCommand('cygpath -au '+dir)
        self.dir = dir.replace('\n','')
      except RuntimeError:
        pass
    self.addSubstitution('DIR', self.dir)
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
    return

  def configure(self):
    self.executeTest(self.configureDirectories)
    self.executeTest(self.configureArchitecture)
    return
