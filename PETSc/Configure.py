import configure

import os
import re

class Configure(configure.Configure):
  def __init__(self, framework):
    configure.Configure.__init__(self)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.framework    = framework
    self.hostMacro    = 'dnl Version: 2.13\ndnl Variable: host_cpu\ndnl Variable: host_vendor\ndnl Variable: host_os\nAC_CANONICAL_HOST'
    return

  def configureDirectories(self):
    if os.environ.has_key('PETSC_DIR'):
      self.dir = os.environ['PETSC_DIR']
    else:
      self.dir = os.getcwd()
    self.addSubstitution('DIR', self.dir)
    self.addDefine('DIR', self.dir, 'The root directory of the Petsc installation')
    return

  def configureArchitecture(self):
    results = self.executeShellCode(self.macroToShell(self.hostMacro))
    if not os.environ.has_key('PETSC_ARCH'):
      self.arch = results['host_os']
    else:
      self.arch = os.environ['PETSC_ARCH']
    if not self.arch.startswith(results['host_os']):
      raise RuntimeError('PETSC_ARCH ('+self.arch+') does not have our guess ('+results['host_os']+') as a prefix!')
    self.addSubstitution('ARCH', self.arch)
    self.addDefine('ARCH', re.sub(r'^(\w+)[-_]?.*$', r'\1', self.arch), 'The primary architecture of this machine')
    return

  def configureLibraryOptions(self):
    self.bopt     = 'g'
    self.useLog   = 1
    self.useStack = 1
    return

  def configure(self):
    self.configureDirectories()
    self.configureArchitecture()
    self.configureLibraryOptions()
    return
