import configure

import os
import re

class Configure(configure.Configure):
  def __init__(self, framework):
    configure.Configure.__init__(self, framework)
    self.headerPrefix = 'PETSC'
    self.substPrefix  = 'PETSC'
    self.defineAutoconfMacros()
    return

  def defineAutoconfMacros(self):
    self.hostMacro = 'dnl Version: 2.13\ndnl Variable: host_cpu\ndnl Variable: host_vendor\ndnl Variable: host_os\nAC_CANONICAL_HOST'
    return

  def configureDirectories(self):
    if self.framework.argDB.has_key('PETSC_DIR'):
      self.dir = self.framework.argDB['PETSC_DIR']
    else:
      self.dir = os.getcwd()
    self.addSubstitution('DIR', self.dir)
    self.addDefine('DIR', self.dir, 'The root directory of the Petsc installation')
    return

  def configureArchitecture(self):
    results = self.executeShellCode(self.macroToShell(self.hostMacro))
    if not self.framework.argDB.has_key('PETSC_ARCH'):
      self.arch = results['host_os']
    else:
      self.arch = self.framework.argDB['PETSC_ARCH']
    if not self.arch.startswith(results['host_os']):
      raise RuntimeError('PETSC_ARCH ('+self.arch+') does not have our guess ('+results['host_os']+') as a prefix!')
    self.addSubstitution('ARCH', self.arch)
    self.archBase = re.sub(r'^(\w+)[-_]?.*$', r'\1', self.arch)
    self.addDefine('ARCH', self.archBase, 'The primary architecture of this machine')
    return

  def configureLibraryOptions(self):
    self.getArgument('bopt', 'g', '-with-')
    self.getArgument('useDebug', 1, '-enable-', int, comment = 'Debugging flag')
    self.addDefine('USE_DEBUG', self.useDebug)
    self.getArgument('useLog',   1, '-enable-', int, comment = 'Logging flag')
    self.addDefine('USE_LOG',   self.useLog)
    self.getArgument('useStack', 1, '-enable-', int, comment = 'Stack tracing flag')
    self.addDefine('USE_STACK', self.useStack)
    return

  def configureCompilers(self):
    # Fortran compiler
    if self.framework.argDB.has_key('FC'):
      self.FC = self.framework.argDB['FC']
    else:
      self.getArgument('FC', 'g77', '-with-')
    self.addSubstitution('FC', self.FC, comment = 'Fortran compiler')
    # Fortran 90 compiler
    self.getArgument('F90Header', 'g77', '-with-', comment = 'C header for F90 interface')
    self.getArgument('F90Source', 'g77', '-with-', comment = 'C source for F90 interface')
    # C compiler
    if self.framework.argDB.has_key('CC'):
      self.CC = self.framework.argDB['CC']
    else:
      self.getArgument('CC', 'gcc', '-with-')
    self.addSubstitution('CC', self.CC, comment = 'C compiler')
    return

  def configureDebuggers(self):
    self.checkProgram('gdb', getFullPath = 1, comment = 'GNU debugger')
    self.checkProgram('dbx', getFullPath = 1, comment = 'DBX debugger')
    self.checkProgram('xdb', getFullPath = 1, comment = 'XDB debugger')
    if hasattr(self, 'gdb'):
      self.addDefine('USE_GDB_DEBUGGER', 1, comment = 'Use GDB as the default debugger')
    elif hasattr(self, 'dbx'):
      self.addDefine('USE_DBX_DEBUGGER', 1, comment = 'Use DBX as the default debugger')
      f = file('conftest', 'w')
      f.write('quit\n')
      f.close()
      foundOption = 0
      if not foundOption:
        (status, output) = commands.getstatusoutput(self.dbx+' -c conftest -p '+os.getpid())
        for line in output:
          if re.match(r'Process '+os.getpid()):
            self.addDefine('USE_P_FOR_DEBUGGER', 1, comment = 'Use -p to indicate a process to the debugger')
            foundOption = 1
            break
      if not foundOption:
        (status, output) = commands.getstatusoutput(self.dbx+' -c conftest -a '+os.getpid())
        for line in output:
          if re.match(r'Process '+os.getpid()):
            self.addDefine('USE_A_FOR_DEBUGGER', 1, comment = 'Use -a to indicate a process to the debugger')
            foundOption = 1
            break
      if not foundOption:
        (status, output) = commands.getstatusoutput(self.dbx+' -c conftest -pid '+os.getpid())
        for line in output:
          if re.match(r'Process '+os.getpid()):
            self.addDefine('USE_PID_FOR_DEBUGGER', 1, comment = 'Use -pid to indicate a process to the debugger')
            foundOption = 1
            break
      os.remove('conftest')
    elif hasattr(self, 'xdb'):
      self.addDefine('USE_XDB_DEBUGGER', 1, comment = 'Use XDB as the default debugger')
      self.addDefine('USE_LARGEP_FOR_DEBUGGER', 1, comment = 'Use -P to indicate a process to the debugger')
    return

  def configureMissingPrototypes(self):
    self.addSubstitution('MISSING_PROTOTYPES',     '', comment = 'C compiler')
    self.addSubstitution('MISSING_PROTOTYPES_CPP', '', comment = 'C compiler')
    self.missingPrototypesExternC = ''
    if self.archBase == 'linux':
      self.missingPrototypesExternC += 'extern void *memalign(int, int);'
    self.addSubstitution('MISSING_PROTOTYPES_EXTERN_C', self.missingPrototypesExternC, comment = 'C compiler')
    return

  def configureHeaders(self):
    #self.checkHeader('dos.h')
    return

  def configure(self):
    self.configureDirectories()
    self.configureArchitecture()
    self.configureLibraryOptions()
    self.configureCompilers()
    self.configureDebuggers()
    self.configureMissingPrototypes()
    self.configureHeaders()
    return
