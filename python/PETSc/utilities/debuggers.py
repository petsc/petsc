#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('PETSc', '-with-debugger=<gdb,dbx,etc>',   nargs.Arg(None, None, 'Debugger to use in PETSc'))
    return
      
  def configureDebuggers(self):
    '''Find a default debugger and determine its arguments'''
    # We use the framework in order to remove the PETSC_ namespace
    if 'with-debugger' in self.framework.argDB:
      self.framework.getExecutable(self.framework.argDB['with-debugger'], getFullPath = 1)
      if not hasattr(self,self.framework.argDB['with-debugger']):
        raise RuntimeError('Cannot locate debugger indicated using --with-debugger='+self.framework.argDB['with-debugger'])
    else:                               
      self.framework.getExecutable('gdb', getFullPath = 1)
      self.framework.getExecutable('dbx', getFullPath = 1)
      self.framework.getExecutable('xdb', getFullPath = 1)
      
    if hasattr(self, 'gdb'):
      self.addDefine('USE_GDB_DEBUGGER', 1)
    elif hasattr(self, 'dbx'):
      self.addDefine('USE_DBX_DEBUGGER', 1)
      f = file('conftest', 'w')
      f.write('quit\n')
      f.close()
      foundOption = 0
      if not foundOption:
        try:
          (output, error, status) = config.base.Configure.executeShellCommand(self.dbx+' -c conftest -p '+os.getpid(), log = self.framework.log)
          if not status:
            for line in output:
              if re.match(r'Process '+os.getpid()):
                self.addDefine('USE_P_FOR_DEBUGGER', 1)
                foundOption = 1
                break
        except RuntimeError: pass
      if not foundOption:
        try:
          (output, error, status) = config.base.Configure.executeShellCommand(self.dbx+' -c conftest -a '+os.getpid(), log = self.framework.log)
          if not status:
            for line in output:
              if re.match(r'Process '+os.getpid()):
                self.addDefine('USE_A_FOR_DEBUGGER', 1)
                foundOption = 1
                break
        except RuntimeError: pass
      if not foundOption:
        try:
          (output, error, status) = config.base.Configure.executeShellCommand(self.dbx+' -c conftest -pid '+os.getpid(), log = self.framework.log)
          if not status:
            for line in output:
              if re.match(r'Process '+os.getpid()):
                self.addDefine('USE_PID_FOR_DEBUGGER', 1)
                foundOption = 1
                break
        except RuntimeError: pass
      os.remove('conftest')
    elif hasattr(self, 'xdb'):
      self.addDefine('USE_XDB_DEBUGGER', 1)
      self.addDefine('USE_LARGEP_FOR_DEBUGGER', 1)
    return

  def configure(self):
    self.executeTest(self.configureDebuggers)
    return
