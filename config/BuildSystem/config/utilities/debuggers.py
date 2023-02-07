from __future__ import generators
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
    help.addArgument('PETSc', '-with-debugger=<gdb,dbx,etc>',   nargs.Arg(None, None, 'Default debugger with PETSc -start_in_debugger option'))
    return

  def configureDebuggers(self):
    '''Find a default debugger and determine its arguments'''
    '''If Darwin first try lldb, next try gdb and dbx'''
    # Use the framework in order to remove the PETSC_ namespace
    if 'with-debugger' in self.argDB:
      found = self.getExecutable(self.argDB['with-debugger'], getFullPath = 1)
      if not found:
        raise RuntimeError('Cannot locate debugger indicated using --with-debugger='+self.argDB['with-debugger'])
      self.addDefine('USE_DEBUGGER','"'+self.argDB['with-debugger']+'"')
    else:
      if config.setCompilers.Configure.isDarwin(self.log):
        self.getExecutable('lldb', getFullPath = 1)
        if hasattr(self,'lldb'):
          self.addDefine('USE_DEBUGGER','"lldb"')
      if not hasattr(self,'lldb'):
        self.getExecutable('gdb', getFullPath = 1)
        if hasattr(self,'gdb'):
          self.addDefine('USE_DEBUGGER','"gdb"')
        else:
          self.getExecutable('dbx', getFullPath = 1)
          if hasattr(self,'dbx'):
            self.addDefine('USE_DEBUGGER','"dbx"')

    if config.setCompilers.Configure.isDarwin(self.log):
      self.getExecutable('dsymutil', getFullPath = 1)
    else:
      self.dsymutil = 'true'
    self.addMakeMacro('DSYMUTIL', self.dsymutil)

    if config.setCompilers.Configure.isDarwin(self.log):
      # This seems to be needed around version 11.5 of XCode
      # It would be good to have a configure test for this, not sure if this will work for older versions of XCode
      self.addDefine('DO_NOT_SWAP_CHILD_FOR_DEBUGGER',1)

    if hasattr(self, 'dbx'):
      import re
      if self.argDB['with-batch']: return
      f = open('conftest', 'w')
      f.write('quit\n')
      f.close()
      foundOption = 0
      if not foundOption:
        pid = os.fork()
        if not pid:
          import time
          import sys
          time.sleep(15)
          sys.exit()
        try:
          (output, error, status) = config.base.Configure.executeShellCommand(self.dbx+' -c conftest -p '+str(pid), log = self.log)
          if not status:
            for line in output:
              if re.match(r'Process '+str(pid), line):
                self.addDefine('USE_P_FOR_DEBUGGER', 1)
                foundOption = 1
                break
        except RuntimeError: pass
      if not foundOption:
        try:
          (output, error, status) = config.base.Configure.executeShellCommand(self.dbx+' -c conftest -a '+str(pid), log = self.log)
          if not status:
            for line in output:
              if re.match(r'Process '+str(pid), line):
                self.addDefine('USE_A_FOR_DEBUGGER', 1)
                foundOption = 1
                break
        except RuntimeError: pass
      if not foundOption:
        try:
          (output, error, status) = config.base.Configure.executeShellCommand(self.dbx+' -c conftest -pid '+str(pid), log = self.log)
          if not status:
            for line in output:
              if re.match(r'Process '+str(pid), line):
                self.addDefine('USE_PID_FOR_DEBUGGER', 1)
                foundOption = 1
                break
        except RuntimeError: pass
      os.remove('conftest')
    return

  def configure(self):
    self.executeTest(self.configureDebuggers)
    return
