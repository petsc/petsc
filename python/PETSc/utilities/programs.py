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
    return

  def configureMkdir(self):
    '''Make sure we can have mkdir automatically make intermediate directories'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.getExecutable('mkdir', getFullPath = 1)
    if hasattr(self.framework, 'mkdir'):
      self.mkdir = self.framework.mkdir
      if os.path.exists('.conftest'): os.rmdir('.conftest')
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(self.mkdir+' -p .conftest/.tmp', log = self.framework.log)
        if not status and os.path.isdir('.conftest/.tmp'):
          self.mkdir = self.mkdir+' -p'
      except RuntimeError: pass
      if os.path.exists('.conftest'): os.removedirs('.conftest/.tmp')
    return

  def configurePrograms(self):
    '''Check for the programs needed to build and run PETSc'''
    # We use the framework in order to remove the PETSC_ namespace
    self.framework.getExecutable('sh',   getFullPath = 1, resultName = 'SHELL')
    self.framework.getExecutable('sed',  getFullPath = 1)
    self.framework.getExecutable('mv',   getFullPath = 1)
    self.framework.getExecutable('diff', getFullPath = 1)
    self.framework.getExecutable('rm -f',getFullPath = 1)
    # check if diff supports -w option for ignoring whitespace
    f = file('diff1', 'w')
    f.write('diff\n')
    f.close()
    f = file('diff2', 'w')
    f.write('diff  \n')
    f.close()
    (out,err,status) = Configure.executeShellCommand(getattr(self.framework, 'diff')+' -w diff1 diff2')
    os.unlink('diff1')
    os.unlink('diff2')
    if not status:    
      self.framework.diff = self.framework.diff + ' -w'
      
    self.framework.getExecutable('ps',   path = '/usr/ucb:/usr/usb', resultName = 'UCBPS')
    if hasattr(self.framework, 'UCBPS'):
      self.addDefine('HAVE_UCBPS', 1)
    self.framework.getExecutable('gzip',getFullPath=1, resultName = 'GZIP')
    if hasattr(self.framework, 'GZIP'):
      self.addDefine('HAVE_GZIP',1)
    return

  def configure(self):
    self.executeTest(self.configureMkdir)
    self.executeTest(self.configurePrograms)    
    return
