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
    self.getExecutable('mkdir', getFullPath = 1)
    if hasattr(self, 'mkdir'):
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
    # should generate error if cannot locate one
    self.getExecutable('sh',   getFullPath = 1, resultName = 'SHELL')
    self.getExecutable('sed',  getFullPath = 1)
    self.getExecutable('mv',   getFullPath = 1)
    self.getExecutable('cp',   getFullPath = 1)
    self.getExecutable('diff', getFullPath = 1)
    self.getExecutable('rm -f',getFullPath = 1, resultName = 'RM')
    # check if diff supports -w option for ignoring whitespace
    f = file('diff1', 'w')
    f.write('diff\n')
    f.close()
    f = file('diff2', 'w')
    f.write('diff  \n')
    f.close()
    (out,err,status) = Configure.executeShellCommand(getattr(self, 'diff')+' -w diff1 diff2')
    os.unlink('diff1')
    os.unlink('diff2')
    if not status:    
      self.diff = self.diff + ' -w'
      
    self.getExecutable('ps',   path = '/usr/ucb:/usr/usb', resultName = 'UCBPS')
    if hasattr(self, 'UCBPS'):
      self.addDefine('HAVE_UCBPS', 1)
    self.getExecutable('gzip',getFullPath=1, resultName = 'GZIP')
    if hasattr(self, 'GZIP'):
      self.addDefine('HAVE_GZIP',1)
    return

  def configure(self):
    self.executeTest(self.configureMkdir)
    self.executeTest(self.configurePrograms)    
    return
