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
    self.getExecutable('mkdir', getFullPath = 1,setMakeMacro=0)
    if hasattr(self, 'mkdir'):
      confDir = '.conftest'
      conftmpDir  = os.path.join('.conftest','tmp')
      if os.path.exists(conftmpDir): os.rmdir(conftmpDir)
      if os.path.exists(confDir): os.rmdir(confDir)
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(self.mkdir+' -p '+conftmpDir, log = self.framework.log)
        if not status and os.path.isdir(conftmpDir):
          self.mkdir = self.mkdir+' -p'
      except RuntimeError: pass
      self.addMakeMacro('MKDIR',self.mkdir)
      if os.path.exists(conftmpDir): os.rmdir(conftmpDir)
      if os.path.exists(confDir): os.rmdir(confDir)
    return

  def configurePrograms(self):
    '''Check for the programs needed to build and run PETSc'''
    self.getExecutable('sh',   getFullPath = 1, resultName = 'SHELL')
    if not hasattr(self, 'sh'): raise RuntimeError('Could not locate sh executable')
    self.getExecutable('sed',  getFullPath = 1)
    if not hasattr(self, 'sed'): raise RuntimeError('Could not locate sed executable')
    self.getExecutable('mv',   getFullPath = 1)
    if not hasattr(self, 'mv'): raise RuntimeError('Could not locate mv executable')
    self.getExecutable('cp',   getFullPath = 1)
    if not hasattr(self, 'cp'): raise RuntimeError('Could not locate cp executable')
    self.getExecutable('grep', getFullPath = 1)    
    if not hasattr(self, 'grep'): raise RuntimeError('Could not locate grep executable')
    self.getExecutable('rm -f',getFullPath = 1, resultName = 'RM')
    if not hasattr(self, 'rm'): raise RuntimeError('Could not locate rm executable')
    self.getExecutable('diff', getFullPath = 1,setMakeMacro=0)
    if hasattr(self, 'diff'):
      # check if diff supports -w option for ignoring whitespace
      f = file('diff1', 'w')
      f.write('diff\n')
      f.close()
      f = file('diff2', 'w')
      f.write('diff  \n')
      f.close()
      (out,err,status) = Configure.executeShellCommand(self.diff+' -w diff1 diff2')
      os.unlink('diff1')
      os.unlink('diff2')
      if not status:    
        self.diff = self.diff + ' -w'
      self.addMakeMacro('DIFF',self.diff)
    else:
      raise RuntimeError('Could not locate diff executable')
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
