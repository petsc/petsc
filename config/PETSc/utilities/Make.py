#!/usr/bin/env python
from __future__ import generators
import user
import config.base

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
    help.addArgument('PETSc', '-with-make=<makename>', nargs.Arg(None, 'make', 'Specify make'))
    return

  def configureMake(self):
    '''Check various things about make'''
    self.getExecutable(self.framework.argDB['with-make'], getFullPath = 1,resultName = 'make')

    if not hasattr(self,'make'):
      import os
      if os.path.exists('/usr/bin/cygcheck.exe') and not os.path.exists('/usr/bin/make'):
        raise RuntimeError('''\
*** Incomplete cygwin install detected . /usr/bin/make is missing. **************
*** Please rerun cygwin-setup and select module "make" for install.**************''')
      else:
        raise RuntimeError('Could not locate the make utility on your system, make sure\n it is in your path or use --with-make=/fullpathnameofmake\n and run config/configure.py again')    
    # Check for GNU make
    haveGNUMake = 0
    self.getExecutable('strings', getFullPath = 1)
    if hasattr(self, 'strings'):
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(self.strings+' '+self.make, log = self.framework.log)
        if not status and output.find('GNU Make') >= 0:
          haveGNUMake = 1
      except RuntimeError, e:
        self.framework.log.write('Make check failed: '+str(e)+'\n')
      if not haveGNUMake:
        try:
          (output, error, status) = config.base.Configure.executeShellCommand(self.strings+' '+self.make+'.exe', log = self.framework.log)
          if not status and output.find('GNU Make') >= 0:
            haveGNUMake = 1
        except RuntimeError, e:
          self.framework.log.write('Make check failed: '+str(e)+'\n')
    # mac has fat binaries where 'string' check fails
    if not haveGNUMake:
      try:
        (output, error, status) = config.base.Configure.executeShellCommand(self.make+' -v dummy-foobar', log = self.framework.log)
        if not status and output.find('GNU Make') >= 0:
          haveGNUMake = 1
      except RuntimeError, e:
        self.framework.log.write('Make check failed: '+str(e)+'\n')
        
    # Setup make flags
    self.flags = ''
    if haveGNUMake:
      self.flags += ' --no-print-directory'
    self.addMakeMacro('OMAKE ', self.make+' '+self.flags)
      
    # Check to see if make allows rules which look inside archives
    if haveGNUMake:
      self.addMakeRule('libc','${LIBNAME}(${OBJSC} ${SOBJSC})')
    else:
      self.addMakeRule('libc','${OBJSC}','-${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSC}')
    self.addMakeRule('libf','${OBJSF}','-${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSF}')
    return

  def configure(self):
    self.executeTest(self.configureMake)
    return
