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
    help.addArgument('Make', '-with-make', nargs.Arg(None, 'make', 'Specify the make executable'))
    return

  def configureMake(self):
    '''Check various things about make'''
    self.getExecutable(self.framework.argDB['with-make'], getFullPath = 1)
    # Check for GNU make
    haveGNUMake = 0
    try:
      (output, error, status) = config.base.Configure.executeShellCommand('strings '+self.make, log = self.framework.log)
      if not status and output.find('GNU Make') >= 0:
        haveGNUMake = 1
    except RuntimeError, e:
      self.framework.log.write('Make check failed: '+str(e)+'\n')
    if not haveGNUMake:
      try:
        (output, error, status) = config.base.Configure.executeShellCommand('strings '+self.make+'.exe', log = self.framework.log)
        if not status and output.find('GNU Make') >= 0:
          haveGNUMake = 1
      except RuntimeError, e:
        self.framework.log.write('Make check failed: '+str(e)+'\n')
    # Setup make flags
    flags = ''
    if haveGNUMake:
      flags += ' --no-print-directory'
    self.framework.addSubstitution('MAKE_FLAGS', flags.strip())
    self.framework.addSubstitution('SET_MAKE', '')
    # Check to see if make allows rules which look inside archives
    if haveGNUMake:
      self.framework.addSubstitution('LIB_C_TARGET', 'libc: ${LIBNAME}(${OBJSC} ${SOBJSC})')
      self.framework.addSubstitution('LIB_F_TARGET', '''
libf: ${OBJSF}
	${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSF}''')
    else:
      self.framework.addSubstitution('LIB_C_TARGET', '''
libc: ${OBJSC}
	${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSC}''')
      self.framework.addSubstitution('LIB_F_TARGET', '''
libf: ${OBJSF}
	${AR} ${AR_FLAGS} ${LIBNAME} ${OBJSF}''')
    return

  def configure(self):
    self.executeTest(self.configureMake)
    return
