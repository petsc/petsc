from __future__ import generators
import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.have_command_argument = False
    return

  def __str__(self):
    return ''

  def setupHelp(self, help):
    import nargs
    return

  def setupDependencies(self, framework):
    config.base.Configure.setupDependencies(self, framework)
    self.compilers    = self.framework.require('config.compilers', self)
    self.setCompilers = self.framework.require('config.setCompilers', self)
    self.functions    = self.framework.require('config.functions', self)
    self.libraries    = framework.require('config.libraries',  self)
    return

  def configureFortranCommandLine(self):
    '''Check for the mechanism to retrieve command line arguments in Fortran'''

    # These are for when the routines are called from Fortran

    self.libraries.pushLanguage('FC')
    self.libraries.saveLog()
    if self.libraries.check('','', call = '      integer i\n      character(len=80) arg\n      i = command_argument_count()\n      call get_command_argument(i,arg)'):
      self.logWrite(self.libraries.restoreLog())
      self.libraries.popLanguage()
      self.have_command_argument = True
    else:
      self.logPrint("Missing GET_COMMAND_ARGUMENT() support in Fortran!")
    return

  def configure(self):
    if hasattr(self.setCompilers, 'FC'):
      self.executeTest(self.configureFortranCommandLine)
    return
