from __future__ import generators
import config.base
import os.path

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.foundADIC    = 0
    return

  def __str__(self):
    if self.foundADIC: return 'ADIC: Using '+self.adiC+'\n'
    return ''
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('ADIC', '-with-adic=<bool>',      nargs.ArgBool(None, 0, 'Activate ADIC'))
    help.addArgument('ADIC', '-with-adic-path=<path>', nargs.Arg(None, None, 'Full path of adic executable'))    
    return

  def generateADIC(self):
    '''Generate location of ADIC'''
    if 'with-adic-path' in self.framework.argDB:
      self.adiC = os.path.abspath(os.path.join(self.framework.argDB['with-adic-path'],'adiC'))
      return os.path.abspath(self.framework.argDB['with-adic-path'])
      raise RuntimeError('You set a value for --with-adic-path, but '+self.framework.argDB['with-adic-path']+' cannot be used\n')
    if self.getExecutable('adiC', getFullPath = 1,setMakeMacro=0):
      # follow any symbolic link of this path
      self.adiC = os.path.realpath(self.adiC)
      return os.path.dirname(os.path.dirname(self.adiC))
    return ''
    return

  def configureLibrary(self):
    '''Set adic make variables'''
    if self.generateADIC():
      self.addMakeMacro('ADIC_DEFINES', '')
      self.addDefine('HAVE_ADIC', 1)
      self.addMakeMacro('ADIC_CC',self.adiC+' -a -d gradient')
      self.foundADIC = 1
    else:
      raise RuntimeError('with-adic is enabled - however adiC executable is not found. Please specify with -with-adic-path option\n')
    return

  def configure(self):
    if self.framework.argDB['with-adic']:
      self.executeTest(self.configureLibrary)
    return
