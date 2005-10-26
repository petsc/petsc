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

    help.addArgument('SourceControl', '-with-bk=<executable>',  nargs.Arg(None, 'bk', 'Specify the BitKeeper executable'))
    help.addArgument('SourceControl', '-with-cvs=<executable>', nargs.Arg(None, 'cvs', 'Specify the CVS executable'))
    help.addArgument('SourceControl', '-with-svn=<executable>', nargs.Arg(None, 'svn', 'Specify the Subversion executable'))
    return

  def configureBitKeeper(self):
    '''Find the BitKeeper executable'''
    if 'with-bk' in self.framework.argDB and self.framework.argDB['with-bk'] == '0':
      return
    self.getExecutable(self.framework.argDB['with-bk'], resultName = 'bk')
    return

  def configureCVS(self):
    '''Find the CVS executable'''
    self.getExecutable(self.framework.argDB['with-cvs'], resultName = 'cvs')
    return

  def configureSubversion(self):
    '''Find the Subversion executable'''
    self.getExecutable(self.framework.argDB['with-svn'], resultName = 'svn')
    return

  def configure(self):
    self.executeTest(self.configureBitKeeper)
    self.executeTest(self.configureCVS)
    self.executeTest(self.configureSubversion)
    return
