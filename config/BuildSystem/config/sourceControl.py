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

    help.addArgument('SourceControl', '-with-git=<prog>', nargs.Arg(None, 'git','Specify the Git executable'))
    help.addArgument('SourceControl', '-with-hg=<prog>',  nargs.Arg(None, 'hg', 'Specify the Mercurial executable'))
    return

  def configureGit(self):
    '''Find the Git executable'''
    if 'with-git' in self.argDB and self.argDB['with-git'] == '0':
      return
    self.getExecutable(self.argDB['with-git'], resultName = 'git', setMakeMacro = 0)
    if hasattr(self,'git'):
      try:
        self.gitversion = self.executeShellCommand(self.git + ' --version', log = self.log)
      except: pass
    return

  def configureMercurial(self):
    '''Find the Mercurial executable'''
    if 'with-hg' in self.argDB and self.argDB['with-hg'] == '0':
      return
    self.getExecutable(self.argDB['with-hg'], resultName = 'hg', setMakeMacro = 0)
    if hasattr(self,'hg'):
      try:
        self.hgversion = self.executeShellCommand(self.hg + ' version -q', log = self.log)
      except: pass
    return

  def configure(self):
    self.executeTest(self.configureGit)
    self.executeTest(self.configureMercurial)
    return
