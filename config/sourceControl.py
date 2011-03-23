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

    help.addArgument('SourceControl', '-with-hg=<prog>',  nargs.Arg(None, 'hg', 'Specify the Mercurial executable'))
    help.addArgument('SourceControl', '-with-cvs=<prog>', nargs.Arg(None, 'cvs', 'Specify the CVS executable'))
    help.addArgument('SourceControl', '-with-svn=<prog>', nargs.Arg(None, 'svn', 'Specify the Subversion executable'))
    return

  def configureMercurial(self):
    '''Find the Mercurial executable'''
    if 'with-hg' in self.framework.argDB and self.framework.argDB['with-hg'] == '0':
      return
    self.getExecutable(self.framework.argDB['with-hg'], resultName = 'hg')
    if hasattr(self,'hg'):
      try:
        self.hgversion = self.executeShellCommand(self.hg + ' version -q')
      except: pass
    return

  def configureCVS(self):
    '''Find the CVS executable'''
    self.getExecutable(self.framework.argDB['with-cvs'], resultName = 'cvs')
    if hasattr(self,'cvs'):
      try:
        self.cvxversion = self.executeShellCommand(self.cvs + ' --version')
      except: pass
    return

  def configureSubversion(self):
    '''Find the Subversion executable'''
    self.getExecutable(self.framework.argDB['with-svn'], resultName = 'svn')
    if hasattr(self,'svn'):
      try:
        self.svnversion = self.executeShellCommand(self.svn + ' --version -q')
      except: pass
    return

  def configure(self):
    self.executeTest(self.configureMercurial)
    self.executeTest(self.configureCVS)
    self.executeTest(self.configureSubversion)
    return
