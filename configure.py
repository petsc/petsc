import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    libraries = [('cygwin', 'log')]
    # Add dependencies
    self.libraries = self.framework.require('config.libraries', self)
    self.libraries.libraries.extend(libraries)
    return

  def configureCygwin(self):
    '''If libcygwin.a is found, define HAVE_CYGWIN'''
    if self.libraries.haveLib('cygwin'):
      self.framework.addDefine('HAVE_CYGWIN', 1)
    return

  def configure(self):
    import os

    self.executeTest(self.configureCygwin)
    return
