import config.base

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headers = self.framework.require('config.headers', self)
    return

  def checkCurses(self):
    '''Verify that we have the curses header'''
    return self.headers.check('curses.h')

  def configure(self):
    self.executeTest(self.checkCurses)
    return
