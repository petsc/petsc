import maker

import imp

class Base (maker.Maker):
  def __init__(self, argDB, base = ''):
    maker.Maker.__init__(self, argDB)
    self.argDB = argDB
    self.base  = base
    return

  def checkBootstrap(self):
    '''If the compiler or runtime is not available, we will have to bootstrap and this function returns true'''
    try:
      import SIDL.Loader
      import SIDLLanguage.Parser
    except ImportError:
      return 1
    return 0

  def getInstalledProject(self, url):
    if not self.argDB.has_key('installedprojects'):
      self.argDB['installedprojects'] = []
    for project in self.argDB['installedprojects']:
      if project.getUrl() == url:
        self.debugPrint('Already installed '+project.getName()+'('+url+')', 3, 'install')
        return project
    return None

  def getMakeModule(self, root):
    name = 'make'
    (fp, pathname, description) = imp.find_module(name, [root])
    try:
      return imp.load_module(name, fp, pathname, description)
    finally:
      if fp: fp.close()
