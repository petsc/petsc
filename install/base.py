import maker

class Base (maker.Maker):
  def __init__(self, argDB, base = ''):
    maker.Maker.__init__(self, argDB)
    self.argDB = argDB
    self.base  = base
    return

  def getInstalledProject(self, url):
    for project in self.argDB['installedprojects']:
      if project.getUrl() == url:
        self.debugPrint('Already installed '+project.getName()+'('+url+')', 3, 'install')
        return project
    return None
