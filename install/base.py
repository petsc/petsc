import maker

import urlparse
# Fix parsing for nonstandard schemes
urlparse.uses_netloc.extend(['bk', 'ssh'])

class Base (maker.Maker):
  def __init__(self, argDB, base = ''):
    maker.Maker.__init__(self, argDB)
    self.base = base
    self.checkPython()
    self.checkNumeric()
    return

  def checkPython(self):
    import sys

    if not hasattr(sys, 'version_info') or float(sys.version_info[0]) < 2 or float(sys.version_info[1]) < 2:
      raise RuntimeError('BuildSystem requires Python version 2.2 or higher. Get Python at http://www.python.org')
    return

  def checkNumeric(self):
    import distutils.sysconfig
    import os

    try:
      import Numeric
    except ImportError, e:
      raise RuntimeError('BuildSystem requires Numeric Python (http://www.pfdubois.com/numpy) to be installed: '+str(e))
    header = os.path.join(distutils.sysconfig.get_python_inc(), 'Numeric', 'arrayobject.h')
    if not os.path.exists(header):
      raise RuntimeError('The include files from the Numeric are misplaced: Cannot find '+header)
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

  def getRepositoryName(self, url):
    '''Return the repository name from a project URL. This is the base filename that should be used for tarball distributions.'''
    import os
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    return os.path.basename(path)

  def getRepositoryPath(self, url, noBase = 0):
    '''Return the repository path from a project URL. This is the name that should be used for alternate retrieval.
    - You can omit the repository name itself by giving the noBase flag'''
    import os
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    sitename = location.split('.')[0]
    pathname = path[1:]
    if noBase: pathname = os.path.dirname(pathname)
    return os.path.join(sitename, pathname)
