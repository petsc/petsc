import maker

import os
import urlparse
# Fix parsing for nonstandard schemes
urlparse.uses_netloc.extend(['bk', 'ssh'])

class Base (maker.Maker):
  def __init__(self, argDB, base = ''):
    maker.Maker.__init__(self, argDB)
    self.base    = base
    self.urlMaps = []
    self.checkPython()
    self.checkNumeric()
    self.setupUrlMapping()
    return

  def checkPython(self):
    import sys

    if not hasattr(sys, 'version_info') or float(sys.version_info[0]) < 2 or float(sys.version_info[1]) < 2:
      raise RuntimeError('BuildSystem requires Python version 2.2 or higher. Get Python at http://www.python.org')
    return

  def checkNumeric(self):
    import distutils.sysconfig

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
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    return os.path.basename(path)

  def getRepositoryPath(self, url, noBase = 0):
    '''Return the repository path from a project URL. This is the name that should be used for alternate retrieval.
    - You can omit the repository name itself by giving the noBase flag'''
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    sitename = location.split('.')[0]
    pathname = path[1:]
    if noBase: pathname = os.path.dirname(pathname)
    return os.path.join(sitename, pathname)

  def getInstallRoot(self, url, isBackup = 0):
    '''Guess the install root from the project URL. Note this method automatically remaps the URL.'''
    url  = self.getMappedUrl(url)
    root = self.getRepositoryName(url)
    if isBackup:
      root = os.path.join('backup', root)
    if self.base:
      root = os.path.join(self.base, root)
    return os.path.abspath(root)

  def bootstrapUrlMap(self, url):
    if self.checkBootstrap():
      (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
      if scheme == 'bk':
        path = os.path.join('/pub', 'petsc', self.getRepositoryPath(url)+'.tgz')
        return (1, urlparse.urlunparse(('ftp', 'ftp.mcs.anl.gov', path, parameters, query, fragment)))
    return (0, url)

  def setupUrlMapping(self):
    self.urlMaps.append(self.bootstrapUrlMap)
    if not self.argDB.has_key('urlMappingModules') or not self.argDB['urlMappingModules']:
      self.argDB['urlMappingModules'] = []
    elif not isinstance(self.argDB['urlMappingModules'], list):
      self.argDB['urlMappingModules'] = [self.argDB['urlMappingModules']]
    for moduleName in self.argDB['urlMappingModules']:
      __import__(moduleName, globals(), locals(), ['setupUrlMapping']).setupUrlMapping(self, self.urlMaps)
    return

  def getMappedUrl(self, url):
    '''Return a new URL produced by a URL map function. Users can register new maps by adding to the list self.urlMaps'''
    for map in self.urlMaps:
      ret, newUrl = map(url)
      if ret: return newUrl
    return url
