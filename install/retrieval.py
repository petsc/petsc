import install.base
import maker

import os
import urllib
import urlparse

# Fix parsing for nonstandard schemes
urlparse.uses_netloc.extend(['bk', 'ssh'])

class Retriever(install.base.Base):
  def __init__(self, argDB, base = ''):
    install.base.Base.__init__(self, argDB, base)
    return

  def getInstallRoot(self, url):
    '''Guess the install root from the project URL'''
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    path = path[1:]
    if self.base:
      path = os.path.join(base, path)
    return os.path.abspath(path)

  def genericRetrieve(self, url, root, canExist = 0, force = 0):
    localFile = root+'.tar.gz'
    if os.path.exists(root):
      if canExist:
        if force:
          output = self.executeShellCommand('rm -rf '+root)
        else:
          return root
      else:
        raise RuntimeError('Root directory '+root+' already exists')
    if os.path.exists(localFile):
      os.remove(localFile)
    urllib.urlretrieve(url, localFile)
    output = self.executeShellCommand('tar -zxf '+localFile)
    os.remove(localFile)
    return root

  def ftpRetrieve(self, url, root, canExist = 0, force = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via ftp', 3, 'install')
    return self.genericRetrieve(url, root, canExist, force)

  def httpRetrieve(self, url, root, canExist = 0, force = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via http', 3, 'install')
    return self.genericRetrieve(url, root, canExist, force)

  def bkRetrieve(self, url, root, canExist = 0, force = 0):
    if self.checkBootstrap():
      (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
      path   = os.path.join('/pub', 'petsc', location.split('.')[0], path[1:]+'.tgz')
      newUrl = urlparse.urlunparse(('ftp', 'ftp.mcs.anl.gov', path, parameters, query, fragment))
      return self.ftpRetrieve(newUrl, root, canExist, force)

    self.debugPrint('Retrieving '+url+' --> '+root+' via bk', 3, 'install')
    if os.path.exists(root):
      output = self.executeShellCommand('cd '+root+'; bk pull')
    else:
      (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
      if location.find('@') < 0:
        login  = location.split('.')[0]
        newUrl = urlparse.urlunparse((scheme, login+'@'+location, path, parameters, query, fragment))
        try:
          output = self.executeShellCommand('bk clone '+newUrl+' '+root)
        except RuntimeError:
          pass
        else:
          return root
      output = self.executeShellCommand('bk clone '+url+' '+root)
    return root

  def sshRetrieve(self, url, root, canExist = 0, force = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via ssh', 3, 'install')
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    (dir, project) = os.path.split(path)
    if os.path.exists(root):
      if canExist:
        if force:
          output = self.executeShellCommand('rm -rf '+root)
        else:
          return root
      else:
        raise RuntimeError('Root directory '+root+' already exists')
    command = 'ssh '+location+' "tar -C '+dir+' -zc '+project+'" | tar -C '+root+' -zx'
    output  = self.executeShellCommand(command)
    return root

  def retrieve(self, url, root = None, canExist = 0, force = 0):
    project = self.getInstalledProject(url)
    if not project is None:
      root     = project.getRoot()
      canExist = 1
    if root is None:
      root = self.getInstallRoot(url)
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    try:
      if self.argDB['retrievalCanExist']:
        canExist = 1
      return getattr(self, scheme+'Retrieve')(url, os.path.abspath(root), canExist, force)
    except AttributeError:
      raise RuntimeError('Invalid transport for retrieval: '+scheme)
