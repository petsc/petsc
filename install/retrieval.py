import install.base
import maker

import os
import urllib
import urlparse

# Fix parsing for nonstandard schemes
urlparse.uses_netloc.extend(['bk', 'ssh'])

class Retriever(install.base.Base):
  def __init__(self, argDB, base = ''):
    install.base.Base.__init__(self, argDB)
    self.argDB = argDB
    self.base  = base
    return

  def getRoot(self, url):
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    path = path[1:]
    if self.base:
      path = os.path.join(base, path)
    return os.path.abspath(path)

  def genericRetrieve(self, url, root, canExist = 0):
    localFile = root+'.tar.gz'
    if os.path.exists(root): raise RuntimeError('Root directory '+root+' already exists')
    if os.path.exists(localFile): raise RuntimeError('File '+localFile+' already exists')
    urllib.urlretrieve(url, localFile)
    output = self.executeShellCommand('tar -zxf '+localFile)
    return root

  def bkRetrieve(self, url, root, canExist = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via bk', 3, 'install')
    if os.path.exists(root):
      if not canExist:
        raise RuntimeError('Root directory '+root+' already exists')
      else:
        return root
    output = self.executeShellCommand('bk clone '+url+' '+root)
    return root

  def ftpRetrieve(self, url, root, canExist = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via ftp', 3, 'install')
    return self.genericRetrieve(url, root, canExist)

  def httpRetrieve(self, url, root, canExist = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via http', 3, 'install')
    return self.genericRetrieve(url, root, canExist)

  def sshRetrieve(self, url, root, canExist = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via ssh', 3, 'install')
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    (dir, project) = os.path.split(path)
    command        = 'ssh '+location+' "tar -C '+dir+' -zc '+project+'" | tar -C '+root+' -zx'
    output         = self.executeShellCommand(command)
    return root

  def retrieve(self, url, root = None, canExist = 0):
    project = self.getInstalledProject(url)
    if not project is None:
      return project.getRoot()
    if root is None: root = self.getRoot(url)
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    try:
      if self.argDB.has_key('retrievalCanExist') and int(self.argDB['retrievalCanExist']):
        canExist = 1
      return getattr(self, scheme+'Retrieve')(url, os.path.abspath(root), canExist)
    except AttributeError:
      raise RuntimeError('Invalid transport for retrieval: '+scheme)
