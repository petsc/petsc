import install.urlMapping

import os
import urllib
import urlparse
# Fix parsing for nonstandard schemes
urlparse.uses_netloc.extend(['bk', 'ssh'])

class Retriever(install.urlMapping.UrlMapping):
  def __init__(self):
    install.urlMapping.UrlMapping.__init__(self)
    return

  def removeRoot(self,root,canExist,force = 0):
    '''Returns 1 if removes root'''
    if os.path.exists(root):
      if canExist:
        if force:
          import shutil
          shutil.rmtree(root)
          return 1
        else:
          return 0
      else:
        raise RuntimeError('Root directory '+root+' already exists')
    return 1
    
  def genericRetrieve(self, url, root, canExist = 0, force = 0):
    '''Fetch the gzipped tarfile indicated by url and expand it into root
    - We append .tgz to url automatically
    - There is currently no check that the root inside the tarfile matches the indicated root'''
    if not self.removeRoot(root, canExist, force): return root
    localFile = root+'.tgz'
    if os.path.exists(localFile):
      os.remove(localFile)
    dir = os.path.dirname(localFile)
    if dir and not os.path.exists(dir):
      os.makedirs(dir)
    urllib.urlretrieve(url+'.tgz', localFile)
    if dir:
      output = self.executeShellCommand('tar -zxf '+localFile+' -C '+dir)
    else:
      output = self.executeShellCommand('tar -zxf '+localFile)
    os.remove(localFile)
    return root

  def ftpRetrieve(self, url, root, canExist = 0, force = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via ftp', 3, 'install')
    return self.genericRetrieve(url, root, canExist, force)

  def httpRetrieve(self, url, root, canExist = 0, force = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via http', 3, 'install')
    return self.genericRetrieve(url, root, canExist, force)

  def fileRetrieve(self, url, root, canExist = 0, force = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via cp', 3, 'install')
    return self.genericRetrieve(url, root, canExist, force)

  def getAuthorizedUrl(self, url):
    '''This returns a tuple of the unauthorized and authorized URLs for the given repository, as well as a flag indicating which was input'''
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    if not location:
      url     = urlparse.urlunparse(('','', path, parameters, query, fragment))
      authUrl = None
      wasAuth = 0
    else:
      index = location.find('@')
      if index >= 0:
        login   = location[0:index]
        authUrl = url
        url     = urlparse.urlunparse((scheme, location[index+1:], path, parameters, query, fragment))
        wasAuth = 1
      else:
        login   = location.split('.')[0]
        authUrl = urlparse.urlunparse((scheme, login+'@'+location, path, parameters, query, fragment))
        wasAuth = 0
    return (url, authUrl, wasAuth)

  def testAuthorizedUrl(self, authUrl):
    return self.executeShellCommand('echo "quit" | ssh -oBatchMode=yes '+authUrl)

  def getBKParentURL(self, root):
    '''Return the parent URL for the BK repository at "root"'''
    return self.executeShellCommand('cd '+root+'; bk parent')[21:]

  def bkRetrieve(self, url, root, canExist = 0, force = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via bk', 3, 'install')
    if os.path.exists(root):
      (url, authUrl, wasAuth) = self.getAuthorizedUrl(self.getBKParentURL(root))
      if not wasAuth:
        self.debugPrint('Changing parent from '+url+' --> '+authUrl, 1, 'install')
        output = self.executeShellCommand('cd '+root+'; bk parent '+authUrl)
      try:
        self.testAuthorizedUrl(authUrl)
        output = self.executeShellCommand('cd '+root+'; bk pull')
      except RuntimeError, e:
        (url, authUrl, wasAuth) = self.getAuthorizedUrl(self.getBKParentURL(root))
        if wasAuth:
          self.debugPrint('Changing parent from '+authUrl+' --> '+url, 1, 'install')
          output = self.executeShellCommand('cd '+root+'; bk parent '+url)
          output = self.executeShellCommand('cd '+root+'; bk pull')
        else:
          raise e
    else:
      (url, authUrl, wasAuth) = self.getAuthorizedUrl(url)
      # Try an authorized login first
      try:
        self.testAuthorizedUrl(authUrl)
        self.executeShellCommand('echo "quit" | ssh -oBatchMode=yes '+authUrl)
        output = self.executeShellCommand('bk clone '+authUrl+' '+root)
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
    if not self.removeRoot(root,canExist,force): return root
    command = 'ssh '+location+' "tar -C '+dir+' -zc '+project+'" | tar -C '+root+' -zx'
    output  = self.executeShellCommand(command)
    return root

  def retrieve(self, url, root = None, canExist = 0, force = 0):
    '''Retrieve the project corresponding to url
    - If root is None, the local root directory is automatically determined. If the project
      was already installed, this root is used. Otherwise a guess is made based upon the url.
    - If canExist is True and the root exists, an update is done instead of a full download.
      The canExist is automatically true if the project has been installed. The retrievalCanExist
      flag can also be used to set this.
    - If force is True, a full dowmload is mandated.
    Providing the root is an easy way to make a copy, for instance when making tarballs.
    '''
    origUrl = url
    url     = self.getMappedUrl(origUrl)
    project = self.getInstalledProject(url)
    if not project is None and root is None:
      root     = project.getRoot()
      canExist = 1
    if root is None:
      root = self.getInstallRoot(origUrl)
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    try:
      if self.argDB['retrievalCanExist']:
        canExist = 1
      return getattr(self, scheme+'Retrieve')(url, os.path.abspath(root), canExist, force)
    except AttributeError:
      raise RuntimeError('Invalid transport for retrieval: '+scheme)
