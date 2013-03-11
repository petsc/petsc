import logger

import os
import urllib
import urlparse
import config.base
# Fix parsing for nonstandard schemes
urlparse.uses_netloc.extend(['bk', 'ssh', 'svn'])

class Retriever(logger.Logger):
  def __init__(self, sourceControl, clArgs = None, argDB = None):
    logger.Logger.__init__(self, clArgs, argDB)
    self.sourceControl = sourceControl
    self.stamp = None
    return

  def getAuthorizedUrl(self, url):
    '''This returns a tuple of the unauthorized and authorized URLs for the given URL, and a flag indicating which was input'''
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    if not location:
      url     = urlparse.urlunparse(('', '', path, parameters, query, fragment))
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
    '''Raise an exception if the URL cannot receive an SSH login without a password'''
    if not authUrl:
      raise RuntimeError('Url is empty')
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(authUrl)
    return self.executeShellCommand('echo "quit" | ssh -oBatchMode=yes '+location)

  def genericRetrieve(self, url, root, name):
    '''Fetch the gzipped tarfile indicated by url and expand it into root
       - All the logic for removing old versions, updating etc. must move'''

    # get the tarball file name from the URL
    filename = os.path.basename(urlparse.urlparse(url)[2])
    localFile = os.path.join(root,'_d_'+filename)
    ext =  os.path.splitext(localFile)[1]
    if ext not in ['.bz2','.tbz','.gz','.tgz','.zip','.ZIP']:
      raise RuntimeError('Unknown compression type in URL: '+ url)
    self.logPrint('Downloading '+url+' to '+localFile)
    if os.path.exists(localFile):
      os.unlink(localFile)

    try:
      urllib.urlretrieve(url, localFile)
    except Exception, e:
      failureMessage = '''\
Unable to download package %s from: %s
* If URL specified manually - perhaps there is a typo?
* If your network is disconnected - please reconnect and rerun ./configure
* Or perhaps you have a firewall blocking the download
* Alternatively, you can download the above URL manually, to /yourselectedlocation/%s
  and use the configure option:
  --download-%s=/yourselectedlocation/%s
''' % (name, url, filename, name.lower(), filename)
      raise RuntimeError(failureMessage)

    self.logPrint('Extracting '+localFile)
    if ext in ['.zip','.ZIP']:
      config.base.Configure.executeShellCommand('cd '+root+'; unzip '+localFile, log = self.log)
      output = config.base.Configure.executeShellCommand('cd '+root+'; zipinfo -1 '+localFile+' | head -n 1', log = self.log)
      dirname = os.path.normpath(output[0].strip())
    else:
      failureMessage = '''\
Downloaded package %s from: %s is not a tarball.
[or installed python cannot process compressed files]
* If you are behind a firewall - please fix your proxy and rerun ./configure
  For example at LANL you may need to set the environmental variable http_proxy (or HTTP_PROXY?) to  http://proxyout.lanl.gov 
* Alternatively, you can download the above URL manually, to /yourselectedlocation/%s
  and use the configure option:
  --download-%s=/yourselectedlocation/%s
''' % (name, url, filename, name.lower(), filename)
      import tarfile
      try:
        tf  = tarfile.open(os.path.join(root, localFile))
      except tarfile.ReadError:
        raise RuntimeError(failureMessage)
      # some tarfiles list packagename/ but some list packagename/filename in the first entry
      if not tf: raise RuntimeError(failureMessage)
      if not tf.firstmember: raise RuntimeError(failureMessage)
      if tf.firstmember.isdir():
        dirname = tf.firstmember.name
      else:
        dirname = os.path.dirname(tf.firstmember.name)
      if hasattr(tf,'extractall'): #python 2.5+
        tf.extractall(root)
      else:
        for tfile in tf.getmembers():
          tf.extract(tfile,root)
      tf.close()

    # fix file permissions for the untared tarballs.
    try:
      config.base.Configure.executeShellCommand('cd '+root+'; chmod -R a+r '+dirname+';find  '+dirname + ' -type d -name "*" -exec chmod a+rx {} \;', log = self.log)
    except RuntimeError, e:
      raise RuntimeError('Error changing permissions for '+dirname+' obtained from '+localFile+ ' : '+str(e))
    os.unlink(localFile)
    return

  def ftpRetrieve(self, url, root, name,force):
    self.logPrint('Retrieving '+url+' --> '+os.path.join(root, name)+' via ftp', 3, 'install')
    return self.genericRetrieve(url, root, name)

  def httpRetrieve(self, url, root, name,force):
    self.logPrint('Retrieving '+url+' --> '+os.path.join(root, name)+' via http', 3, 'install')
    return self.genericRetrieve(url, root, name)

  def fileRetrieve(self, url, root, name,force):
    self.logPrint('Retrieving '+url+' --> '+os.path.join(root, name)+' via cp', 3, 'install')
    return self.genericRetrieve(url, root, name)

  def svnRetrieve(self, url, root, name,force):
    if not hasattr(self.sourceControl, 'svn'):
      raise RuntimeError('Cannot retrieve a SVN repository since svn was not found')
    self.logPrint('Retrieving '+url+' --> '+os.path.join(root, name)+' via svn', 3, 'install')
    try:
      config.base.Configure.executeShellCommand(self.sourceControl.svn+' checkout http'+url[3:]+' '+os.path.join(root, name))
    except RuntimeError:
      pass


  # This is the old code for updating a BK repository
  # Stamp used to be stored with a url
  def bkUpdate(self):
    if not self.stamp is None and url in self.stamp:
      if not self.stamp[url] == self.bkHeadRevision(root):
        raise RuntimeError('Existing stamp for '+url+' does not match revision of repository in '+root)
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
    return

  def bkClone(self, url, root, name):
    '''Clone a Bitkeeper repository located at url into root/name
       - If self.stamp exists, clone only up to that revision'''
    failureMessage = '''\
Unable to bk clone %s
You may be off the network. Connect to the internet and run ./configure again
or from the directory %s try:
  bk clone %s
and if that succeeds then rerun ./configure
''' % (name, root, url, name)
    try:
      if not self.stamp is None and url in self.stamp:
        (output, error, status) = self.executeShellCommand('bk clone -r'+self.stamp[url]+' '+url+' '+os.path.join(root, name))
      else:
        (output, error, status) = self.executeShellCommand('bk clone '+url+' '+os.path.join(root, name))
    except RuntimeError, e:
      status = 1
      output = str(e)
      error  = ''
    if status:
      if output.find('ommand not found') >= 0:
        failureMessage = 'Unable to locate bk (Bitkeeper) to download repository; make sure bk is in your path'
      elif output.find('Cannot resolve host') >= 0:
        failureMessage = output+'\n'+error+'\n'+failureMessage
      else:
        (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
        try:
          self.bkClone(urlparse.urlunparse(('http', location, path, parameters, query, fragment)), root, name)
        except RuntimeError, e:
          failureMessage += '\n'+str(e)
        else:
          return
      raise RuntimeError(failureMessage)
    return

  def bkRetrieve(self, url, root, name):
    if not hasattr(self.sourceControl, 'bk'):
      raise RuntimeError('Cannot retrieve a BitKeeper repository since BK was not found')
    self.logPrint('Retrieving '+url+' --> '+os.path.join(root, name)+' via bk', 3, 'install')
    (url, authUrl, wasAuth) = self.getAuthorizedUrl(url)
    try:
      self.testAuthorizedUrl(authUrl)
      self.bkClone(authUrl, root, name)
    except RuntimeError:
      pass
    else:
      return
    return self.bkClone(url, root, name)

  def retrieve(self, url, root = None, canExist = 0, force = 0):
    '''Retrieve the project corresponding to url
    - If root is None, the local root directory is automatically determined. If the project
      was already installed, this root is used. Otherwise a guess is made based upon the url.
    - If canExist is True and the root exists, an update is done instead of a full download.
      The canExist is automatically true if the project has been installed. The retrievalCanExist
      flag can also be used to set this.
    - If force is True, a full download is mandated.
    Providing the root is an easy way to make a copy, for instance when making tarballs.
    '''
    if root is None:
      root = self.getInstallRoot(url)
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    if hasattr(self,scheme+'Retrieve'):
      getattr(self, scheme+'Retrieve')(url, os.path.abspath(root), canExist, force)
    else:
      raise RuntimeError('Invalid transport for retrieval: '+scheme)
    return

  ##############################################
  # This is the old shit
  ##############################################
  def removeRoot(self, root, canExist, force = 0):
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

  def getBKParentURL(self, root):
    '''Return the parent URL for the BK repository at "root"'''
    return self.executeShellCommand('cd '+root+'; bk parent')[21:]

  def bkHeadRevision(self, root):
    '''Return the last change set revision in the repository'''
    return self.executeShellCommand('cd '+root+'; bk changes -and:REV: | head -1')

  def bkfileRetrieve(self, url, root, canExist = 0, force = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via local bk', 3, 'install')
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    return self.bkRetrieve(urlparse.urlunparse(('file', location, path, parameters, query, fragment)), root, canExist, force)

  def sshRetrieve(self, url, root, canExist = 0, force = 0):
    command = 'hg clone '+url+' '+os.path.join(root,os.path.basename(url))
    output  = config.base.Configure.executeShellCommand(command)      
    return root

  def oldRetrieve(self, url, root = None, canExist = 0, force = 0):
    '''Retrieve the project corresponding to url
    - If root is None, the local root directory is automatically determined. If the project
      was already installed, this root is used. Otherwise a guess is made based upon the url.
    - If canExist is True and the root exists, an update is done instead of a full download.
      The canExist is automatically true if the project has been installed. The retrievalCanExist
      flag can also be used to set this.
    - If force is True, a full download is mandated.
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
