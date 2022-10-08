from __future__ import absolute_import
import logger

import os
from urllib.request import urlretrieve
from urllib import parse as urlparse_local
import config.base
import socket
import shutil

# Fix parsing for nonstandard schemes
urlparse_local.uses_netloc.extend(['bk', 'ssh', 'svn'])

class Retriever(logger.Logger):
  def __init__(self, sourceControl, clArgs = None, argDB = None):
    logger.Logger.__init__(self, clArgs, argDB)
    self.sourceControl = sourceControl
    self.gitsubmodules = []
    self.gitprereq = 1
    self.git_urls = []
    self.hg_urls = []
    self.dir_urls = []
    self.link_urls = []
    self.tarball_urls = []
    self.stamp = None
    return

  def isGitURL(self, url):
    parsed = urlparse_local.urlparse(url)
    if (parsed[0] == 'git') or (parsed[0] == 'ssh' and parsed[2].endswith('.git')) or (parsed[0] == 'https' and parsed[2].endswith('.git')):
      return True
    elif os.path.isdir(url) and self.isDirectoryGitRepo(url):
      return True
    return False

  def setupURLs(self,packagename,urls,gitsubmodules,gitprereq):
    self.packagename = packagename
    self.gitsubmodules = gitsubmodules
    self.gitprereq = gitprereq
    for url in urls:
      parsed = urlparse_local.urlparse(url)
      if self.isGitURL(url):
        self.git_urls.append(self.removePrefix(url,'git://'))
      elif parsed[0] == 'hg'or (parsed[0] == 'ssh' and parsed[1].startswith('hg@')):
        self.hg_urls.append(self.removePrefix(url,'hg://'))
      elif parsed[0] == 'dir' or os.path.isdir(url):
        self.dir_urls.append(self.removePrefix(url,'dir://'))
      elif parsed[0] == 'link':
        self.link_urls.append(self.removePrefix(url,'link://'))
      else:
        # check for ftp.mcs.anl.gov - and use https://,www.mcs.anl.gov,ftp://
        if url.find('ftp.mcs.anl.gov') != -1:
          https_url = url.replace('http://','https://').replace('ftp://','http://')
          self.tarball_urls.extend([https_url,https_url.replace('ftp.mcs.anl.gov/pub/petsc/','www.mcs.anl.gov/petsc/mirror/'),https_url.replace('https://','ftp://')])
        else:
          self.tarball_urls.extend([url])

  def isDirectoryGitRepo(self, directory):
    if not hasattr(self.sourceControl, 'git'):
      self.logPrint('git not found in self.sourceControl - cannot evaluate isDirectoryGitRepo(): '+directory)
      return False
    from config.base import Configure
    for loc in ['.git','']:
      cmd = '%s rev-parse --resolve-git-dir  %s'  % (self.sourceControl.git, os.path.join(directory,loc))
      (output, error, ret) = Configure.executeShellCommand(cmd, checkCommand = Configure.passCheckCommand, log = self.log)
      if not ret:
        return True
    return False

  @staticmethod
  def removeTarget(t):
    if os.path.islink(t) or os.path.isfile(t):
      os.unlink(t) # same as os.remove(t)
    elif os.path.isdir(t):
      shutil.rmtree(t)

  @staticmethod
  def getDownloadFailureMessage(package, url, filename=None):
    slashFilename = '/'+filename if filename else ''
    return '''\
Unable to download package %s from: %s
* If URL specified manually - perhaps there is a typo?
* If your network is disconnected - please reconnect and rerun ./configure
* Or perhaps you have a firewall blocking the download
* You can run with --with-packages-download-dir=/adirectory and ./configure will instruct you what packages to download manually
* or you can download the above URL manually, to /yourselectedlocation%s
  and use the configure option:
  --download-%s=/yourselectedlocation%s
    ''' % (package.upper(), url, slashFilename, package, slashFilename)

  @staticmethod
  def removePrefix(url,prefix):
    '''Replacement for str.removeprefix() supported only since Python 3.9'''
    if url.startswith(prefix):
      return url[len(prefix):]
    return url

  def generateURLs(self):
    if hasattr(self.sourceControl, 'git') and self.gitprereq:
      for url in self.git_urls:
        yield('git',url)
    else:
      self.logPrint('Git not found or gitprereq check failed! skipping giturls: '+str(self.git_urls)+'\n')
    if hasattr(self.sourceControl, 'hg'):
      for url in self.hg_urls:
        yield('hg',url)
    else:
      self.logPrint('Hg not found - skipping hgurls: '+str(self.hg_urls)+'\n')
    for url in self.dir_urls:
      yield('dir',url)
    for url in self.link_urls:
      yield('link',url)
    for url in self.tarball_urls:
      yield('tarball',url)

  def genericRetrieve(self,proto,url,root):
    '''Fetch package from version control repository or tarfile indicated by URL and extract it into root'''
    if proto == 'git':
      return self.gitRetrieve(url,root)
    elif proto == 'hg':
      return self.hgRetrieve(url,root)
    elif proto == 'dir':
      return self.dirRetrieve(url,root)
    elif proto == 'link':
      self.linkRetrieve(url,root)
    elif proto == 'tarball':
      self.tarballRetrieve(url,root)

  def dirRetrieve(self, url, root):
    self.logPrint('Retrieving %s as directory' % url, 3, 'install')
    if not os.path.isdir(url): raise RuntimeError('URL %s is not a directory' % url)

    t = os.path.join(root,os.path.basename(url))
    self.removeTarget(t)
    shutil.copytree(url,t)

  def linkRetrieve(self, url, root):
    self.logPrint('Retrieving %s as link' % url, 3, 'install')
    if not os.path.isdir(url): raise RuntimeError('URL %s is not pointing to a directory' % url)

    t = os.path.join(root,os.path.basename(url))
    self.removeTarget(t)
    os.symlink(os.path.abspath(url),t)

  def gitRetrieve(self, url, root):
    self.logPrint('Retrieving %s as git repo' % url, 3, 'install')
    if not hasattr(self.sourceControl, 'git'):
      raise RuntimeError('self.sourceControl.git not set')
    if os.path.isdir(url) and not self.isDirectoryGitRepo(url):
      raise RuntimeError('URL %s is a directory but not a git repository' % url)

    newgitrepo = os.path.join(root,'git.'+self.packagename)
    self.removeTarget(newgitrepo)

    try:
      submodopt =''
      for itm in self.gitsubmodules:
        submodopt += ' --recurse-submodules='+itm
      config.base.Configure.executeShellCommand('%s clone %s %s %s' % (self.sourceControl.git, submodopt, url, newgitrepo), log = self.log, timeout = 120.0)
    except  RuntimeError as e:
      self.logPrint('ERROR: '+str(e))
      err = str(e)
      failureMessage = self.getDownloadFailureMessage(self.packagename, url)
      raise RuntimeError('Unable to clone '+self.packagename+'\n'+err+failureMessage)

  def hgRetrieve(self, url, root):
    self.logPrint('Retrieving %s as hg repo' % url, 3, 'install')
    if not hasattr(self.sourceControl, 'hg'):
      raise RuntimeError('self.sourceControl.hg not set')

    newgitrepo = os.path.join(root,'hg.'+self.packagename)
    self.removeTarget(newgitrepo)
    try:
      config.base.Configure.executeShellCommand('%s clone %s %s' % (self.sourceControl.hg, url, newgitrepo), log = self.log, timeout = 120.0)
    except  RuntimeError as e:
      self.logPrint('ERROR: '+str(e))
      err = str(e)
      failureMessage = self.getDownloadFailureMessage(self.packagename, url)
      raise RuntimeError('Unable to clone '+self.packagename+'\n'+err+failureMessage)

  def tarballRetrieve(self, url, root):
    parsed = urlparse_local.urlparse(url)
    filename = os.path.basename(parsed[2])
    localFile = os.path.join(root,'_d_'+filename)
    self.logPrint('Retrieving %s as tarball to %s' % (url,localFile) , 3, 'install')
    ext =  os.path.splitext(localFile)[1]
    if ext not in ['.bz2','.tbz','.gz','.tgz','.zip','.ZIP']:
      raise RuntimeError('Unknown compression type in URL: '+ url)

    self.removeTarget(localFile)

    if parsed[0] == 'file' and not parsed[1]:
      url = parsed[2]
    if os.path.exists(url):
      if not os.path.isfile(url):
        raise RuntimeError('Local path exists but is not a regular file: '+ url)
      # copy local file
      shutil.copyfile(url, localFile)
    else:
      # fetch remote file
      try:
        sav_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(30)
        urlretrieve(url, localFile)
        socket.setdefaulttimeout(sav_timeout)
      except Exception as e:
        socket.setdefaulttimeout(sav_timeout)
        failureMessage = self.getDownloadFailureMessage(self.packagename, url, filename)
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
* You can run with --with-packages-download-dir=/adirectory and ./configure will instruct you what packages to download manually
* or you can download the above URL manually, to /yourselectedlocation/%s
  and use the configure option:
  --download-%s=/yourselectedlocation/%s
''' % (self.packagename.upper(), url, filename, self.packagename, filename)
      import tarfile
      try:
        tf  = tarfile.open(os.path.join(root, localFile))
      except tarfile.ReadError as e:
        raise RuntimeError(str(e)+'\n'+failureMessage)
      if not tf: raise RuntimeError(failureMessage)
      #git puts 'pax_global_header' as the first entry and some tar utils process this as a file
      firstname = tf.getnames()[0]
      if firstname == 'pax_global_header':
        firstmember = tf.getmembers()[1]
      else:
        firstmember = tf.getmembers()[0]
      # some tarfiles list packagename/ but some list packagename/filename in the first entry
      if firstmember.isdir():
        dirname = firstmember.name
      else:
        dirname = os.path.dirname(firstmember.name)
      tf.extractall(root)
      tf.close()

    # fix file permissions for the untared tarballs.
    try:
      # check if 'dirname' is set'
      if dirname:
        config.base.Configure.executeShellCommand('cd '+root+'; chmod -R a+r '+dirname+';find  '+dirname + ' -type d -name "*" -exec chmod a+rx {} \;', log = self.log)
      else:
        self.logPrintBox('WARNING: Could not determine dirname extracted by '+localFile+' to fix file permissions')
    except RuntimeError as e:
      raise RuntimeError('Error changing permissions for '+dirname+' obtained from '+localFile+ ' : '+str(e))
    os.unlink(localFile)
