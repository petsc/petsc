from __future__ import absolute_import
import logger

import os
try:
  from urllib import urlretrieve
except ImportError:
  from urllib.request import urlretrieve
try:
  import urlparse as urlparse_local # novermin
except ImportError:
  from urllib import parse as urlparse_local # novermin
import config.base
import socket
import shutil

# Fix parsing for nonstandard schemes
urlparse_local.uses_netloc.extend(['bk', 'ssh', 'svn'])

class Retriever(logger.Logger):
  def __init__(self, sourceControl, clArgs = None, argDB = None):
    logger.Logger.__init__(self, clArgs, argDB)
    self.sourceControl = sourceControl
    self.stamp = None
    return

  def isDirectoryGitRepo(self, directory):
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

  def genericRetrieve(self, url, root, package, submodules):
    '''Fetch package from version control repository or tarfile indicated by URL and extract it into root'''

    parsed = urlparse_local.urlparse(url)
    if parsed[0] == 'dir':
      f = self.dirRetrieve
    elif parsed[0] == 'link':
      f = self.linkRetrieve
    elif parsed[0] == 'git':
      f = self.gitRetrieve
    elif parsed[0] == 'ssh'   and parsed[2].endswith('.git'):
      f = self.gitRetrieve
    elif parsed[0] == 'https' and parsed[2].endswith('.git'):
      f = self.gitRetrieve
    elif parsed[0] == 'hg':
      f = self.hgRetrieve
    elif parsed[0] == 'ssh' and parsed[1].startswith('hg@'):
      f = self.hgRetrieve
    elif os.path.isdir(url):
      if self.isDirectoryGitRepo(url):
        f = self.gitRetrieve
      else:
        f = self.dirRetrieve
    else:
      f = self.tarballRetrieve
    return f(url, root, package, submodules)

  def dirRetrieve(self, url, root, package, submodules):
    self.logPrint('Retrieving %s as directory' % url, 3, 'install')
    d = self.removePrefix(url, 'dir://')
    if not os.path.isdir(d): raise RuntimeError('URL %s is not a directory' % url)

    t = os.path.join(root,os.path.basename(d))
    self.removeTarget(t)
    shutil.copytree(d,t)

  def linkRetrieve(self, url, root, package, submodules):
    self.logPrint('Retrieving %s as link' % url, 3, 'install')
    d = self.removePrefix(url, 'link://')
    if not os.path.isdir(d): raise RuntimeError('URL %s is not pointing to a directory' % url)

    t = os.path.join(root,os.path.basename(d))
    self.removeTarget(t)
    os.symlink(os.path.abspath(d),t)

  def gitRetrieve(self, url, root, package, submodules):
    self.logPrint('Retrieving %s as git repo' % url, 3, 'install')
    if not hasattr(self.sourceControl, 'git'):
      raise RuntimeError('self.sourceControl.git not set')
    d = self.removePrefix(url, 'git://')
    if os.path.isdir(d) and not self.isDirectoryGitRepo(d):
      raise RuntimeError('URL %s is a directory but not a git repository' % url)

    newgitrepo = os.path.join(root,'git.'+package)
    self.removeTarget(newgitrepo)

    try:
      submodopt =''
      for itm in submodules:
        submodopt += ' --recurse-submodules='+itm
      config.base.Configure.executeShellCommand('%s clone %s %s %s' % (self.sourceControl.git, submodopt, d, newgitrepo), log = self.log, timeout = 120.0)
    except  RuntimeError as e:
      self.logPrint('ERROR: '+str(e))
      err = str(e)
      failureMessage = self.getDownloadFailureMessage(package, url)
      raise RuntimeError('Unable to clone '+package+'\n'+err+failureMessage)

  def hgRetrieve(self, url, root, package, submodules):
    self.logPrint('Retrieving %s as hg repo' % url, 3, 'install')
    if not hasattr(self.sourceControl, 'hg'):
      raise RuntimeError('self.sourceControl.hg not set')
    d = self.removePrefix(url, 'hg://')

    newgitrepo = os.path.join(root,'hg.'+package)
    self.removeTarget(newgitrepo)
    try:
      config.base.Configure.executeShellCommand('%s clone %s %s' % (self.sourceControl.hg, d, newgitrepo), log = self.log, timeout = 120.0)
    except  RuntimeError as e:
      self.logPrint('ERROR: '+str(e))
      err = str(e)
      failureMessage = self.getDownloadFailureMessage(package, url)
      raise RuntimeError('Unable to clone '+package+'\n'+err+failureMessage)

  def tarballRetrieve(self, url, root, package, submodules):
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
        failureMessage = self.getDownloadFailureMessage(package, url, filename)
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
''' % (package.upper(), url, filename, package, filename)
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
