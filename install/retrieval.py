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
    print 'PATH'+path
    path = path[1:]
    print 'PATH'+path
    if self.base:
      path = os.path.join(self.base, path)
    print 'PATH'+path
    return os.path.abspath(path)

  def removeRoot(self,root,canExist,force = 0):
    'Returns 1 if removes root'
    if os.path.exists(root):
      if canExist:
        if force:
          output = self.executeShellCommand('rm -rf '+root)
          return 1
        else:
          return 0
      else:
        raise RuntimeError('Root directory '+root+' already exists')
    return 1
    
  def genericRetrieve(self, url, root, canExist = 0, force = 0):
    if not self.removeRoot(root,canExist,force): return root
    localFile = root+'.tgz'
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

  def fileRetrieve(self, url, root, canExist = 0, force = 0):
    self.debugPrint('Retrieving '+url+' --> '+root+' via cp', 3, 'install')
    return self.genericRetrieve(url, root, canExist, force)

  def bkRetrieve(self, url, root, canExist = 0, force = 0):
    if self.checkBootstrap():
      (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
      path   = os.path.join('/pub', 'petsc', location.split('.')[0], path[1:]+'.tgz')
      newUrl = urlparse.urlunparse(('ftp', 'ftp.mcs.anl.gov', path, parameters, query, fragment))
      return self.retrieve(newUrl, root, canExist, force)

    self.debugPrint('Retrieving '+url+' --> '+root+' via bk', 3, 'install')
    if os.path.exists(root):
      output = self.executeShellCommand('cd '+root+'; bk pull')
    else:
      (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
      if not location:
        url = urlparse.urlunparse(('','', path, parameters, query, fragment))
      elif location.find('@') < 0:
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
    if not self.removeRoot(root,canExist,force): return root
    command = 'ssh '+location+' "tar -C '+dir+' -zc '+project+'" | tar -C '+root+' -zx'
    output  = self.executeShellCommand(command)
    return root

  def retrieve(self, url, root = None, canExist = 0, force = 0):
    remapurls = {'bk://sidl.bkbits.net/Compiler' : 'bk:///home/web/Compiler',
                 'bk://sidl.bkbits.net/Runtime' : 'bk:///home/web/Runtime',
                 'ftp://ftp.mcs.anl.gov/pub/petsc/sidl/Compiler.tgz' : 'file:///home/web/Compiler.tgz',
                 'ftp://ftp.mcs.anl.gov/pub/petsc/sidl/Runtime.tgz' : 'file:///home/web/Runtime.tgz',
                 'ftp://ftp.mcs.anl.gov/pub/petsc/sidl/ply.tgz' : 'file:///home/web/ply.tgz',
                 'ftp://ftp.mcs.anl.gov/pub/petsc/home/web/Compiler.tgz' : 'file:///home/web/Compiler.tgz',
                 'ftp://ftp.mcs.anl.gov/pub/petsc/home/web/Runtime.tgz' : 'file:///home/web/Runtime.tgz',
                 'ftp://ftp.mcs.anl.gov/pub/petsc/home/web/ply.tgz' : 'file:///home/web/ply.tgz'}
    remaproot = {'bk://sidl.bkbits.net/Compiler' : 'Compiler',
                 'bk://sidl.bkbits.net/Runtime' : 'Runtime',
                 'ftp://ftp.mcs.anl.gov/pub/petsc/sidl/Compiler.tgz' : 'Compiler',
                 'ftp://ftp.mcs.anl.gov/pub/petsc/sidl/Runtime.tgz' : 'Runtime',
                 'ftp://ftp.mcs.anl.gov/pub/petsc/sidl/ply.tgz' : 'ply-dev',
                 'ftp://ftp.mcs.anl.gov/pub/petsc/home/web/Compiler.tgz' : 'Compiler',
                 'ftp://ftp.mcs.anl.gov/pub/petsc/home/web/Runtime.tgz' : 'Runtime',
                 'ftp://ftp.mcs.anl.gov/pub/petsc/home/web/ply.tgz' : 'ply-dev'}
    print 'URL in '+url
    if remaproot.has_key(url):
      root = remaproot[url]
      print 'Remapped root '+root
    if remapurls.has_key(url):
      url = remapurls[url]
      print 'Remapped url '+url

    project = self.getInstalledProject(url)
    if not project is None:
      root     = project.getRoot()
      canExist = 1
    print 'url'+url
    if root is None:
      root = self.getInstallRoot(url)
    print 'root'+root
    print 'url'+url
    (scheme, location, path, parameters, query, fragment) = urlparse.urlparse(url)
    print 'location'+location
    print 'path'+path
    try:
      if self.argDB['retrievalCanExist']:
        canExist = 1

      return getattr(self, scheme+'Retrieve')(url, os.path.abspath(root), canExist, force)
    except AttributeError:
      raise RuntimeError('Invalid transport for retrieval: '+scheme)
