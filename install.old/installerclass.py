import install.urlMapping

import os
import sys
import nargs

class Installer(install.urlMapping.UrlMapping):
  def __init__(self, clArgs = None, argDB = None, stamp = None):
    import install.build
    import install.retrieval

    install.urlMapping.UrlMapping.__init__(self, clArgs, argDB, stamp)
    self.retriever = install.retrieval.Retriever(stamp)
    self.builder   = install.build.Builder(stamp)
    self.force     = self.argDB['forceInstall']
    self.checkPython()
    self.checkNumeric()
    return

  def setupArgDB(self, argDB, clArgs):
    '''Setup argument types, using the database created by base.Base'''

    argDB.setType('backup',            nargs.ArgBool(None, 0, 'Backup makes a tar archive of the generated source rather than installing', isTemporary = 1), forceLocal = 1)
    argDB.setType('remove',            nargs.ArgBool(None, 0, 'Remove the indicated project rather than installing', isTemporary = 1), forceLocal = 1)
    argDB.setType('forceInstall',      nargs.ArgBool(None, 0, 'Forced installation overwrites any existing project', isTemporary = 1), forceLocal = 1)
    argDB.setType('retrievalCanExist', nargs.ArgBool(None, 0, 'Allow a project to exist prior to installation', isTemporary = 1), forceLocal = 1)
    argDB.setType('urlMappingModules', nargs.Arg(None, '', 'Module name or list of names with a method setupUrlMapping(urlMaps)'), forceLocal = 1)
    argDB.setType('profile',           nargs.ArgBool(None, 0, 'Profile the installation', isTemporary = 1), forceLocal = 1)
    install.urlMapping.UrlMapping.setupArgDB(self, argDB, clArgs)
    return argDB

  def checkPython(self):
    import sys

    if not hasattr(sys, 'version_info') or float(sys.version_info[0]) != 2 or float(sys.version_info[1]) < 4:
      raise RuntimeError('BuildSystem requires Python2 version 2.4 or higher. Get Python at http://www.python.org')
    return

  def checkNumeric(self):
    import distutils.sysconfig

    try:
      import Numeric
    except ImportError, e:
      raise RuntimeError('BuildSystem requires Numeric Python. Select numpy version 23.0 at http://sourceforge.net/projects/numpy')
    header = os.path.join(distutils.sysconfig.get_python_inc(), 'Numeric', 'arrayobject.h')
    if not os.path.exists(header):
      raise RuntimeError('The include files from the Numeric are misplaced: Cannot find '+header)
    return

  def install(self, url):
    self.debugPrint('Installing '+url, 3, 'install')
    root = self.retriever.retrieve(url, force = self.force);
    self.builder.build(root)
    return

  def bootstrapInstall(self, url, argDB):
    self.debugPrint('Installing '+url+' from bootstrap', 3, 'install')
    root = self.retriever.retrieve(url, force = self.force);
    self.builder.build(root)
    return

  def backup(self, url):
    '''This forces a fresh copy of the project instead of using the one in the database, and returns the tarball created'''
    import shutil

    self.debugPrint('Backing up '+url, 3, 'install')
    root = self.retriever.retrieve(url, self.getInstallRoot(url, isBackup = 1), force = self.force);
    # Must save checkpoint in the project root
    self.argDB.saveFilename = os.path.join(root, 'RDict.db')
    self.builder.build(root, ['sidlCheckpoint', 'deactivate'], ignoreDependencies = 1)
    self.argDB.save(force = 1)
    # Remove frozen makefile
    makefile = os.path.join(root, 'make.pyc')
    if os.path.isfile(makefile):
      print 'Removing old makefile '+makefile
      os.remove(makefile)
    # Tar up project
    tarball = self.getRepositoryName(self.getMappedUrl(url))+'.tgz'
    output  = self.executeShellCommand('tar -czf '+tarball+' -C '+os.path.dirname(root)+' '+os.path.basename(root))
    # Reset this since we are removing the directory
    self.argDB.saveFilename = 'RDict.db'
    shutil.rmtree(os.path.dirname(root))
    return tarball

  def removeProject(self, url):
    '''Remove a project'''
    import shutil

    proj = self.getInstalledProject(url)
    if proj is None:
      self.debugPrint(url+' is not installed', 1, 'install')
      return
    self.debugPrint('Removing '+url, 3, 'install')
    # Uninstall project
    print self.builder.build(proj.getRoot(), ['activate', 'uninstall'], ignoreDependencies = 1)
    # Remove files
    print shutil.rmtree(proj.getRoot())
    return

  def remove(self, url):
    '''Remove a project and all project which depend on it'''
    import build.buildGraph

    self.debugPrint('Removing '+url+' and dependents', 3, 'install')
    proj = self.getInstalledProject(url)
    if proj is None:
      self.debugPrint(url+' is not installed', 1, 'install')
      return
    for p in build.buildGraph.BuildGraph.depthFirstVisit(self.argDB['projectDependenceGraph'], proj, outEdges = 0):
      self.removeProject(p.getUrl())
    return
