#!/usr/bin/env python
import importer
import install.base
import install.build
import install.retrieval

import os
import sys

class Installer(install.base.Base):
  def __init__(self, clArgs = None, localDict = 0, initDict = None):
    install.base.Base.__init__(self, self.setupArgDB(clArgs, localDict, initDict))
    self.retriever = install.retrieval.Retriever(self.argDB)
    self.builder   = install.build.Builder(self.argDB)
    self.force     = self.argDB['forceInstall']
    return

  def setupArgDB(self, clArgs, localDict, initDict):
    import nargs
    import RDict

    if localDict:
      parentDirectory = None
    else:
      parentDirectory = os.path.dirname(sys.modules['RDict'].__file__)
    argDB = RDict.RDict(parentDirectory = parentDirectory)

    argDB.setType('backup',            nargs.ArgBool(None, None, 'Backup makes a tar archive of the generated source rather than installing'), forceLocal = 1)
    argDB.setType('forceInstall',      nargs.ArgBool(None, None, 'Forced installation overwrites any existing project'), forceLocal = 1)
    argDB.setType('retrievalCanExist', nargs.ArgBool(None, None, 'Allow a project to exist prior to installation'), forceLocal = 1)
    argDB.setType('urlMappingModules', nargs.Arg(None, None, 'Module name or list of names with a method setupUrlMapping(urlMaps)'), forceLocal = 1)

    argDB['backup']            = 0
    argDB['forceInstall']      = 0
    argDB['retrievalCanExist'] = 0
    argDB['urlMappingModules'] = ''

    argDB.insertArgs(clArgs)
    argDB.insertArgs(initDict)
    return argDB

  def install(self, url):
    self.debugPrint('Installing '+url, 3, 'install')
    root = self.retriever.retrieve(url, force = self.force);
    self.builder.build(root)
    return

  def bootstrapInstall(self, url, argDB):
    self.debugPrint('Installing '+url+' from bootstrap', 3, 'install')
    root = self.retriever.retrieve(url, force = self.force);
    # This is for purging the sidl after the build
    self.argDB['fileset'] = 'sidl'
    self.builder.build(root, target = ['default', 'purge'], setupTarget = 'setupBootstrap')
    # Fixup install arguments
    argDB['installedprojects']  = self.argDB['installedprojects']
    argDB['installedLanguages'] = self.argDB['installedLanguages']
    return

  def backup(self, url):
    '''This forces a fresh copy of the project instead of using the one in the database'''
    import shutil

    self.debugPrint('Backing up '+url, 3, 'install')
    root = self.retriever.retrieve(url, self.getInstallRoot(url, isBackup = 1), force = self.force);
    self.builder.build(root, 'sidl', ignoreDependencies = 1)
    output = self.executeShellCommand('tar -czf '+self.getRepositoryName(self.getMappedUrl(url))+'.tgz -C '+os.path.dirname(root)+' '+os.path.basename(root))
    shutil.rmtree(os.path.dirname(root))
    return

