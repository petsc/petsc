#!/usr/bin/env python
import install.base
import install.build
import install.retrieval
import nargs

import sys

class Installer(install.base.Base):
  def __init__(self, clArgs = None):
    self.argDB = nargs.ArgDict('ArgDict')
    self.argDB.setLocalType('backup',       nargs.ArgBool('Backup makes a tar archive of the generated source rather than installing'))
    self.argDB.setLocalType('forceInstall', nargs.ArgBool('Forced installation overwrites any existing project'))
    self.argDB.insertArgList(clArgs)

    install.base.Base.__init__(self, self.argDB)
    self.retriever = install.retrieval.Retriever(self.argDB)
    self.builder   = install.build.Builder(self.argDB)
    self.force     = self.argDB.has_key('forceInstall') and self.argDB['forceInstall']
    return

  def install(self, projectUrl):
    self.debugPrint('Installing '+projectUrl, 3, 'install')
    root = self.retriever.retrieve(projectUrl, force = self.force);
    self.builder.build(root)
    return

  def backup(self, projectUrl):
    self.debugPrint('Backing up '+projectUrl, 3, 'install')
    root = self.retriever.retrieve(projectUrl, force = self.force);
    self.builder.build(root, 'sidl')
    return

if __name__ == '__main__':
  installer = Installer(sys.argv[1:])
  for url in installer.argDB.target:
    if url == 'default':
      url = 'bk://sidl.bkbits.net/Compiler'
    if installer.argDB.has_key('backup') and installer.argDB['backup']:
      installer.backup(url)
    else:
      installer.install(url)
