#!/usr/bin/env python
import install.base
import install.build
import install.retrieval
import nargs

import sys

class Installer(install.base.Base):
  def __init__(self, clArgs = None):
    self.argDB     = nargs.ArgDict('ArgDict', clArgs)
    install.base.Base.__init__(self, self.argDB)
    self.retriever = install.retrieval.Retriever(self.argDB)
    self.builder   = install.build.Builder(self.argDB)
    return

  def install(self, projectUrl):
    self.debugPrint('Installing '+projectUrl, 3, 'install')
    force = self.argDB.has_key('forceInstall') and int(self.argDB['forceInstall'])
    root  = self.retriever.retrieve(projectUrl, force = force);
    self.builder.build(root)
    if self.argDB.has_key('forceInstall'): del self.argDB['forceInstall']
    return

if __name__ == '__main__':
  installer = Installer(sys.argv[1:])
  for url in installer.argDB.target:
    if url == 'default':
      url = 'bk://sidl.bkbits.net/Compiler'
    installer.install(url)
