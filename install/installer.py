#!/usr/bin/env python

import os
import sys


if __name__ == '__main__':
  import setuprc
  setuprc.setupRC(os.path.split(os.path.abspath(os.path.dirname(sys.modules['setuprc'].__file__)))[0])

  import installerclass
  installer   = installerclass.Installer(sys.argv[1:])
  compilerUrl = 'bk://sidl.bkbits.net/Compiler'
  # Must copy list since target is reset by each make below
  for url in installer.argDB.target[:]:
    if url == 'default':
      url = compilerUrl
    if installer.argDB['backup']:
      installer.backup(url)
    else:
      if installer.checkBootstrap():
        booter = Installer(localDict = 1, initDict = installer.argDB)
        # Must build and install BuildSystem
        booter.builder.build(os.path.dirname(booter.builder.getRoot()))
        # Install Compiler and Runtime
        booter.bootstrapInstall('bk://sidl.bkbits.net/Compiler', installer.argDB)
      if installer.checkBootstrap():
        raise RuntimeError('Should not still be bootstraping')
      installer.install(url)
