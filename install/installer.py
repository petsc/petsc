#!/usr/bin/env python
import os
import sys

def runinstaller(opts = []):
  try:
    import install.setuprc
    install.setuprc.setupRC(os.path.dirname(os.path.abspath(os.path.dirname(sys.modules['install.setuprc'].__file__))))
  except ImportError:
    import setuprc
    setuprc.setupRC(os.path.dirname(os.path.abspath(os.path.dirname(sys.modules['setuprc'].__file__))))

  import importer
  import installerclass
  installer   = installerclass.Installer(sys.argv[1:]+opts)
    
  compilerUrl = 'bk://sidl.bkbits.net/Compiler'
  # Must copy list since target is reset by each make below
  for url in installer.argDB.target[:]:
    if url == 'default':
      url = compilerUrl
    if installer.argDB['backup']:
      installer.backup(url)
    else:
      if installer.checkBootstrap():
        booter = installerclass.Installer(argDB = installer.argDB)
        # Must build and install BuildSystem
        booter.builder.build(os.path.dirname(booter.builder.getRoot()))
        # Install Compiler and Runtime
        booter.bootstrapInstall(compilerUrl, installer.argDB)
      if installer.checkBootstrap():
        raise RuntimeError('Should not still be bootstraping')
      if not url == compilerUrl:
        installer.install(url)
  
if __name__ == '__main__':
  runinstaller()
