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
    elif installer.argDB['remove']:
      installer.remove(url)
    else:
      if installer.checkBootstrap():
        booter = installerclass.Installer(argDB = installer.argDB)
        # Must build and install BuildSystem
        booter.builder.build(booter.retriever.retrieve('bk://sidl.bkbits.net/BuildSystem'))
        # Install Compiler and Runtime
        booter.bootstrapInstall(compilerUrl, installer.argDB)
      if installer.checkBootstrap():
        raise RuntimeError('ERROR: Bootstrap mode still active. This probably means that the Runtime or Compiler failed to install correctly.')
      if not url == compilerUrl:
        if installer.argDB['profile']:
          import profile
          profile.run('installer.install(url)')
        else:
          installer.install(url)
  
if __name__ == '__main__':
  runinstaller()
