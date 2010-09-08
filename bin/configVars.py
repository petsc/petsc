#!/usr/bin/env python
import os, sys

sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config'))
sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))

import script

class ConfigReader(script.Script):
  def __init__(self):
    import RDict
    import os

    argDB = RDict.RDict(None, None, 0, 0)
    argDB.saveFilename = os.path.join(os.environ['PETSC_DIR'], os.environ['PETSC_ARCH'], 'conf', 'RDict.db')
    argDB.load()
    script.Script.__init__(self, argDB = argDB)
    return

  def run(self):
    self.setup()
    framework = self.loadConfigure()
    mpi = framework.require('config.packages.MPI', None)
    print mpi.include, mpi.lib
    arch = framework.require('PETSc.utilities.arch', None)
    print arch.arch
    print 'Configure is cached:',('configureCache' in self.argDB)
    for k in framework.argDB.keys():
      if k.startswith('known'):
        print k,framework.argDB[k]
    return

if __name__ == '__main__':
  print 'Starting'
  ConfigReader().run()
  print 'Ending'
