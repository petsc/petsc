#!/usr/bin/env python
from __future__ import generators
import config.base
import os
import re
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['bk://petsc.bkbits.net/lgrind-dev']
    return

  def Install(self):
    lgrindDir = self.getDir()
    # Get the LGRIND directories
    installDir = os.path.join(lgrindDir, self.arch.arch)
    if os.path.isfile(os.path.join(installDir,'lgrind')) or os.path.isfile(os.path.join(installDir,'lgrind.exe')):
      self.framework.log.write('Found Lgrind executable; skipping compile\n')
      lgrindexe = os.path.join(installDir,'source','lgrind')
      if os.path.exists(lgrindexe+'.exe'): lgrindexe = lgrindexe+'.exe'
    else:
      self.framework.log.write('Did not find Lgrind executable; compiling lgrind\n')
      try:
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(lgrindDir,'source')+';make', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on lgrind: '+str(e))
      try:
        lgrindexe = os.path.join(lgrindDir,'source','lgrind')
        if os.path.exists(lgrindexe+'.exe'): lgrindexe = lgrindexe+'.exe'
        output  = config.base.Configure.executeShellCommand('cp '+lgrindexe+' '+installDir, timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error copying lgrind executable: '+str(e))
    self.framework.actions.addArgument('lgrind', 'Install', 'Installed lgrind into '+installDir)
    self.lgrind = lgrindexe
    self.addMakeMacro('LGRIND',os.path.join(self.lgrind,lgrindexe))
    self.addMakeMacro('LGRIND_DIR',installDir)
    return

  def configure(self):
    '''Determine whether the Lgrind exist or not'''
    if os.path.exists(os.path.join(self.framework.argDB['PETSC_DIR'], 'BitKeeper')):
      self.framework.log.write('BitKeeper clone of PETSc, checking for Lgrind\n')
      self.Install()
    else:
      self.framework.log.write("Not BitKeeper clone of PETSc, don't need Lgrind\n")
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(sys.argv[1:])
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
