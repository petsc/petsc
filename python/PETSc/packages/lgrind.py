#!/usr/bin/env python
from __future__ import generators
import config.base
import os
import re
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['bk://petsc.bkbits.net/lgrind-dev','ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/lgrind-dev.tar.gz']
    self.required     = 1
    return

  def Install(self):
    lgrindDir = self.getDir()
    # Get the LGRIND directories
    installDir = os.path.join(lgrindDir, self.arch.arch)
    if os.path.isfile(os.path.join(installDir,'lgrind')) or os.path.isfile(os.path.join(installDir,'lgrind.exe')):
      self.framework.log.write('Found Lgrind executable; skipping compile\n')
      lgrindexe = os.path.join(installDir,'source','lgrind')
      if os.path.exists(lgrindexe+'.exe'):
        lgrindexe = lgrindexe+'.exe'
        lgrind    = 'lgrind.exe'
      else: lgrind = 'lgrind'
    else:
      self.framework.log.write('Did not find Lgrind executable; compiling lgrind\n')
      try:
        self.framework.pushLanguage('C')
        output = config.base.Configure.executeShellCommand('cd '+os.path.join(lgrindDir,'source')+'; make clean; make CC=\''+self.framework.getCompiler()+'\'',timeout=2500,log = self.framework.log)[0]
        self.framework.popLanguage()
      except RuntimeError, e:
        self.framework.popLanguage()
        if self.framework.argDB['with-batch']:
          self.logPrintBox('Batch build that could not generate lgrind, you will not be able to build documentation')
          return
        raise RuntimeError('Error running make on lgrind: '+str(e))
      try:
        lgrindexe = os.path.join(lgrindDir,'source','lgrind')
        if os.path.exists(lgrindexe+'.exe'):
          lgrindexe = lgrindexe+'.exe'
          lgrind    = 'lgrind.exe'
        else: lgrind = 'lgrind'
        output  = config.base.Configure.executeShellCommand('mv '+lgrindexe+' '+installDir, timeout=25, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error copying lgrind executable: '+str(e))
    output = config.base.Configure.executeShellCommand('cd '+os.path.join(lgrindDir,'source')+'; make clean',timeout=25, log = self.framework.log)[0]
    self.framework.actions.addArgument('lgrind', 'Install', 'Installed lgrind into '+installDir)
    self.lgrind = lgrindexe
    self.addMakeMacro('LGRIND',os.path.join(installDir,lgrind))
    self.addMakeMacro('LGRIND_DIR',lgrindDir)
    return

  def configure(self):
    '''Determine whether the Lgrind exist or not'''
    if (os.path.exists(os.path.join(self.petscdir.dir, 'BitKeeper')) or os.path.exists(os.path.join(self.petscdir.dir, 'BK'))) and self.framework.argDB['with-lgrind']:
      self.framework.log.write('BitKeeper clone of PETSc, checking for Lgrind\n')
      self.Install()
    else:
      self.framework.log.write("Not BitKeeper clone of PETSc don't need Lgrind\n")
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(sys.argv[1:])
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
