#!/usr/bin/env python
from __future__ import generators
import config.base
import os
import re
import PETSc.package
    
class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/c2html.tar.gz']

  def Install(self):
    c2htmlDir = self.getDir()
    # Configure and Build c2html
    args = ['--prefix='+self.installDir, '--with-cc='+'"'+self.setCompilers.CC+'"']
    args = ' '.join(args)
    fd = file(os.path.join(self.installDir,'c2html'), 'w')
    fd.write(args)
    fd.close()
    if not os.path.isfile(os.path.join(self.confDir,'c2html')) or not (self.getChecksum(os.path.join(self.confDir,'c2html')) == self.getChecksum(os.path.join(c2htmlDir,'c2html'))):
      try:
        output  = config.base.Configure.executeShellCommand('cd '+c2htmlDir+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        if self.framework.argDB['with-batch']:
          self.logPrintBox('Batch build that could not generate c2html, you will not be able to generate document')
          return
        raise RuntimeError('Error running configure on C2html: '+str(e))
      try:
        output  = config.base.Configure.executeShellCommand('cd '+c2htmlDir+';make; make install; make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        if self.framework.argDB['with-batch']:
          self.logPrintBox('Batch build that could not generate c2html, you will not be able to generate document')
        raise RuntimeError('Error running make; make install on C2html: '+str(e))
        output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(c2htmlDir,'c2html')+' '+self.confDir+'/c2html', timeout=5, log = self.framework.log)[0]      
      self.framework.actions.addArgument('C2HTML', 'Install', 'Installed c2html into '+self.installDir)
    self.binDir = os.path.join(self.installDir, 'bin')
    self.c2html = os.path.join(self.binDir, 'c2html')
    self.addMakeMacro('C2HTML',self.c2html)
    return

  def configure(self):
    '''Determine whether the c2html exist or not'''
    if self.petscdir.isClone:
      self.framework.logPrint('PETSc clone, checking for c2html\n')
      self.installDir  = os.path.join(self.petscdir.dir,self.arch.arch)
      self.confDir     = os.path.join(self.petscdir.dir,self.arch.arch,'conf')
      self.Install()
    else:
      self.framework.logPrint("Not a clone of PETSc, don't need c2html\n")
    return


if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(sys.argv[1:])
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
