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
    installDir = os.path.join(c2htmlDir, self.arch.arch)
    # Configure and Build c2html
    args = ['--prefix='+installDir, '--with-cc='+'"'+self.setCompilers.CC+'"']
    args = ' '.join(args)
    try:
      fd      = file(os.path.join(installDir,'config.args'))
      oldargs = fd.readline()
      fd.close()
    except:
      oldargs = ''
    if not oldargs == args:
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
      fd = file(os.path.join(installDir,'config.args'), 'w')
      fd.write(args)
      fd.close()
      self.framework.actions.addArgument('C2HTML', 'Install', 'Installed c2html into '+installDir)
    self.binDir = os.path.join(installDir, 'bin')
    self.c2html = os.path.join(self.binDir, 'c2html')
    self.addMakeMacro('C2HTML',self.c2html)
    return

  def configure(self):
    '''Determine whether the c2html exist or not'''
    if os.path.exists(os.path.join(self.petscdir.dir, 'BitKeeper'))  or os.path.exists(os.path.join(self.petscdir.dir, 'BK')):
      self.framework.log.write('BitKeeper clone of PETSc, checking for c2html\n')
      self.Install()
    else:
      self.framework.log.write("Not BitKeeper clone of PETSc or cross compiling, don't need c2html\n")
    return


if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(sys.argv[1:])
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
