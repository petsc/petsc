#!/usr/bin/env python
from __future__ import generators
import config.base
import os
import re
import PETSc.package
    
class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download         = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/c2html.tar.gz']
    self.complex          = 1
    self.double           = 0;
    self.requires32bitint = 0;
    
  def Install(self):

    if self.framework.argDB['with-batch']:
       args = ['--prefix='+self.installDir]
    else:
       args = ['--prefix='+self.installDir, '--with-cc='+'"'+self.setCompilers.CC+'"']          
    args = ' '.join(args)
    fd = file(os.path.join(self.packageDir,'c2html.args'), 'w')
    fd.write(args)
    fd.close()
    if self.installNeeded('c2html.args'):
      try:
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';./configure '+args, timeout=900, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running configure on c2html: '+str(e))
      try:
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';make; make install; make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make; make install on c2html: '+str(e))
      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(self.packageDir,'c2html.args')+' '+self.confDir+'/c2html', timeout=5, log = self.framework.log)[0]      
      self.framework.actions.addArgument('C2HTML', 'Install', 'Installed c2html into '+self.installDir)
    self.binDir = os.path.join(self.installDir, 'bin')
    self.c2html = os.path.join(self.binDir, 'c2html')
    self.addMakeMacro('C2HTML',self.c2html)
    return self.installDir

  def alternateConfigureLibrary(self):
    self.checkDownload(1)
    
  def configure(self):
    '''Determine whether the c2html exist or not'''
    if self.petscdir.isClone:
      self.framework.logPrint('PETSc clone, checking for c2html\n')
      self.getExecutable('c2html', getFullPath = 1)
      
      if hasattr(self, 'c2html'):
        self.addMakeMacro('C2HTML ', self.c2html)
        self.framework.logPrint('Found c2html, will not install c2html')
      else:
        self.framework.logPrint('Installing c2html')
        if not self.framework.argDB.has_key('download-c2html') or not self.framework.argDB['download-c2html']: self.framework.argDB['download-c2html'] = 1
        PETSc.package.Package.configure(self)
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
