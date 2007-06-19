#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/Chaco-2.2.tar.gz']
    self.functions    = ['interface']
    self.includes     = [] #Chaco does not have an include file
    self.needsMath    = 1
    self.liblist      = [['libchaco.a']]
    self.license      = 'http://www.cs.sandia.gov/~web9200/9200_download.html'
    return

  def Install(self):
    # Get the Chaco directories
    chacoDir = self.getDir()
    
    # Configure and Build Chaco
    g = open(os.path.join(chacoDir,'make.inc'),'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS = '+self.setCompilers.getCompilerFlags()+'\n')
    g.write('OFLAGS = '+self.setCompilers.getCompilerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.close()
    
    if self.installNeeded('make.inc'):
      try:
        self.logPrintBox('Compiling chaco; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+chacoDir+';CHACO_INSTALL_DIR='+self.installDir+';export CHACO_INSTALL_DIR; cd code; make clean; make; cd '+self.installDir+'; '+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+' '+self.libdir+'/libchaco.a `find '+chacoDir+'/code -name "*.o"`; cd '+self.libdir+'; ar d libchaco.a main.o', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on CHACO: '+str(e))
      self.checkInstall(output,'make.inc')
    return self.installDir
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()

