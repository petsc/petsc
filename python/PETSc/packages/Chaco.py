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
    self.includedir   = ''
    self.needsMath    = 1
    self.liblist      = [['libchaco.a']]
    self.license      = 'http://www.cs.sandia.gov/~web9200/9200_download.html'
    return

  def Install(self):
    # Get the Chaco directories
    chacoDir = self.getDir()
    installDir = os.path.join(chacoDir, self.arch.arch)
    
    # Configure and Build Chaco
    if os.path.isfile(os.path.join(chacoDir,'make.inc')):
      output = config.base.Configure.executeShellCommand('cd '+chacoDir+'; rm -f make.inc', timeout=2500, log = self.framework.log)[0]
    g = open(os.path.join(chacoDir,'make.inc'),'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS = '+self.setCompilers.getCompilerFlags()+'\n')
    g.write('OFLAGS = '+self.setCompilers.getCompilerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.close()
    
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(os.path.join(installDir,'make.inc')) or not (self.getChecksum(os.path.join(installDir,'make.inc')) == self.getChecksum(os.path.join(chacoDir,'make.inc'))):
      self.framework.log.write('Have to rebuild Chaco, make.inc != '+installDir+'/make.inc\n')
      try:
        self.logPrintBox('Compiling chaco; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+chacoDir+';CHACO_INSTALL_DIR='+installDir+';export CHACO_INSTALL_DIR; cd code; make clean; make; cd '+installDir+'; mkdir '+self.libdir+'; ar cr '+self.libdir+'/libchaco.a `find ../code -name "*.o"`; cd '+self.libdir+'; ar d libchaco.a main.o', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on CHACO: '+str(e))
      if not os.path.isfile(os.path.join(installDir,self.libdir,'libchaco.a')):
        self.framework.log.write('Error running make on CHACO   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on CHACO follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on CHACO *******\n')
        raise RuntimeError('Error running make on CHACO, libraries not installed')
      
      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(chacoDir,'make.inc')+' '+installDir, timeout=5, log = self.framework.log)[0]
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed CHACO into '+installDir)
    return self.getDir()
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()

