#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/PARTY_1.99.tar.gz']
    self.functions    = ['party_lib']
    self.includes     = ['party_lib.h']
    self.liblist      = [['libparty.a']]
    self.license      = 'http://wwwcs.upb.de/fachbereich/AG/monien/RESEARCH/PART/party.html'
    return

  def Install(self):
    # Get the PARTY directories
    partyDir = self.getDir()
    
    # Configure and Build PARTY
    g = open(os.path.join(partyDir,'make.inc'),'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC = '+self.setCompilers.getCompiler()+' '+self.setCompilers.getCompilerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.close()
    
    if not os.path.isfile(os.path.join(self.confDir,'PARTY')) or not (self.getChecksum(os.path.join(self.confDir,'PARTY')) == self.getChecksum(os.path.join(partyDir,'make.inc'))):
      self.framework.log.write('Have to rebuild Party, make.inc != '+self.confDir+'/PARTY\n')
      try:
        self.logPrintBox('Compiling party; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(partyDir,'src')+'; PARTY_INSTALL_DIR='+self.installDir+';export PARTY_INSTALL_DIR; make clean; make all; cd ..; mv *.a '+os.path.join(self.installDir,self.libdir)+'/.; cp party_lib.h '+os.path.join(self.installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on PARTY: '+str(e))
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
