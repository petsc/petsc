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
    self.libdir       = ''
    self.includedir   = ''
    self.liblist      = [['libparty.a']]
    self.license      = 'http://wwwcs.upb.de/fachbereich/AG/monien/RESEARCH/PART/party.html'
    return

  def Install(self):
    # Get the PARTY directories
    partyDir = self.getDir()
    installDir = os.path.join(partyDir, self.arch.arch)
    
    # Configure and Build PARTY
    if os.path.isfile(os.path.join(partyDir,'make.inc')):
      output = config.base.Configure.executeShellCommand('cd '+partyDir+'; rm -f make.inc', timeout=2500, log = self.framework.log)[0]
    g = open(os.path.join(partyDir,'make.inc'),'w')
    self.setcompilers.pushLanguage('C')
    g.write('CC = '+self.setcompilers.getCompiler()+' '+self.setcompilers.getCompilerFlags()+'\n')
    self.setcompilers.popLanguage()
    g.close()
    
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(os.path.join(installDir,'make.inc')) or not (self.getChecksum(os.path.join(installDir,'make.inc')) == self.getChecksum(os.path.join(partyDir,'make.inc'))):
      self.framework.log.write('Have to rebuild Party, make.inc != '+installDir+'/make.inc\n')
      try:
        self.logPrintBox('Compiling party; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+os.path.join(partyDir,'src')+'; PARTY_INSTALL_DIR='+installDir+';export PARTY_INSTALL_DIR; make clean; make all; cd ..; mv *.a '+os.path.join(installDir,self.libdir)+'/.; cp party_lib.h '+os.path.join(installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on PARTY: '+str(e))
      if not os.path.isfile(os.path.join(installDir,self.libdir,'libparty.a')):
        self.framework.log.write('Error running make on PARTY   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on PARTY follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on PARTY *******\n')
        raise RuntimeError('Error running make on PARTY, libraries not installed')

      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(partyDir,'make.inc')+' '+installDir, timeout=5, log = self.framework.log)[0]
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed PARTY into '+installDir)
    return self.getDir()

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
