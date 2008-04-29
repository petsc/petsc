#!/usr/bin/env python
import PETSc.package
import config.base

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['hg://petsc.cs.iit.edu/petsc/ParMetis-dev','ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/ParMetis-dev-p2.tar.gz']
    self.functions    = ['ParMETIS_V3_PartKway']
    self.includes     = ['parmetis.h']
    self.liblist      = [['libparmetis.a','libmetis.a']]
    self.needsMath    = 1
    self.complex      = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi = framework.require('config.packages.MPI', self)
    self.deps = [self.mpi]
    return

  def Install(self):
    import os
    import sys

    makeinc        = os.path.join(self.packageDir,'make.inc')
    installmakeinc = os.path.join(self.confDir,'ParMetis')
    metisconfigheader    = os.path.join(self.packageDir,'METISLib','configureheader.h')
    parmetisconfigheader = os.path.join(self.packageDir,'ParMETISLib','configureheader.h')
    
    # Configure ParMetis 
    g = open(makeinc,'w')
    g.write('SHELL          = '+self.programs.SHELL+'\n')
    g.write('CP             = '+self.programs.cp+'\n')
    g.write('RM             = '+self.programs.RM+'\n')
    g.write('MKDIR          = '+self.programs.mkdir+'\n')

    g.write('AR             = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS        = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX  = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB         = '+self.setCompilers.RANLIB+'\n')

    g.write('PARMETIS_ROOT  = '+self.packageDir+'\n')
    g.write('PREFIX         = '+self.installDir+'\n')
    g.write('METISLIB       = $(PARMETIS_ROOT)/libmetis.$(AR_LIB_SUFFIX)\n')
    g.write('PARMETISLIB    = $(PARMETIS_ROOT)/libparmetis.$(AR_LIB_SUFFIX)\n')
    
    self.setCompilers.pushLanguage('C')
    cflags = self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')
    cflags += ' ' + self.headers.toString(self.mpi.include)+' '+self.headers.toString('.')
        
    g.write('CC             = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS         = '+cflags)
    self.setCompilers.popLanguage()
    g.close()

    if self.installNeeded('make.inc'):    # Now compile & install
      self.framework.outputHeader(metisconfigheader)
      self.framework.outputHeader(parmetisconfigheader)
      try:
        self.logPrintBox('Compiling & installing Parmetis; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; make clean; make lib; make minstall; make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on ParMetis: '+str(e))
      self.checkInstall(output,'make.inc')
    return self.installDir
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
