#!/usr/bin/env python
import PETSc.package
import config.base

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download     = ['hg://petsc.cs.iit.edu/petsc/ParMetis-dev','ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/ParMetis-dev-p1.tar.gz']
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
    # Get the ParMetis directories
    parmetisDir    = self.getDir()
    installDir     = os.path.join(self.petscdir.dir,self.arch.arch)
    confDir        = os.path.join(self.petscdir.dir,self.arch.arch,'conf')
    makeinc        = os.path.join(parmetisDir,'make.inc')
    installmakeinc = os.path.join(confDir,'PARMETIS')
    configheader   = os.path.join(parmetisDir,'ParMETISLib','configureheader.h')

    # Configure ParMetis 
    if os.path.isfile(makeinc):
      os.unlink(makeinc)
    g = open(makeinc,'w')
    g.write('SHELL          = '+self.programs.SHELL+'\n')
    g.write('CP             = '+self.programs.cp+'\n')
    g.write('RM             = '+self.programs.RM+'\n')
    g.write('MKDIR          = '+self.programs.mkdir+'\n')

    g.write('AR             = '+self.setCompilers.AR+'\n')
    g.write('ARFLAGS        = '+self.setCompilers.AR_FLAGS+'\n')
    g.write('AR_LIB_SUFFIX  = '+self.setCompilers.AR_LIB_SUFFIX+'\n')
    g.write('RANLIB         = '+self.setCompilers.RANLIB+'\n')

    g.write('PARMETIS_ROOT  = '+parmetisDir+'\n')
    g.write('PREFIX         = '+installDir+'\n')
    g.write('METISLIB       = $(PARMETIS_ROOT)/libmetis.$(AR_LIB_SUFFIX)\n')
    g.write('PARMETISLIB    = $(PARMETIS_ROOT)/libparmetis.$(AR_LIB_SUFFIX)\n')
    
    self.setCompilers.pushLanguage('C')
    cflags = self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','')
    cflags += ' ' + self.headers.toString(self.mpi.include)+' '+self.headers.toString('.')
        
    g.write('CC             = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS         = '+cflags)
    self.setCompilers.popLanguage()
    g.close()

    # Now compile & install
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    
    if not os.path.isfile(installmakeinc) or not (self.getChecksum(installmakeinc) == self.getChecksum(makeinc)):
      self.framework.log.write('Have to rebuild ParMetis, make.inc != '+installmakeinc+'\n')
      self.framework.outputHeader(configheader)
      try:
        self.logPrintBox('Compiling & installing Parmetis; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+parmetisDir+'; make clean; make lib; make minstall; make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on ParMetis: '+str(e))
    else:
      self.framework.log.write('Did not need to compile downloaded ParMetis\n')
    output  = config.base.Configure.executeShellCommand('cp -f '+makeinc+' '+installmakeinc, timeout=5, log = self.framework.log)[0]
    self.framework.actions.addArgument('ParMetis', 'Install', 'Installed ParMetis into '+installDir)
    return installDir
  
if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
