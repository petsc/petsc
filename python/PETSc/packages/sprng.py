#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/sprng-1.0.tar.gz']
    self.functions = ['make_new_seed_mpi'] 
    self.includes  = ['sprng.h'] 
    self.liblist   = [['libcmrg.a','liblcg64.a','liblcg.a','liblfg.a','libmlfg.a']]
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.deps       = [self.mpi]
    return

  def Install(self):    
    # Get the sprng directories
    sprngDir = self.getDir()  #~sprng-1.0
    srcDir = os.path.join(sprngDir,'SRC') #~sprng-1.0/SRC
    
    # Configure and Build sprng
    g = open(os.path.join(srcDir,'make.PETSC'),'w')
    g.write('AR         = ar\n')
    g.write('ARFLAGS 	= cr\n')
    g.write('RANLIB 	= '+self.setCompilers.RANLIB+'\n')
    self.setCompilers.pushLanguage('C')	
    g.write('CC 	= '+self.setCompilers.getCompiler()+'\n')
    self.setCompilers.popLanguage()
    g.write('CLD 	 = $(CC)\n')
    g.write('MPICC  	 = $(CC)\n')
    g.write('MPIDEF      = -DSPRNG_MPI\n') #Only if you plan to use MPI
    g.write('MPILIB     = '+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('MPI_INCLUDE = '+self.headers.toString(self.mpi.include)+'\n')
    
    g.write('CFLAGS 	= -O3 -DLittleEndian $(PMLCGDEF) $(MPIDEF) -D$(PLAT) $(MPI_INCLUDE)\n')
    g.write('CLDFLAGS 	= -03\n')
    g.write('CPP 	= cpp -P\n')
    
    g.write('F77 	= echo\n')
    g.write('F77LD 	= $(F77)\n')
    g.write('FFXN 	= -DAdd_\n')
    g.write('FSUFFIX 	= F\n')
    g.write('MPIF77 	= echo\n')
    g.write('FFLAGS 	= -O3 $(PMLCGDEF) $(MPIDEF) -D$(PLAT) $(MPI_INCLUDE)\n')
    g.write('F77LDFLAGS = -O3\n')
    g.close()
    if not os.path.isfile(os.path.join(self.confDir,'sprng')) or not (self.getChecksum(os.path.join(self.confDir,'sprng')) == self.getChecksum(os.path.join(srcDir,'make.PETSC'))):  
      self.framework.log.write('Have to rebuild SPRNG, make.PETSC != '+self.installDir+'/make.PETSC\n')
      try:
        self.logPrintBox('Compiling SPRNG; this may take several minutes')
        output  = config.base.Configure.executeShellCommand('cd '+sprngDir+';SPRNG_INSTALL_DIR='+self.installDir+';export SPRNG_INSTALL_DIR; make realclean; cd SRC; make; cd ..;  cp lib/*.a '+os.path.join(self.installDir,self.libdir)+'; cp include/*.h '+os.path.join(self.installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SPRNG: '+str(e))
      self.checkInstall(output,'make.PETSC')
    return self.installDir

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
