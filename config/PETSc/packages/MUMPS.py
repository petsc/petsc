#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/MUMPS_4.7.3.tar.gz']
    self.liblist   = [['libcmumps.a','libdmumps.a','libsmumps.a','libzmumps.a','libpord.a'],
                      ['libcmumps.a','libdmumps.a','libsmumps.a','libzmumps.a','libpord.a','libpthread.a']]
    self.functions = ['dmumps_c']
    self.includes  = ['dmumps_c.h']
    self.complex   = 1
    return

  def setupDependencies(self, framework):
    PETSc.package.Package.setupDependencies(self, framework)
    self.mpi        = framework.require('config.packages.MPI',self)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.blacs      = framework.require('PETSc.packages.blacs',self)
    self.scalapack  = framework.require('PETSc.packages.SCALAPACK',self)
    self.parmetis   = framework.require('PETSc.packages.ParMetis',self)
    self.deps       = [self.parmetis,self.scalapack,self.blacs,self.mpi,self.blasLapack]
    return
        
  def Install(self):

    g = open(os.path.join(self.packageDir,'Makefile.inc'),'w')
    g.write('LPORDDIR   = ../PORD/lib/\n')
    g.write('IPORD      = -I../PORD/include/\n')
    g.write('LPORD      = -L$(LPORDDIR) -lpord\n')
    g.write('IMETIS =\n')
    g.write('LMETIS = '+self.libraries.toString(self.parmetis.lib)+'\n') 

    # Disable threads on BGL
    if self.libraryOptions.isBGL():
      g.write('ORDERINGSC = -DWITHOUT_PTHREAD -Dmetis -Dpord\n')
    else:
      g.write('ORDERINGSC = -Dmetis -Dpord\n')
    if self.compilers.FortranDefineCompilerOption:
      g.write('ORDERINGSF = '+self.compilers.FortranDefineCompilerOption+'metis'+' -Dpord\n')
    else:
      raise RuntimeError('Fortran compiler cannot handle preprocessing directives from command line.')     
    g.write('LORDERINGS = $(LMETIS) $(LPORD) $(LSCOTCH)\n')
    g.write('IORDERINGS = $(IMETIS) $(IPORD) $(ISCOTCH)\n')

    g.write('RM = /bin/rm -f\n')
    self.setCompilers.pushLanguage('C')
    g.write('CC = '+self.setCompilers.getCompiler()+'\n')
    g.write('OPTC    = ' + self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','') +'\n')
    self.setCompilers.popLanguage()
    if not self.compilers.fortranIsF90:
      raise RuntimeError('Installing MUMPS requires a F90 compiler') 
    self.setCompilers.pushLanguage('FC') 
    g.write('FC = '+self.setCompilers.getCompiler()+'\n')
    g.write('FL = '+self.setCompilers.getCompiler()+'\n')
    g.write('OPTF    = ' + self.setCompilers.getCompilerFlags().replace('-Wall','').replace('-Wshadow','') +'\n')
    self.setCompilers.popLanguage()

    # set fortran name mangling
    # this mangling information is for both BLAS and the Fortran compiler so cannot use the BlasLapack mangling flag    
    if self.compilers.fortranManglingDoubleUnderscore:
      g.write('CDEFS   = -DAdd__\n')
    elif self.compilers.fortranMangling == 'underscore':
      g.write('CDEFS   = -DAdd_\n')
    elif self.compilers.fortranMangling == 'caps':
      g.write('CDEFS   = -DUPPPER\n')

    g.write('AR      = '+self.setCompilers.AR+' '+self.setCompilers.AR_FLAGS+'\n')
    g.write('RANLIB  = '+self.setCompilers.RANLIB+'\n') 
    g.write('SCALAP  = '+self.libraries.toString(self.scalapack.lib)+' '+self.libraries.toString(self.blacs.lib)+'\n')
    g.write('INCPAR  = '+self.headers.toString(self.mpi.include)+'\n')
    g.write('LIBPAR  = $(SCALAP) '+self.libraries.toString(self.mpi.lib)+'\n') #PARALLE LIBRARIES USED by MUMPS
    g.write('INCSEQ  = -I../libseq\n')
    g.write('LIBSEQ  =  $(LAPACK) -L../libseq -lmpiseq\n')
    g.write('LIBBLAS = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('OPTL    = -O -I.\n')
    g.write('INC = $(INCPAR)\n')
    g.write('LIB = $(LIBPAR)\n')
    g.write('LIBSEQNEEDED =\n')
    g.close()
    if self.installNeeded('Makefile.inc'):
      try:
        output  = config.base.Configure.executeShellCommand('cd '+self.packageDir+';make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        pass
      try:
        self.logPrintBox('Compiling Mumps; this may take several minutes')
        output = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; make all',timeout=2500, log = self.framework.log)[0]
        libDir     = os.path.join(self.installDir, self.libdir)
        includeDir = os.path.join(self.installDir, self.includedir)
        output = config.base.Configure.executeShellCommand('cd '+self.packageDir+'; mv lib/*.* '+libDir+'/.; cp include/*.* '+includeDir+'/.;', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on MUMPS: '+str(e))
      self.checkInstall(output,'Makefile.inc')
    return self.installDir

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()

