#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package

class Configure(PETSc.package.Package):
  def __init__(self, framework):
    PETSc.package.Package.__init__(self, framework)
    self.compiler     = self.framework.require('config.compilers',self)
    self.mpi          = self.framework.require('PETSc.packages.MPI',self)
    self.blasLapack   = self.framework.require('PETSc.packages.BlasLapack',self)
    self.blacs        = self.framework.require('PETSc.packages.blacs',self)
    self.scalapack    = self.framework.require('PETSc.packages.SCALAPACK',self)
    self.download     = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/MUMPS_4.3.2.tar.gz']
    self.deps         = [self.scalapack,self.blacs,self.mpi,self.blasLapack]
    self.liblist      = ['libcmumps.a','libdmumps.a','libsmumps.a','libzmumps.a','libpord.a']
    self.functions    = ['dmumps_c']
    self.includes     = ['dmumps_c.h']
    return
        
  def Install(self):
    # Get the MUMPS directories
    mumpsDir = self.getDir()
    installDir = os.path.join(mumpsDir, self.arch.arch)
    
    # Configure and Build MUMPS
    if os.path.isfile(os.path.join(mumpsDir,'Makefile.inc')):
      output  = config.base.Configure.executeShellCommand('cd '+mumpsDir+'; rm -f Makefile.inc', timeout=2500, log = self.framework.log)[0]
    g = open(os.path.join(mumpsDir,'Makefile.inc'),'w')
    g.write('LPORDDIR   = ../PORD/lib/\n')
    g.write('IPORD      = -I../PORD/include/\n')
    g.write('LPORD      = -L$(LPORDDIR) -lpord\n')
    g.write('ORDERINGSF = -Dpord\n')
    g.write('ORDERINGSC = $(ORDERINGSF)\n')
    g.write('LORDERINGS = $(LMETIS) $(LPORD) $(LSCOTCH)\n')
    g.write('IORDERINGS = $(IMETIS) $(IPORD) $(ISCOTCH)\n')
    g.write('RM = /bin/rm -f\n')
    self.setcompilers.pushLanguage('C')
    g.write('CC = '+self.setcompilers.getCompiler()+'\n')
    self.setcompilers.popLanguage()
    if not self.compiler.fortranIsF90:
      raise RuntimeError('Invalid F90 compiler') 
    self.setcompilers.pushLanguage('FC') 
    g.write('FC = '+self.setcompilers.getCompiler()+'\n')
    g.write('FL = '+self.setcompilers.getCompiler()+'\n')
    self.setcompilers.popLanguage()
    
    g.write('AR      = ar vr\n')
    g.write('RANLIB  = '+self.setcompilers.RANLIB+'\n') 
    g.write('SCALAP  = '+self.libraries.toString(self.scalapack.lib)+' '+self.libraries.toString(self.blacs.lib)+'\n')
    g.write('INCPAR  = -I'+self.libraries.toString(self.mpi.include)+'\n')
    g.write('LIBPAR  = $(SCALAP) '+self.libraries.toString(self.mpi.lib)+'\n') #PARALLE LIBRARIES USED by MUMPS
    g.write('INCSEQ  = -I../libseq\n')
    g.write('LIBSEQ  =  $(LAPACK) -L../libseq -lmpiseq\n')
    g.write('LIBBLAS = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('CDEFS   = -DAdd_\n')
    g.write('OPTF    = -O\n')
    g.write('OPTL    = -O\n')
    g.write('OPTC    = -O -I.\n')
    g.write('INC = $(INCPAR)\n')
    g.write('LIB = $(LIBPAR)\n')
    g.write('LIBSEQNEEDED =\n')
    g.close()
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(os.path.join(installDir,'Makefile.inc')) or not (self.getChecksum(os.path.join(installDir,'Makefile.inc')) == self.getChecksum(os.path.join(mumpsDir,'Makefile.inc'))):
      self.framework.log.write('Have to rebuild MUMPS, Makefile.inc != '+installDir+'/Makefile.inc\n')
      try:
        output  = config.base.Configure.executeShellCommand('cd '+mumpsDir+';make clean', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        pass
      try:
        self.logPrintBox('Compiling Mumps; this may take several minutes')
        output = config.base.Configure.executeShellCommand('cd '+mumpsDir+'; make all',timeout=2500, log = self.framework.log)[0]
        libDir     = os.path.join(installDir, self.libdir)
        includeDir = os.path.join(installDir, self.includedir)
        if not os.path.isdir(libDir):
          os.mkdir(libDir)
        if not os.path.isdir(includeDir):
          os.mkdir(includeDir)        
        output = config.base.Configure.executeShellCommand('cd '+mumpsDir+'; mv lib/*.* '+libDir+'/.; cp include/*.* '+includeDir+'/.;', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on MUMPS: '+str(e))
    else:
      self.framework.log.write('Do not need to compile downloaded MUMPS\n')
    if not os.path.isfile(os.path.join(installDir,self.libdir,'libdmumps.a')):
      self.framework.log.write('Error running make on MUMPS   ******(libraries not installed)*******\n')
      self.framework.log.write('********Output of running make on MUMPS follows *******\n')        
      self.framework.log.write(output)
      self.framework.log.write('********End of Output of running make on MUMPS *******\n')
      raise RuntimeError('Error running make on MUMPS, libraries not installed')
    
    output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(mumpsDir,'Makefile.inc')+' '+installDir, timeout=5, log = self.framework.log)[0]

    self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed MUMPS into '+installDir)
    return self.getDir()

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setup()
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()

