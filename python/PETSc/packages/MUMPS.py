#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os
import PETSc.package
import md5

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
    self.functions    = ['dmumps_c']
    self.includes     = ['dmumps_c.h']
    return

  def getChecksum(self,source, chunkSize = 1024*1024):  
    '''Return the md5 checksum for a given file, which may also be specified by its filename
       - The chunkSize argument specifies the size of blocks read from the file'''
    if isinstance(source, file):
      f = source
    else:
      f = file(source)
    m = md5.new()
    size = chunkSize
    buf  = f.read(size)
    while buf:
      m.update(buf)
      buf = f.read(size)
    f.close()
    return m.hexdigest()

  def generateLibList(self,dir):
    libs = ['cmumps','dmumps','smumps','zmumps','pord']
    alllibs = []
    for l in libs:
      alllibs.append('lib'+l+'.a')
    # Now specify -L mumps-lib-path only to the first library
    alllibs[0] = os.path.join(dir,alllibs[0])
    return alllibs
        
  def Install(self):
    # Get the MUMPS directories
    mumpsDir = self.getDir()
    installDir = os.path.join(mumpsDir, self.arch.arch)
    
    # Configure and Build MUMPS
    g = open(os.path.join(mumpsDir,'Makefile.inc'),'w')
    g.write('LPORDDIR   = ../PORD/lib/\n')
    args = ['LPORDDIR   = ../PORD/lib/']
    g.write('IPORD      = -I../PORD/include/\n')
    args.append('LPORD      = -L$(LPORDDIR) -lpord')
    g.write('LPORD      = -L$(LPORDDIR) -lpord\n')
    args.append('LPORD      = -L$(LPORDDIR) -lpord')
    g.write('ORDERINGSF = -Dpord\n')
    args.append('ORDERINGSF = -Dpord')
    g.write('ORDERINGSC = $(ORDERINGSF)\n')
    args.append('ORDERINGSC = $(ORDERINGSF)')
    g.write('LORDERINGS = $(LMETIS) $(LPORD) $(LSCOTCH)\n')
    args.append('LORDERINGS = $(LMETIS) $(LPORD) $(LSCOTCH)')
    g.write('IORDERINGS = $(IMETIS) $(IPORD) $(ISCOTCH)\n')
    args.append('IORDERINGS = $(IMETIS) $(IPORD) $(ISCOTCH)')
    g.write('RM = /bin/rm -f\n')
    args.append('RM = /bin/rm -f')
    self.setcompilers.pushLanguage('C')
    g.write('CC = '+self.setcompilers.getCompiler()+'\n')
    args.append('CC = '+self.setcompilers.getCompiler())
    self.setcompilers.popLanguage()
    if not self.compiler.fortranIsF90:
      raise RuntimeError('Invalid F90 compiler') 
    self.setcompilers.pushLanguage('FC') 
    g.write('FC = '+self.setcompilers.getCompiler()+'\n')
    args.append('FC = '+self.setcompilers.getCompiler())
    g.write('FL = '+self.setcompilers.getCompiler()+'\n')
    args.append('FL = '+self.setcompilers.getCompiler())
    self.setcompilers.popLanguage()
    
    g.write('AR      = ar vr\n')
    args.append('AR      = ar vr')
    g.write('RANLIB  = '+self.setcompilers.RANLIB+'\n') 
    args.append('RANLIB  = '+self.setcompilers.RANLIB)
    g.write('SCALAP  = '+self.libraries.toString(self.scalapack.lib)+' '+self.libraries.toString(self.blacs.lib)+'\n')
    args.append('SCALAP  = '+self.libraries.toString(self.scalapack.lib)+' '+self.libraries.toString(self.blacs.lib))
    g.write('INCPAR  = -I'+self.libraries.toString(self.mpi.include)+'\n')
    args.append('INCPAR  = -I'+self.libraries.toString(self.mpi.include))
    g.write('LIBPAR  = $(SCALAP) '+self.libraries.toString(self.mpi.lib)+'\n') #PARALLE LIBRARIES USED by MUMPS
    args.append('LIBPAR  = $(SCALAP) '+self.libraries.toString(self.mpi.lib))
    g.write('INCSEQ  = -I../libseq\n')
    args.append('INCSEQ  = -I../libseq')
    g.write('LIBSEQ  =  $(LAPACK) -L../libseq -lmpiseq\n')
    args.append('LIBSEQ  =  $(LAPACK) -L../libseq -lmpiseq')
    g.write('LIBBLAS = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    args.append('LIBBLAS = '+self.libraries.toString(self.blasLapack.dlib))
    g.write('CDEFS   = -DAdd_\n')
    args.append('CDEFS   = -DAdd_')
    g.write('OPTF    = -O\n')
    args.append('OPTF    = -O')
    g.write('OPTL    = -O\n')
    args.append('OPTL    = -O')
    g.write('OPTC    = -O -I.\n')
    args.append('OPTC    = -O -I.')
    g.write('INC = $(INCPAR)\n')
    args.append('INC = $(INCPAR)')
    g.write('LIB = $(LIBPAR)\n')
    args.append('LIB = $(LIBPAR)')
    g.write('LIBSEQNEEDED =\n')
    args.append('LIBSEQNEEDED =')
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
        self.logPrint("Compiling Mumps; this may take several minutes\n", debugSection='screen')
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

    fd   = file(os.path.join(installDir,'config.args'), 'w') #this is ugly, rm???
    args = ' '.join(args)
    fd.write(args)
    fd.close()
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

