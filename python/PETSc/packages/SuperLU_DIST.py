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
    self.mpi          = self.framework.require('PETSc.packages.MPI',self)
    self.blasLapack   = self.framework.require('PETSc.packages.BlasLapack',self)
    self.download     = ['http://crd.lbl.gov/~xiaoye/SuperLU/superlu_dist_2.0.tar.gz']
    self.deps         = [self.mpi,self.blasLapack]
    self.functions    = ['set_default_options_dist']
    self.includes     = ['superlu_ddefs.h']
    self.libdir       = ''
    self.includedir   = 'SRC'
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
    alllibs = []
    alllibs.append(os.path.join(dir,'superlu.a'))
    return alllibs
                
  def Install(self):
    # Get the SUPERLU_DIST directories
    superluDir = self.getDir()
    installDir = os.path.join(superluDir, self.arch.arch)
    
    # Configure and Build SUPERLU_DIST
    if os.path.isfile(os.path.join(superluDir,'make.inc')):
      output  = config.base.Configure.executeShellCommand('cd '+superluDir+'; rm -f make.inc', timeout=2500, log = self.framework.log)[0]
    g = open(os.path.join(superluDir,'make.inc'),'w')
    g.write('DSUPERLULIB  = ../superlu.a\n')
    g.write('BLASDEF      = -DUSE_VENDOR_BLAS\n')
    g.write('BLASLIB      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('IMPI         = -I'+self.libraries.toString(self.mpi.include)+'\n')
    g.write('MPILIB       = '+self.libraries.toString(self.mpi.lib)+'\n')
    g.write('SYS_LIB      = \n')
    g.write('LIBS         = $(DSUPERLULIB) $(BLASLIB) $(PERFLIB) $(MPILIB) $(SYS_LIB)\n')
    g.write('ARCH         = '+self.setcompilers.AR+'\n')
    g.write('ARCHFLAGS    = '+self.setcompilers.AR_FLAGS+'\n')
    g.write('RANLIB       = '+self.setcompilers.RANLIB+'\n')
    self.setcompilers.pushLanguage('C')
    g.write('CC           = '+self.setcompilers.getCompiler()+'\n')
    g.write('CFLAGS       = '+self.setcompilers.getCompilerFlags()+'\n')
    g.write('LOADER       = '+self.setcompilers.getCompiler()+'\n') 
    g.write('LOADOPTS     = \n')
    self.setcompilers.popLanguage()
    self.setcompilers.pushLanguage('FC')
    g.write('FORTRAN      = '+self.setcompilers.getCompiler()+'\n')
    g.write('FFLAGS       = '+self.setcompilers.getCompilerFlags()+'\n')
    self.setcompilers.popLanguage()
    g.write('CDEFS        = -DAdd_\n')
    g.close()
    if not os.path.isdir(installDir):
      os.mkdir(installDir)
    if not os.path.isfile(os.path.join(installDir,'make.inc')) or not (self.getChecksum(os.path.join(installDir,'make.inc')) == self.getChecksum(os.path.join(superluDir,'make.inc'))):  
      self.framework.log.write('Have to rebuild SUPERLU_DIST, make.inc != '+installDir+'/make.inc\n')
      try:
        self.logPrint("Compiling superlu_dist; this may take several minutes\n", debugSection='screen')
        output  = config.base.Configure.executeShellCommand('cd '+superluDir+';SUPERLU_DIST_INSTALL_DIR='+installDir+';export SUPERLU_DIST_INSTALL_DIR; make clean; make lib; mv *.a '+os.path.join(installDir,self.libdir)+'; mkdir '+os.path.join(installDir,self.includedir)+'; cp SRC/*.h '+os.path.join(installDir,self.includedir)+'/.', timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on SUPERLU_DIST: '+str(e))
      if not os.path.isdir(os.path.join(installDir,self.libdir)):
        self.framework.log.write('Error running make on SUPERLU_DIST   ******(libraries not installed)*******\n')
        self.framework.log.write('********Output of running make on SUPERLU_DIST follows *******\n')        
        self.framework.log.write(output)
        self.framework.log.write('********End of Output of running make on SUPERLU_DIST *******\n')
        raise RuntimeError('Error running make on SUPERLU_DIST, libraries not installed')
      
      output  = config.base.Configure.executeShellCommand('cp -f '+os.path.join(superluDir,'make.inc')+' '+installDir, timeout=5, log = self.framework.log)[0]
      self.framework.actions.addArgument(self.PACKAGE, 'Install', 'Installed SUPERLU_DIST into '+installDir)
    return self.getDir()

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging(framework.clArgs)
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
