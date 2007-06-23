#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download  = ['ftp://ftp.mcs.anl.gov/pub/petsc/externalpackages/UMFPACK-5.1.0.tar.gz']
    self.liblist   = [['libumfpack.a','libamd.a']]
    self.functions = ['umfpack_di_report_info'] 
    self.includes  = ['umfpack.h']
    self.complex   = 1
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.blasLapack = framework.require('config.packages.BlasLapack',self)
    self.deps       = [self.blasLapack]
    return
          
  def Install(self):
        
    umfpackDir = self.getDir() #~UMFPACK-v#
    installDir = os.path.join(self.defaultInstallDir, self.arch) #$PETSC_DIR/$PETESC_ARCH
    confDir    = os.path.join(installDir, 'conf')  #$PETSC_DIR/$PETSC_ARCH/conf
    incDir     = os.path.join(installDir,'include')
    libDir     = os.path.join(installDir,'lib')    
    self.framework.log.write('umfpackDir = '+umfpackDir+' installDir '+installDir+'\n')

    g = open(os.path.join(umfpackDir,'UFconfig/UFconfig.mk'),'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS       = '+self.setCompilers.getCompilerFlags()+'\n')
    self.setCompilers.popLanguage()
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    g.write('AR = ar cr\n')
    g.write('RM = rm -f\n')
    g.write('MV = mv -f\n')
    g.write('BLAS      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    g.write('UMFPACK_CONFIG =\n')
    g.write('CLEAN = *.o *.obj *.ln *.bb *.bbg *.da *.tcov *.gcov gmon.out *.bak *.d\n')
    g.close()
    
    # Build UMFPACK
    try:
      self.logPrintBox('Compiling UMFPACK; this may take several minutes')
      output = config.base.Configure.executeShellCommand('cd '+umfpackDir+'/UMFPACK; UMFPACK_INSTALL_DIR='+installDir+'/lib; export UMFPACK_INSTALL_DIR; make; make clean; mv Lib/*.a '+libDir+'; cp Include/*.h '+incDir+'; cd ..; cp UFconfig/*.h '+incDir+'; cd AMD; mv Lib/*.a '+libDir+'; cp Include/*.h '+incDir, timeout=2500, log = self.framework.log)[0]
    
    except RuntimeError, e:
      raise RuntimeError('Error running make on UMFPACK: '+str(e))
    return installDir
