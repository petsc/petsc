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
    self.framework.log.write('umfpackDir = '+self.packageDir+' installDir '+self.installDir+'\n')

    mkfile = 'UFconfig/UFconfig.mk'
    g = open(os.path.join(self.packageDir, mkfile), 'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    g.write('CFLAGS       = '+self.setCompilers.getCompilerFlags()+''' -DUF_long="long long" -DUF_long_max=LONG_LONG_MAX -DUF_long_id='"%lld"' \n''')
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
    if self.installNeeded(mkfile):
      try:
        self.logPrintBox('Compiling UMFPACK; this may take several minutes')
        output = config.base.Configure.executeShellCommand('cd '+self.packageDir+'/UMFPACK; UMFPACK_INSTALL_DIR='+self.installDir+'''/lib; export UMFPACK_INSTALL_DIR; make; make clean; mv Lib/*.a '''+self.libDir+'; cp Include/*.h '+self.includeDir+'; cd ..; cp UFconfig/*.h '+self.includeDir+'; cd AMD; mv Lib/*.a '+self.libDir+'; cp Include/*.h '+self.includeDir, timeout=2500, log = self.framework.log)[0]
      except RuntimeError, e:
        raise RuntimeError('Error running make on UMFPACK: '+str(e))
      self.checkInstall(output, mkfile)
    return self.installDir
