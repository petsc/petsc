#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import config.package
import os

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.download   = ['http://downloads.sourceforge.net/exodusii/exodusii-4.75.tar.gz']
    self.liblist    = [['libexoIIv2for.a', 'libexoIIv2c.a']]
    self.functions  = ['ex_close'] 
    self.includes   = ['exodusII.h']
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.netcdf = framework.require('config.packages.NetCDF', self)
    self.deps   = [self.netcdf]
    return
          
  def Install(self):
    self.framework.log.write('exodusIIDir = '+self.packageDir+' installDir '+self.installDir+'\n')

    mkfile = 'UFconfig/UFconfig.mk'
    g = open(os.path.join(self.packageDir, mkfile), 'w')
    self.setCompilers.pushLanguage('C')
    g.write('CC           = '+self.setCompilers.getCompiler()+'\n')
    if self.checkCompile('#ifdef PETSC_HAVE_LIMITS_H\n  #include <limits.h>\n#endif\n', 'long long i=ULONG_MAX;\n\nif (i);\n'):
      ulong_max = 'ULONG_MAX'
    else:
      ulong_max = '9223372036854775807LL'
    g.write('CFLAGS       = '+self.setCompilers.getCompilerFlags()+''' -DUF_long="long long" -DUF_long_max=''' + ulong_max + ''' -DUF_long_id='"%lld"' \n''')
    self.setCompilers.popLanguage()
    g.write('RANLIB       = '+self.setCompilers.RANLIB+'\n')
    g.write('AR = ar cr\n')
    g.write('RM = rm -f\n')
    g.write('MV = mv -f\n')
    g.write('BLAS      = '+self.libraries.toString(self.blasLapack.dlib)+'\n')
    if self.blasLapack.mangling == 'underscore':
      flg = ''
    elif self.blasLapack.mangling == 'caps':
      flg = '-DBLAS_CAPS_DOES_NOT_WORK'
    else:
      flg = '-DBLAS_NO_UNDERSCORE'
    g.write('UMFPACK_CONFIG   = '+flg+'\n')
    g.write('CLEAN = *.o *.obj *.ln *.bb *.bbg *.da *.tcov *.gcov gmon.out *.bak *.d\n')
    g.close()
    
    # Build UMFPACK
    if self.installNeeded(mkfile):
      try:
        self.logPrintBox('Compiling UMFPACK; this may take several minutes')
        output,err,ret = config.base.Configure.executeShellCommand('cd '+self.packageDir+'/UMFPACK; UMFPACK_INSTALL_DIR='+self.installDir+'''/lib; export UMFPACK_INSTALL_DIR; make; make clean; mv Lib/*.a '''+self.libDir+'; cp Include/*.h '+self.includeDir+'; cd ..; cp UFconfig/*.h '+self.includeDir+'; cd AMD; mv Lib/*.a '+self.libDir+'; cp Include/*.h '+self.includeDir, timeout=2500, log = self.framework.log)
      except RuntimeError, e:
        raise RuntimeError('Error running make on UMFPACK: '+str(e))
      self.checkInstall(output+err, mkfile)
    return self.installDir
